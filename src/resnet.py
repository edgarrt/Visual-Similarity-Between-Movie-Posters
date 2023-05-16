import pytorch_lightning as pl
import torch
import torch.nn.init as init
from sklearn.preprocessing import normalize
from torch import nn, optim
from torchvision.models import resnet50

from losses import Proxy_Anchor
from utils import recall_at_k


class Resnet50(pl.LightningModule):
    def __init__(
        self,
        embedding_size,
        num_classes: int = None,
        pretrained=True,
        is_norm=True,
        bn_freeze=True,
        lr: float = 0.0001,
        weight_decay: float = 0.0001,
        margin: float = 0.1,
        alpha: int = 32,
        k: int = 1,
    ):
        super(Resnet50, self).__init__()
        self.validation_step_outputs = []
        self.num_classes = num_classes
        self.lr = lr
        self.margin = margin
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.k = k
        self.is_norm = is_norm
        self.embedding_size = embedding_size

        self.model = resnet50(pretrained)
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)
        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size)

        self._initialize_weights()

        self.criterion = Proxy_Anchor(
            nb_classes=self.num_classes,
            sz_embed=self.embedding_size,
            mrg=self.margin,
            alpha=self.alpha,
        )

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)
        else:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
                    m.weight.requires_grad_(True)
                    m.bias.requires_grad_(True)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)

        x = max_x + avg_x
        x = x.view(x.size(0), -1)
        x = self.model.embedding(x)

        if self.is_norm:
            x = self.l2_norm(x)

        return x

    def training_step(self, batch, batch_index):
        target = batch["targets"]
        im = batch["image"]

        predicted_y = self.forward(im.squeeze())

        loss = self.criterion(predicted_y, target.squeeze())
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_index):
        target = batch["targets"]
        im = batch["image"]

        predicted_y = self.forward(im.squeeze())

        val_loss = self.criterion(predicted_y, target.squeeze())
        self.validation_step_outputs.append(
            {"embeddings": predicted_y, "labels": target}
        )

        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, logger=True)

    def on_validation_epoch_end(self):
        embeddings = (
            torch.cat([x["embeddings"] for x in self.validation_step_outputs], dim=0)
            .cpu()
            .detach()
            .numpy()
        )
        labels = (
            torch.cat([x["labels"] for x in self.validation_step_outputs], dim=0)
            .cpu()
            .detach()
            .numpy()
        )

        embeddings_normalized = normalize(embeddings)

        # recall @ k
        recall_k = recall_at_k(embeddings_normalized, labels, self.num_classes, self.k)
        self.log(
            f"val_recall@{self.k}", recall_k, on_epoch=True, prog_bar=True, logger=True
        )
        self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_index):
        target = batch["targets"]
        im = batch["image"]

        predicted_y = self.forward(im.squeeze())
        test_loss = self.criterion(predicted_y, target.squeeze())
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        param_groups = [
            {
                "params": list(
                    set(self.model.parameters()).difference(
                        set(self.model.embedding.parameters())
                    )
                )
            },
            {"params": self.model.embedding.parameters()},
            {"params": self.criterion.parameters()},
        ]

        optimizer = optim.AdamW(
            param_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.25)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode="fan_out")
        init.constant_(self.model.embedding.bias, 0)
