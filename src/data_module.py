import multiprocessing

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset import ImageDataset
from utils import collate_batch


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str = None,
        test_dir: str = None,
        valid_dir: str = None,
        image_path: str = None,
        batch_size: int = 8,
        num_workers: int = multiprocessing.cpu_count(),
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_path = image_path

        self.train_df = ImageDataset(train_dir, image_path=image_path)
        self.test_df = ImageDataset(test_dir, image_path=image_path)
        self.valid_df = ImageDataset(valid_dir, image_path=image_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_df,
            collate_fn=collate_batch,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_df,
            collate_fn=collate_batch,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_df,
            collate_fn=collate_batch,
            batch_size=1,
            num_workers=self.num_workers,
        )
