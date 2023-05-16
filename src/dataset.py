from pathlib import Path

import cv2
import pandas as pd
import torchvision.transforms as transforms
from torch import tensor
from torch.utils.data import Dataset

# imagenet normalization
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class ImageDataset(Dataset):
    def __init__(self, _dir: str, image_path: str):
        super().__init__()
        self.image_path = image_path
        files = Path(_dir).glob("*.csv")
        dfs = [pd.read_csv(f, index_col=0, lineterminator="\n") for f in files]
        df = pd.concat(dfs, ignore_index=True)
        self.genres_single = df.iloc[:, 6].values
        self.image_names = df.iloc[:, 7].values
        self.genres = df.iloc[:, 8:].values

        self.transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image = cv2.imread(f"{self.image_path}{self.image_names[index]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image)
        sample = {
            "metadata": {
                "genre": self.genres_single[index],
                "image_name": self.image_names[index],
            },
            "image": image,
            "targets": tensor(self.genres[index]),
        }
        return sample
