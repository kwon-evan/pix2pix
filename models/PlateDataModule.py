from typing import Optional
import os
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning import LightningDataModule


# Costum dataset 생성
class PlateDataset(Dataset):
    def __init__(self, data_dir, transform=None, direction='b2a'):
        super().__init__()
        self.direction = direction
        self.path2a = os.path.join(data_dir, 'a')
        self.path2b = os.path.join(data_dir, 'b')
        self.img_filenames = [x for x in os.listdir(self.path2a)]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                transforms.Resize((256, 256))
            ])
        else:
            self.transform = transform

    def __getitem__(self, index):
        a = Image.open(os.path.join(self.path2a, self.img_filenames[index])).convert('RGB')
        b = Image.open(os.path.join(self.path2b, self.img_filenames[index])).convert('RGB')

        if self.transform:
            a = self.transform(a)
            b = self.transform(b)

        if self.direction == 'b2a':
            return b, a
        else:
            return a, b

    def __len__(self):
        return len(self.img_filenames)


class PlateDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 batch_size: int,
                 num_workers: int,
                 transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

        self.train = None
        self.val = None
        self.test = None

    def prepare_data(self):
        import os.path
        if not os.path.exists(self.data_dir):
            assert "No Such Files or Dirs!"

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            plates_full = PlateDataset(self.data_dir, transform=self.transform)
            full_size = len(plates_full)
            train_size = int(full_size * 0.9)
            val_size = full_size - train_size
            self.train, self.val = random_split(plates_full, [train_size, val_size])

        if stage == "test" or stage is None:
            self.test = PlateDataset(self.data_dir, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)
