from PIL import Image
from pandas import read_csv
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from pathlib import Path
import os


class SegDataset(Dataset):
    def __init__(self, image_dir: str, csv_path: str, resize: tuple):
        super(SegDataset, self).__init__()
        self.image_dir = Path(image_dir)
        if not len(os.listdir(self.image_dir)):
            raise RuntimeError("No src file in your 'image_dir'.Please check your path!")
        self.csv_path = Path(csv_path)
        if not os.path.exists(self.csv_path):
            raise RuntimeError("DIF doesn't exist.")
        self.resize = resize
        self.transformer = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=0, std=255)
        ])
        self.df = read_csv(self.csv_path, encoding='utf-8')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 重新规范尺寸大小
        data = Image.open(self.df['img_path'][idx])
        mask = Image.open(self.df['mask_path'][idx])
        # 归一化
        data = self.transformer(data)
        mask = self.transformer(mask)
        assert data.shape[1:] == mask.shape[1:], "Inputs and outputs must be have the same size."
        return data, mask


def get_dataloader(image_dir, csv_path, resize, batch_size, num_workers, train_percent=0.9):
    dataset = SegDataset(image_dir, csv_path, resize)
    num_sample = len(dataset)
    num_train = int(train_percent * num_sample)
    num_valid = num_sample - num_train
    train_ds, valid_ds = random_split(dataset, [num_train, num_valid])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
                          persistent_workers=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                          persistent_workers=True)
    return train_dl, valid_dl, len(dataset), len(train_ds), len(valid_ds)
