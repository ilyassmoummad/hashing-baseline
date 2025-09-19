import os
import requests
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch


def download_file_and_fix_paths(url: str, dest_dir: str) -> str:
    """
    Download a file from a URL into dest_dir.
    If file exists, reuse it.
    After downloading, fix any absolute or prefixed paths in the file by rewriting
    them as relative paths (relative to dest_dir).
    Returns the local filepath.
    """
    os.makedirs(dest_dir, exist_ok=True)
    filename = os.path.basename(url)
    filepath = os.path.join(dest_dir, filename)

    if not os.path.exists(filepath):
        print(f"[INFO] Downloading {url}")
        response = requests.get(url)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(response.content)

        _fix_paths_in_txt(filepath, dest_dir)
    else:
        print(f"[INFO] {filepath} already exists, skipping download")

    return filepath


def _fix_paths_in_txt(filepath: str, root_dir: str):
    """
    Fix absolute or prefixed paths inside a downloaded .txt split file by converting
    them to relative paths with respect to root_dir.
    This handles absolute paths and removes './' prefixes.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    fixed_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        path = parts[0]

        # Normalize path separators
        path = os.path.normpath(path)

        # Convert absolute paths to relative paths if possible
        if os.path.isabs(path):
            try:
                path = os.path.relpath(path, root_dir)
            except ValueError:
                # Different drive or can't relativize: fallback to basename
                path = os.path.basename(path)
        else:
            # Remove leading './' or '.\' if any
            while path.startswith(".{}" .format(os.sep)):
                path = path[2:]

        fixed_lines.append(" ".join([path] + parts[1:]) + "\n")

    with open(filepath, 'w') as f:
        f.writelines(fixed_lines)


def get_transforms():
    resize_size = 256
    crop_size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return T.Compose([
        T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


class BaseDataset(Dataset):
    def __init__(self, root, list_path, transform=None):
        self.root = root
        self.transform = transform
        with open(list_path, 'r') as f:
            lines = f.readlines()

        self.samples = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            rel_path = parts[0].lstrip("./")  # Remove leading ./ if any
            full_path = os.path.join(root, rel_path)
            if not os.path.exists(full_path):
                # fallback: try basename only
                full_path = os.path.join(root, os.path.basename(rel_path))
            labels = torch.tensor([int(x) for x in parts[1:]], dtype=torch.float32)
            self.samples.append((full_path, labels))

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.samples)


class NusWideDataset(BaseDataset):
    def __init__(self, root, list_path, transform=None):
        super().__init__(root, list_path, transform)


class CocoDataset(BaseDataset):
    def __init__(self, root, list_path, transform=None):
        super().__init__(root, list_path, transform)


class Flickr25KDataset(BaseDataset):
    def __init__(self, root, list_path, transform=None):
        super().__init__(root, list_path, transform)

class Imagenet100Dataset(BaseDataset):
    def __init__(self, root, list_path, transform=None):
        super().__init__(root, list_path, transform)


def get_flickr25k_dataloaders(data_dir, batch_size=32, num_workers=4):
    base_url = "https://raw.githubusercontent.com/swuxyj/DeepHash-pytorch/master/data/mirflickr/"
    train_txt = download_file_and_fix_paths(base_url + "train.txt", data_dir)
    database_txt = download_file_and_fix_paths(base_url + "database.txt", data_dir)
    test_txt = download_file_and_fix_paths(base_url + "test.txt", data_dir)

    transform = get_transforms()

    train_dataset = Flickr25KDataset(data_dir, train_txt, transform)
    database_dataset = Flickr25KDataset(data_dir, database_txt, transform)
    test_dataset = Flickr25KDataset(data_dir, test_txt, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    database_loader = DataLoader(database_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)
    query_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, database_loader, query_loader


def get_nuswide_dataloaders(data_dir, batch_size=32, num_workers=4):
    base_url = "https://raw.githubusercontent.com/swuxyj/DeepHash-pytorch/master/data/nuswide_21/"
    train_txt = download_file_and_fix_paths(base_url + "train.txt", data_dir)
    database_txt = download_file_and_fix_paths(base_url + "database.txt", data_dir)
    test_txt = download_file_and_fix_paths(base_url + "test.txt", data_dir)

    transform = get_transforms()

    train_dataset = NusWideDataset(data_dir, train_txt, transform)
    database_dataset = NusWideDataset(data_dir, database_txt, transform)
    test_dataset = NusWideDataset(data_dir, test_txt, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    database_loader = DataLoader(database_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)
    query_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, database_loader, query_loader


def get_coco_dataloaders(data_dir, batch_size=32, num_workers=4):
    base_url = "https://raw.githubusercontent.com/swuxyj/DeepHash-pytorch/master/data/coco/"
    train_txt = download_file_and_fix_paths(base_url + "train.txt", data_dir)
    database_txt = download_file_and_fix_paths(base_url + "database.txt", data_dir)
    test_txt = download_file_and_fix_paths(base_url + "test.txt", data_dir)

    transform = get_transforms()

    train_dataset = CocoDataset(data_dir, train_txt, transform)
    database_dataset = CocoDataset(data_dir, database_txt, transform)
    test_dataset = CocoDataset(data_dir, test_txt, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    database_loader = DataLoader(database_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)
    query_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, database_loader, query_loader