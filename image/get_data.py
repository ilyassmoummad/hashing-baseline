import torch
from dataset import get_flickr25k_dataloaders, get_coco_dataloaders, get_nuswide_dataloaders
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader


def get_transforms():
    """Define the image transformations."""
    resize_size = 256
    crop_size = 224
    mean = torch.tensor((0.485, 0.456, 0.406)) # ImageNet mean
    std = torch.tensor((0.229, 0.224, 0.225)) # ImageNet std
    norm = T.Normalize(mean, std)

    return T.Compose([
        T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        norm,
    ])


def get_dataloaders(args):
    """Create and return data loaders for the specified dataset."""
    transform = get_transforms()

    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            root=args.data_dir,
            transform=transform,
            train=True,
            download=True
        )

        query_dataset = datasets.CIFAR10(
            root=args.data_dir,
            transform=transform,
            train=False,
            download=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.bs,
            num_workers=args.n_workers,
            pin_memory=False,
            drop_last=False,
        )

        query_loader = DataLoader(
            query_dataset,
            batch_size=args.bs,
            num_workers=args.n_workers,
            pin_memory=False,
            drop_last=False,
        )

        database_loader = train_loader

    elif args.dataset == 'flickr25k':
        train_loader, database_loader, query_loader = get_flickr25k_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.bs,
            num_workers=args.n_workers
        )

    elif args.dataset == 'nuswide':
        train_loader, database_loader, query_loader = get_nuswide_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.bs,
            num_workers=args.n_workers
        )

    elif args.dataset == 'coco':
        train_loader, database_loader, query_loader = get_coco_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.bs,
            num_workers=args.n_workers
        )

    return train_loader, database_loader, query_loader