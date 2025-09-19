import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from dataset import VocalSoundDataset, CREMADataset, GTZANDataset
import os


def get_dataloaders(args):
    """Create and return data loaders for the specified audio dataset."""

    os.makedirs(args.data_dir, exist_ok=True)

    if args.dataset == 'vocalsound':
        train_dataset = VocalSoundDataset(root_dir=args.data_dir, split="train")
        test_dataset = VocalSoundDataset(root_dir=args.data_dir, split="test")
        collate_fn = None

    elif args.dataset == 'cremad':
        train_dataset = CREMADataset(root_dir=args.data_dir, split="train")
        test_dataset = CREMADataset(root_dir=args.data_dir, split="test")
        collate_fn = None

    elif args.dataset == 'gtzan':
        train_dataset = GTZANDataset(root_dir=args.data_dir, folds=[i for i in range(4)])
        test_dataset = GTZANDataset(root_dir=args.data_dir, folds=[i for i in range(4, 10)])
        collate_fn = None

    elif args.dataset == 'esc50':
        if not os.listdir(args.data_dir):
            ds = load_dataset("ashraq/esc50")
            ds.save_to_disk(args.data_dir)
        else:
            ds = load_from_disk(args.data_dir)

        train_dataset = ds['train'].filter(lambda example: example['fold'] == 1) # 1st fold for PCA (or training)
        test_dataset = ds['train'].filter(lambda example: example['fold'] != 1) # Other folds for testing

        def collate_fn(batch):
            audios = [torch.tensor(item["audio"]["array"]) for item in batch]
            labels = [item["target"] for item in batch]

            audios = torch.stack(audios)
            labels = torch.tensor(labels)
            return audios, labels

    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        num_workers=args.n_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.bs,
        num_workers=args.n_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    from args import args
    train_loader, test_loader = get_dataloaders(args)
    print(len(train_loader), len(test_loader))