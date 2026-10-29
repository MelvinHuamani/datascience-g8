from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "Potato Leaf Disease Dataset"

IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def create_dataloaders(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    batch_size: int = 32,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, list[str]]:
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    validation_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    data_dir = Path(data_dir)
    base_dataset = datasets.ImageFolder(data_dir)
    train_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    validation_dataset = datasets.ImageFolder(data_dir, transform=validation_transform)

    indices = list(range(len(base_dataset)))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        stratify=base_dataset.targets,
    )

    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda,
    )
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )

    return train_loader, validation_loader, base_dataset.classes
