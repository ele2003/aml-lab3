import os
import shutil

import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

# Cartella che contiene tiny-imagenet-200/ (scaricato da Colab, NON su GitHub)
DATA_ROOT = 'dataset/tiny-imagenet/tiny-imagenet-200'

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = T.Compose([
    T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


def prepare_val_split(root=DATA_ROOT):
    """Adjust the format of the val split to be used with ImageFolder."""
    val_dir = f'{root}/val'
    images_dir = f'{val_dir}/images'
    if not os.path.isdir(images_dir):  # già sistemato
        return

    with open(f'{val_dir}/val_annotations.txt') as f:
        for line in f:
            fn, cls, *_ = line.split('\t')
            os.makedirs(f'{val_dir}/{cls}', exist_ok=True)
            shutil.copyfile(f'{images_dir}/{fn}', f'{val_dir}/{cls}/{fn}')

    shutil.rmtree(images_dir)


def get_dataloaders(root=DATA_ROOT, batch_size=32, num_workers=8):
    prepare_val_split(root)

    # root/{classX}/x001.jpg
    train_dataset = ImageFolder(root=f'{root}/train', transform=transform)
    val_dataset = ImageFolder(root=f'{root}/val', transform=transform)

    print(f"Length of train dataset: {len(train_dataset)}")
    print(f"Length of val dataset: {len(val_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
