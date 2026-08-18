#########################################
# Step 1: Dataset loading in PyTorch
#########################################
import src.config as cfg
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

#training transforms (with augmentation)
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(cfg.IMAGENET_MEAN, cfg.IMAGENET_STD)
])

#Validation / Test transforms (NO augmentation, only resize + normalize)
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(cfg.IMAGENET_MEAN, cfg.IMAGENET_STD)
])

def get_dataloaders(batch_size=16, num_workers=2, seed=42):       
        """
        Loads dataset, applies transforms, splits into train/val/test,
        and returns dataloaders + class info.
        """

        # Load full dataset
        full_dataset = datasets.ImageFolder(root=cfg.DATA_DIR, transform=train_transforms)
        
        #Check class mapping
        #index assignment to each class folder is based on the alphabetical order of folder names
        #print(full_dataset.class_to_idx) 

        # ---- Class information (EXPOSED) ----
        class_names = full_dataset.classes
        num_classes = len(class_names)

        # Split dataset (Train / Validation / Test)
        train_size = int(0.7 * len(full_dataset))
        val_size   = int(0.15 * len(full_dataset))
        test_size  = len(full_dataset) - train_size - val_size

        # Reproducible split (Without a fixed seed = different train/val/test splits every run)
        generator = torch.Generator().manual_seed(seed)

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=generator
        )

        # Apply correct transforms
        val_dataset.dataset.transform = test_transforms
        test_dataset.dataset.transform = test_transforms

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        return train_loader, val_loader, test_loader, class_names, num_classes