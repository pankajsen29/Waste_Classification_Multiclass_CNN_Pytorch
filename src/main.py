######## dataset ######
from src.data import get_dataloaders

BATCH_SIZE = 16
NUM_WORKERS = 0

train_loader, val_loader, test_loader, class_names, num_classes = get_dataloaders(BATCH_SIZE, NUM_WORKERS)

#TESTCODE
images, labels = next(iter(train_loader))
print(images.shape, labels.shape)
#o/p: torch.Size([8, 3, 224, 224]) torch.Size([8])
#images=a single batch pulled from train_loader
#lables=targets for that batch