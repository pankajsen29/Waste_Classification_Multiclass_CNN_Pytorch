#####################################################
# Inference Step 1: 
# - defines image preprocessing transform and 
# - provides class information of the trained dataset
#####################################################

from torchvision import transforms
from torchvision import datasets

DATA_DIR = "data/RealWaste"

#ImageNet normalization (mandatory for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


#same as the Validation / Test transforms (NO augmentation, only resize + normalize)
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

def get_class_info():       
        
        # Load full dataset
        full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=inference_transforms)
        
        #Check class mapping
        #index assignment to each class folder is based on the alphabetical order of folder names
        #print(full_dataset.class_to_idx) 

        # ---- Class information (EXPOSED) ----
        class_names = full_dataset.classes
        num_classes = len(class_names)

        return class_names, num_classes