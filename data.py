from transformers import ViTImageProcessor
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
image_mean = processor.image_mean  # Tokenize and pad/truncate labels
image_std = processor.image_std
size = processor.size["height"]


from torchvision.transforms import (CenterCrop,
                                   Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)

normalize = Normalize(mean=image_mean, std=image_std)
_train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

_val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

class CustomDataset(Dataset):
    def __init__(self, dataframe, image_path,transform=None, tokenizer=None, resize_transform=None):
        self.dataframe = dataframe
        self.transform = transform
        # self.tokenizer = tokenizer
        self.max_length = 256
        self.resize_transform = resize_transform
        self.class_labels=self.dataframe.columns[2:-1].tolist()
        self.dataframe['binary_labels'] = pd.Series(self.dataframe[self.class_labels].fillna('').values.tolist())
        self.image_path=image_path

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.image_path+self.dataframe.iloc[idx]['ImageID']# Assuming the first column is the image path
        bin_label = self.dataframe.iloc[idx]['binary_labels'] # Assuming the third column is the binary label
        # malignant = self.dataframe.iloc[idx, ] # Assuming the fourth column is the malignant label

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply resizing transformation if provided
        if self.resize_transform:
            image = self.resize_transform(image)


        # Apply other transformations if provided
        if self.transform:
            image = self.transform(image)
        # print(s'bin_label type: ', type(bin_label))

        bin_label = torch.tensor(bin_label, dtype = torch.float32)
        # malignant = torch.tensor(malignant, dtype = torch.float32)
        # Tokenize and pad/truncate labels
        # encoding = tokenizer(label, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        return {
            'pixel_values': image,
            # 'input_ids': encoding['input_ids'].squeeze(),
            # 'attention_mask': encoding['attention_mask'].squeeze(),
            'label': bin_label,
            # 'actual_label' : label,
            # "malignant" : malignant
        }


def load_dataset(data,datapoints):
    if(data=='fitz'):
        resize_transform = Resize((224,224))  # Change 256 and 256 to your desired dimensions
        df=pd.read_csv('Dataset/image.csv')
        df=df.drop(index=[78,217,549,986,1092,1756,1987,2136,2471,2552,2593,2688,3084,3602]).reset_index(drop=True)
        train_ds = CustomDataset(dataframe=df[:datapoints].reset_index(drop=True), image_path='Dataset/fitz_images/',transform=_train_transforms, resize_transform=resize_transform)
        val_ds = CustomDataset(dataframe=df[3000:].reset_index(drop=True), image_path='Dataset/fitz_images/',transform=_train_transforms, resize_transform=resize_transform)
     
        return train_ds,val_ds


        



