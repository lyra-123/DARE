import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from PIL import Image
import os
import pandas as pd


class VideoFrameDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            root_dir (string): Directory with all the frame images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform or ToTensor()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get the path of the image
        img_path = os.path.join(self.root_dir, 'images',f"{idx+1:04d}.JPEG")
        # Read the image
        # image = read_image(img_path)
        image = Image.open(img_path)
        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Extract the F1 scores from the annotations DataFrame
        # Note that idx starts from 0, so we need to match it with the correct row in the CSV
        # by adding 1, since our images start from 1.png

        f1_scores = self.annotations.iloc[idx, 1:].values.astype('float32')
        f1_scores = torch.tensor(f1_scores, dtype=torch.float)

        return image, f1_scores