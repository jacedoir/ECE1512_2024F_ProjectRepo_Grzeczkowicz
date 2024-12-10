import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MHISTDataset(Dataset):
    def __init__(self, root_dir, annotations_file, transform=None):
        """
        Args:
            root_dir (str): Directory containing images.
            annotations_file (str): Path to the annotations CSV file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotations_file)
        
        # Define class names based on unique labels in the annotations file
        self.classes = ["HP", "SSA"]  # Hardcoded for MHIST; can also infer dynamically if needed
        
        # Map labels to indices (e.g., "HP" -> 0, "SSA" -> 1)
        self.class_to_idx = {label: idx for idx, label in enumerate(self.classes)}
        self.annotations['label_idx'] = self.annotations.iloc[:, 1].map(self.class_to_idx)

        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get image path and label index
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label_idx = self.annotations.iloc[idx, -1]  # Use the mapped label index

        if self.transform:
            image = self.transform(image)

        return image, label_idx
