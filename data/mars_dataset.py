from __future__ import absolute_import, division, print_function

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import PIL.Image as pil
import random
import shutil
import glob
from collections import defaultdict
from tqdm import tqdm


class MarsImageDataset(Dataset):
    """
    Dataset for Mars imagery from either rover or satellite perspectives.
    Includes functionality for organizing raw images into a proper dataset structure.
    """
    def __init__(self, data_path, split='train', height=384, width=512, 
                 augment=True, source_type='mixed', img_ext=['jpg', 'png', 'jpeg']):
        super(MarsImageDataset, self).__init__()
        
        self.data_path = data_path
        self.split = split
        self.height = height
        self.width = width
        self.augment = augment and split == 'train'  # Only augment training data
        self.source_type = source_type  # 'rover', 'satellite', or 'mixed'
        self.img_ext = img_ext
        
        # Collect all image paths
        self.image_paths = self._get_image_paths()
        
        # Set up transformations
        self._setup_transforms()
        
        # Print dataset info
        print(f"Mars dataset loaded with {len(self.image_paths)} images")
        print(f"  Source type: {source_type}")
        print(f"  Split: {split}")
        
    def _get_image_paths(self):
        """Collect image paths based on source type"""
        paths = []
        
        # Define subdirectories based on source type
        if self.source_type == 'mixed':
            subdirs = ['rover', 'satellite']
        else:
            subdirs = [self.source_type]
        
        # Walk through the directories
        for subdir in subdirs:
            source_dir = os.path.join(self.data_path, subdir, self.split)
            if not os.path.exists(source_dir):
                continue
                
            for root, _, files in os.walk(source_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in self.img_ext):
                        img_path = os.path.join(root, file)
                        # Store the path along with the source type
                        paths.append({
                            'path': img_path,
                            'source': subdir
                        })
        
        return paths
    
    def _setup_transforms(self):
        """Set up image transformations"""
        # Basic transforms for all splits
        self.basic_transform = transforms.Compose([
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Augmentations for training
        if self.augment:
            self.augment_transforms = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomHorizontalFlip(),
            ])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        item = self.image_paths[idx]
        img_path = item['path']
        source_type = item['source']
        
        # Load image
        try:
            img = pil.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder if image loading fails
            return self.__getitem__(random.randint(0, len(self) - 1))
            
        # Apply augmentations if training
        if self.augment:
            img = self.augment_transforms(img)
            
        # Apply basic transforms
        img_tensor = self.basic_transform(img)
        
        # Create metadata dictionary
        metadata = {
            'source': source_type,
            'path': img_path,
            'altitude': 0 if source_type == 'rover' else 10000  # Default values
        }
        
        return {
            'image': img_tensor,
            'metadata': metadata
        }
    
    @staticmethod
    def organize_dataset(input_dir, output_dir, source_type="mixed", val_split=0.1, test_split=0.1):
        """
        Organize Mars imagery into a proper dataset structure
        
        Args:
            input_dir: Directory containing Mars images
            output_dir: Output directory for organized dataset
            source_type: Source type of the images (rover, satellite, or mixed)
            val_split: Fraction of images to use for validation
            test_split: Fraction of images to use for testing
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create split directories
        splits = ['train', 'val', 'test']
        for split in splits:
            os.makedirs(os.path.join(output_dir, split), exist_ok=True)
            if source_type == "mixed":
                os.makedirs(os.path.join(output_dir, split, "rover"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, split, "satellite"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, split, "ingenuity"), exist_ok=True)
        
        # Get all image files
        image_files = []
        for ext in ['png', 'jpg', 'jpeg']:
            image_files.extend(glob.glob(os.path.join(input_dir, f"*.{ext}")))
        
        if len(image_files) == 0:
            raise ValueError(f"No images found in {input_dir}")
        
        # Shuffle images
        random.shuffle(image_files)
        
        # Calculate split indices
        n_images = len(image_files)
        n_val = int(n_images * val_split)
        n_test = int(n_images * test_split)
        n_train = n_images - n_val - n_test
        
        # Split images
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # Copy files to respective directories
        splits_files = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        for split, files in splits_files.items():
            print(f"\nProcessing {split} set...")
            for src_path in tqdm(files):
                file_name = os.path.basename(src_path)
                
                if source_type == "mixed":
                    # Determine source type from filename or path
                    if "rover" in src_path.lower():
                        sub_dir = "rover"
                    elif "satellite" in src_path.lower():
                        sub_dir = "satellite"
                    else:
                        sub_dir = "ingenuity"
                    dst_path = os.path.join(output_dir, split, sub_dir, file_name)
                else:
                    dst_path = os.path.join(output_dir, split, file_name)
                
                shutil.copy2(src_path, dst_path)
        
        print(f"\nDataset organization complete!")
        print(f"Train set: {len(train_files)} images")
        print(f"Validation set: {len(val_files)} images")
        print(f"Test set: {len(test_files)} images")


def get_mars_dataloaders(data_path, batch_size=8, num_workers=4, 
                         height=384, width=512, source_type='mixed'):
    """
    Create dataloaders for Mars imagery datasets
    
    Args:
        data_path: Root path to the Mars dataset
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        height, width: Image dimensions
        source_type: Type of Mars imagery ('rover', 'satellite', or 'mixed')
        
    Returns:
        Dictionary containing train, val, and test dataloaders
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset = MarsImageDataset(
            data_path=data_path,
            split=split,
            height=height,
            width=width,
            augment=(split == 'train'),
            source_type=source_type
        )
        
        # Create dataloader
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train')
        )
    
    return dataloaders
