from __future__ import absolute_import, division, print_function

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.depth_anything_model import DepthAnything
from data.mars_dataset import get_mars_dataloaders
from train.mars_adapter import MarsDomainAdapter
from utils.visualization import MarsTerrainVisualizer


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fine-tune Depth Anything for Mars terrain')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default="./mars_data",
                        help='Path to Mars imagery dataset')
    parser.add_argument('--source_type', type=str, default="mixed",
                        choices=["mixed", "rover", "satellite"],
                        help='Type of Mars imagery to train on')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr_scheduler_step_size', type=int, default=15,
                        help='Step size for learning rate scheduler')
    parser.add_argument('--lr_scheduler_gamma', type=float, default=0.5,
                        help='Gamma for learning rate scheduler')
    parser.add_argument('--model_weights', type=str, default=None,
                        help='Path to pretrained model weights')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default="./trained_models",
                        help='Directory to save trained models')
    
    # Device arguments
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    
    # Image size
    parser.add_argument('--height', type=int, default=384,
                        help='Image height')
    parser.add_argument('--width', type=int, default=512,
                        help='Image width')
    
    # Loss weights
    parser.add_argument('--depth_loss_weight', type=float, default=1.0,
                        help='Weight for depth loss')
    parser.add_argument('--smoothness_loss_weight', type=float, default=0.001,
                        help='Weight for smoothness loss')
    
    return parser.parse_args()


def prepare_synthetic_mars_data(args):
    """
    Prepare synthetic Mars data for fine-tuning when no real dataset is available
    
    Args:
        args: Command-line arguments
        
    Returns:
        data_path: Path to prepared synthetic data
    """
    import cv2
    import shutil
    from PIL import Image, ImageEnhance, ImageFilter
    
    print("Preparing synthetic Mars data from asset samples...")
    
    # Create dataset structure
    data_path = os.path.join(args.output_dir, "synthetic_mars_data")
    os.makedirs(data_path, exist_ok=True)
    
    # Create source-specific directories
    for source in ['rover', 'satellite']:
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(data_path, source, split), exist_ok=True)
    
    # Use assets as starting point for rover images
    rover_src = os.path.join("assets", "rover")
    if os.path.exists(rover_src):
        rover_images = glob.glob(os.path.join(rover_src, "*.png")) + \
                      glob.glob(os.path.join(rover_src, "*.jpg"))
                      
        # Split into train/val/test (70/15/15)
        num_train = int(len(rover_images) * 0.7)
        num_val = int(len(rover_images) * 0.15)
        
        np.random.shuffle(rover_images)
        
        train_imgs = rover_images[:num_train]
        val_imgs = rover_images[num_train:num_train+num_val]
        test_imgs = rover_images[num_train+num_val:]
        
        # Copy and augment rover images
        for i, img_path in enumerate(tqdm(train_imgs)):
            # Original image
            img = Image.open(img_path).convert('RGB')
            filename = f"rover_train_{i:04d}.png"
            img.save(os.path.join(data_path, "rover", "train", filename))
            
            # Augmented versions
            for j in range(3):  # Create multiple augmentations
                aug_img = img.copy()
                
                # Apply random Mars-like color adjustments
                contrast = ImageEnhance.Contrast(aug_img)
                aug_img = contrast.enhance(np.random.uniform(0.8, 1.2))
                
                # Add reddish tint (Mars dust effect)
                color = ImageEnhance.Color(aug_img)
                aug_img = color.enhance(np.random.uniform(0.9, 1.1))
                
                # Convert to numpy for more adjustments
                aug_np = np.array(aug_img)
                
                # Add more red channel (Mars atmosphere)
                aug_np[:,:,0] = np.clip(aug_np[:,:,0] * np.random.uniform(1.05, 1.15), 0, 255).astype(np.uint8)
                
                # Add noise (dust)
                dust = np.random.normal(0, 5, aug_np.shape).astype(np.int16)
                aug_np = np.clip(aug_np.astype(np.int16) + dust, 0, 255).astype(np.uint8)
                
                aug_img = Image.fromarray(aug_np)
                
                # Save augmented image
                filename = f"rover_train_{i:04d}_aug{j}.png"
                aug_img.save(os.path.join(data_path, "rover", "train", filename))
        
        # Copy validation and test images
        for src_list, dest_dir, prefix in [
            (val_imgs, os.path.join(data_path, "rover", "val"), "rover_val_"),
            (test_imgs, os.path.join(data_path, "rover", "test"), "rover_test_")
        ]:
            for i, img_path in enumerate(src_list):
                img = Image.open(img_path).convert('RGB')
                filename = f"{prefix}{i:04d}.png"
                img.save(os.path.join(dest_dir, filename))
    
    # Use orbital/satellite images if available
    sat_src = os.path.join("assets", "satellite")
    if os.path.exists(sat_src):
        # Similar process for satellite images
        # ...
        pass
    
    print(f"Synthetic Mars dataset created at {data_path}")
    return data_path


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if Mars data directory exists, if not, prepare synthetic data
    if not os.path.exists(args.data_dir):
        print(f"Mars data directory not found: {args.data_dir}")
        print("Creating synthetic Mars data from available assets...")
        args.data_dir = prepare_synthetic_mars_data(args)
    
    # Create configuration for domain adapter
    config = {
        'data_path': args.data_dir,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'height': args.height,
        'width': args.width,
        'save_dir': args.output_dir,
        'source_type': args.source_type,
        'lr_scheduler_step_size': args.lr_scheduler_step_size,
        'lr_scheduler_gamma': args.lr_scheduler_gamma,
        'depth_loss_weight': args.depth_loss_weight,
        'smoothness_loss_weight': args.smoothness_loss_weight
    }
    
    # Create the Mars domain adapter
    adapter = MarsDomainAdapter(config)
    
    # Load custom model weights if specified
    if args.model_weights is not None and os.path.exists(args.model_weights):
        print(f"Loading custom weights from {args.model_weights}")
        adapter.load_model(args.model_weights)
    
    # Start fine-tuning
    print(f"Starting Mars domain adaptation for {args.epochs} epochs...")
    adapter.finetune(num_epochs=args.epochs)
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "mars_depth_anything_final.pth")
    adapter.save_model("mars_depth_anything_final.pth")
    print(f"Fine-tuning complete! Final model saved to {final_model_path}")
    
    # Test the fine-tuned model on a few examples
    print("Testing the fine-tuned model on validation examples...")
    
    # Get validation loader
    val_loader = adapter.dataloaders['val']
    
    # Create visualizer
    visualizer = MarsTerrainVisualizer(mars_themed=True)
    
    # Run a few examples through the model
    model = adapter.model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= 5:  # Just test on first 5 batches
                break
                
            images = batch['image'].to(device)
            metadata = batch['metadata']
            
            # Forward pass
            outputs = model(images, metadata)
            
            # Get predicted depth
            disp = outputs[("disp", 0)]
            
            # Convert to depth for visualization
            from layers import disp_to_depth
            _, depth = disp_to_depth(disp, min_depth=0.1, max_depth=100.0)
            
            # Visualize each image in the batch
            for b in range(images.shape[0]):
                img = images[b]
                pred_depth = depth[b]
                
                # Visualize
                vis_path = os.path.join(args.output_dir, f"example_{batch_idx}_{b}.png")
                visualizer.visualize_depth_map(img, pred_depth, save_path=vis_path, show=False)
    
    print(f"Validation examples saved to {args.output_dir}")
    print("Mars domain adaptation complete!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
