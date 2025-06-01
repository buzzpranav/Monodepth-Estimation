#!/usr/bin/env python
"""
Mars Terrain Depth Estimation
This is the main script for running the depth estimation model for Mars terrain reconstruction.
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image

# Import model components
from models.model import DepthAnything
from utils.terrain_reconstruction_fixed import MarsTerrainMapper

def parse_args():
    parser = argparse.ArgumentParser(
        description="Mars Terrain Depth Estimation"
    )
    parser.add_argument(
        "--input", type=str, default=None, 
        help="Input image file or directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="Directory to save results"
    )
    parser.add_argument(
        "--source", type=str, default="auto",
        choices=["auto", "rover", "ingenuity", "satellite"],
        help="Source of the image(s) - affects depth scaling"
    )
    parser.add_argument(
        "--max_size", type=int, default=1024,
        help="Maximum image size (preserves aspect ratio)"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show results interactively"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for processing multiple images"
    )
    parser.add_argument(
        "--terrain_reconstruction", action="store_true",
        help="Generate 3D terrain reconstruction"
    )
    
    return parser.parse_args()

def run_depth_estimation(model, image_paths, args):
    """Run depth estimation on the given image paths"""
    device = next(model.parameters()).device
    terrain_mapper = MarsTerrainMapper(model=model)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for image_path in image_paths:
        print(f"Processing {image_path}...")
        
        # Open image
        img = Image.open(image_path).convert("RGB")
        
        # Resize if needed
        if max(img.size) > args.max_size:
            scale = args.max_size / max(img.size)
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.LANCZOS)
        
        # Prepare input tensor
        img_tensor = torch.from_numpy(np.array(img).transpose((2, 0, 1))).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Determine source type
        if args.source == "auto":
            if "rover" in image_path.lower():
                source_type = "rover"
            elif "ingenuity" in image_path.lower() or "aerial" in image_path.lower():
                source_type = "ingenuity"
            elif "satellite" in image_path.lower() or "orbital" in image_path.lower():
                source_type = "satellite"
            else:
                source_type = "rover"  # Default
        else:
            source_type = args.source
        
        # Prepare metadata
        metadata = {"source": source_type}
        
        # Run inference
        with torch.no_grad():
            outputs = model(img_tensor, metadata)
            
            # Get disparity map at the highest resolution
            disp = outputs[("disp", 0)]
            
            # Convert to depth
            depth = 1.0 / disp.clamp(min=1e-6)
        
        # Convert to numpy for visualization
        disp_np = disp.cpu().squeeze().numpy()
        depth_np = depth.cpu().squeeze().numpy()
        
        # Save results
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Create visualization figure
        plt.figure(figsize=(18, 6))
        
        # Input image
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title("Input Image")
        plt.axis('off')
        
        # Disparity map
        plt.subplot(1, 3, 2)
        plt.imshow(disp_np, cmap='magma')
        plt.title(f"Disparity Map ({source_type})")
        plt.colorbar(shrink=0.8)
        plt.axis('off')
        
        # Depth map 
        plt.subplot(1, 3, 3)
        plt.imshow(depth_np, cmap='viridis')
        plt.title("Depth Map")
        plt.colorbar(shrink=0.8)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"{image_name}_depth.png"))
        
        if args.show:
            plt.show()
        else:
            plt.close()
              # Save depth data as numpy array
        np.save(os.path.join(args.output_dir, f"{image_name}_depth.npy"), depth_np)
        
        # Generate terrain reconstruction if requested
        if args.terrain_reconstruction:
            print("Terrain reconstruction is not implemented in this version.")
        
        print(f"Results saved to {args.output_dir}")

def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading Depth Anything model...")
    model = DepthAnything(pretrained=True)
    model.to(device)
    model.eval()
    
    # Get input paths
    if args.input is None:
        print("No input specified. Please provide an image or directory.")
        return
        
    if os.path.isdir(args.input):
        # Get all images from directory
        image_paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            image_paths.extend(glob.glob(os.path.join(args.input, ext)))
        
        if not image_paths:
            print(f"No images found in {args.input}")
            return
    else:
        # Single image
        if not os.path.exists(args.input):
            print(f"Input file {args.input} does not exist.")
            return
        image_paths = [args.input]
    
    print(f"Found {len(image_paths)} images to process.")
    
    # Run depth estimation
    run_depth_estimation(model, image_paths, args)

if __name__ == "__main__":
    main()
