from __future__ import absolute_import, division, print_function

import os
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import torch
from torchvision import transforms
import time

# Import the new depth estimation model
from models.depth_anything_model import DepthAnything
from utils.visualization import MarsTerrainVisualizer

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing function for Mars terrain depth estimation')

    # Input arguments
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to a single image or folder of images')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save output visualizations')
    
    # Image format arguments
    parser.add_argument('--ext', type=str, default="png",
                        help='Image extension to search for in folder')
    
    # Model arguments
    parser.add_argument('--mars_weights', type=str, default=None,
                        help='Path to Mars-specific finetuned weights (optional)')
    
    # Device arguments
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')

    return parser.parse_args()

def monodepth(args):
    """Process images using the Mars terrain depth estimation model"""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model - using the new Depth Anything model
    print("Loading Depth Anything model...")
    model = DepthAnything(pretrained=True)
    
    # Apply Mars domain adaptation if specified
    if args.mars_weights is not None and os.path.exists(args.mars_weights):
        print(f"Loading Mars-specific weights from {args.mars_weights}")
        model.convert_to_mars_domain(args.mars_weights)
    else:
        print("Using general domain adaptation for Mars terrain")
        model.convert_to_mars_domain()
    
    # Move model to device and set to eval mode
    model.to(device)
    model.eval()
    
    # Initialize the visualizer
    visualizer = MarsTerrainVisualizer(mars_themed=True)
      # Get input paths
    if os.path.isfile(args.image_path):
        paths = [args.image_path]
        output_directory = args.output_dir
    elif os.path.isdir(args.image_path):
        paths = glob.glob(os.path.join(args.image_path, f'*.{args.ext}'))
        output_directory = args.output_dir
    else:
        print(f"Warning: Path not found directly: {args.image_path}")
        # Try to handle case where filename might have spaces instead of underscores
        if '_' in args.image_path:
            space_path = args.image_path.replace('_', ' ')
            if os.path.exists(space_path):
                print(f"Found file with spaces instead of underscores: {space_path}")
                paths = [space_path]
                output_directory = args.output_dir
            else:
                raise Exception(f"Cannot find image path: {args.image_path}")
        else:
            raise Exception(f"Cannot find image path: {args.image_path}")
    
    print(f"Found {len(paths)} images")
    
    # Process each image
    with torch.no_grad():
        for idx, image_path in enumerate(paths):
            print(f"Processing image {idx+1}/{len(paths)}: {image_path}")
            start_time = time.time()
            
            # Load and preprocess image
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            
            # Prepare input tensor
            img_tensor = transforms.ToTensor()(input_image).unsqueeze(0)
            img_tensor = img_tensor.to(device)
            
            # Generate depth prediction
            outputs = model(img_tensor)
            disp = outputs[("disp", 0)]
            
            # Resize to original resolution
            if disp.shape[2:] != (original_height, original_width):
                disp_resized = torch.nn.functional.interpolate(
                    disp, (original_height, original_width), 
                    mode="bilinear", align_corners=False
                )
            else:
                disp_resized = disp
              # Create visualization
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            
            # Save visualization
            name_dest_im = os.path.join(output_directory, f"{output_name}_depth.png")
            visualizer.visualize_depth_map(input_image, disp_resized_np, save_path=name_dest_im, show=False)
            
            print(f"   Processed in {time.time() - start_time:.2f}s - saved to {name_dest_im}")
    
    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    monodepth(args)