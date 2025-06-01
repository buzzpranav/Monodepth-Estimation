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
import time

# Import model components
from models.enhanced_model import EnhancedDepthAnything
from models.model import DepthAnything
from utils.terrain_reconstruction_fixed import MarsTerrainMapper
from utils.advanced_visualization import AdvancedMarsVisualizer

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
    parser.add_argument(
        "--enhanced_model", action="store_true",
        help="Use the enhanced depth estimation model with uncertainty"
    )
    parser.add_argument(
        "--visualizations", type=str, default="standard",
        choices=["standard", "advanced", "all", "interactive", "anaglyph", "terrain_features"],
        help="Visualization type to generate"
    )
    parser.add_argument(
        "--flyover", action="store_true",
        help="Generate terrain flyover animation (requires plotly)"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run benchmark mode to compare processing time and quality"
    )
    parser.add_argument(
        "--multi_scale_fusion", action="store_true",
        help="Enable multi-scale fusion for improved accuracy"
    )
    
    return parser.parse_args()

def run_depth_estimation(model, image_paths, args):
    """Run depth estimation on the given image paths"""
    device = next(model.parameters()).device
    terrain_mapper = MarsTerrainMapper(model=model)
    
    # Initialize advanced visualizer if needed
    if args.visualizations != "standard" or args.flyover:
        visualizer = AdvancedMarsVisualizer(min_depth=0.1, max_depth=100.0)
    
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
        
        # Start timing if in benchmark mode
        if args.benchmark:
            start_time = time.time()
        
        # Run inference
        with torch.no_grad():
            outputs = model(img_tensor, metadata)
            
            # Get disparity map at the highest resolution
            disp = outputs[("disp", 0)]
            
            # Get uncertainty map if available (from enhanced model)
            uncertainty = outputs.get(("uncertainty", 0), None)
            
            # Convert to depth
            depth = 1.0 / disp.clamp(min=1e-6)
        
        # Measure inference time for benchmarking
        if args.benchmark:
            inference_time = time.time() - start_time
            print(f"Inference time: {inference_time:.3f} seconds")
        
        # Convert to numpy for visualization
        disp_np = disp.cpu().squeeze().numpy()
        depth_np = depth.cpu().squeeze().numpy()
        uncertainty_np = uncertainty.cpu().squeeze().numpy() if uncertainty is not None else None
        
        # Create results directory
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        result_dir = os.path.join(args.output_dir, image_name)
        os.makedirs(result_dir, exist_ok=True)
        
        # Create standard visualization
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
        plt.savefig(os.path.join(result_dir, f"{image_name}_depth_standard.png"))
        
        if args.show and args.visualizations == "standard":
            plt.show()
        else:
            plt.close()
        
        # Save depth data as numpy array
        np.save(os.path.join(result_dir, f"{image_name}_depth.npy"), depth_np)
        
        # Create advanced visualizations if requested
        if args.visualizations in ["advanced", "all"]:
            # Create multi-view visualization
            visualizer.create_multi_view_visualization(
                img, depth_np, uncertainty_np, normals=None,
                save_path=os.path.join(result_dir, f"{image_name}_multiview.png"),
                show=args.show, view_3d=True
            )
        
        # Create interactive visualization if requested
        if args.visualizations in ["interactive", "all"]:
            visualizer.create_interactive_visualization(
                img, depth_np, uncertainty_np,
                save_path=os.path.join(result_dir, f"{image_name}_interactive")
            )
        
        # Create terrain feature visualization if requested
        if args.visualizations in ["terrain_features", "all"]:
            visualizer.visualize_terrain_features(
                depth_np, img,
                save_path=os.path.join(result_dir, f"{image_name}_terrain_features.png"),
                show=args.show
            )
        
        # Create anaglyph 3D visualization if requested
        if args.visualizations in ["anaglyph", "all"]:
            visualizer.create_anaglyph_3d(
                img, depth_np,
                save_path=os.path.join(result_dir, f"{image_name}_anaglyph.png"),
                show=args.show
            )
        
        # Create flyover animation if requested
        if args.flyover:
            visualizer.create_terrain_flyover_animation(
                depth_np, img, 
                save_path=os.path.join(result_dir, f"{image_name}_flyover")
            )
        
        # Generate 3D terrain reconstruction if requested
        if args.terrain_reconstruction:
            # Setup reconstruction parameters
            recon_output = os.path.join(result_dir, f"{image_name}_reconstruction.ply")
            
            # Get 3D points and colors
            points3D, colors = terrain_mapper.reconstructor.reconstruct_terrain(
                depth_np, np.array(img)
            )
            
            # Save reconstruction if open3d is available
            try:
                import open3d as o3d
                
                # Create point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points3D)
                if colors is not None:
                    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
                
                # Save as PLY file
                o3d.io.write_point_cloud(recon_output, pcd)
                print(f"3D reconstruction saved to {recon_output}")
            except ImportError:
                print("open3d not available. Cannot save 3D reconstruction.")
        
        print(f"Results saved to {result_dir}")

def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model - either standard or enhanced
    if args.enhanced_model:
        print("Loading Enhanced Depth Anything model with uncertainty estimation...")
        model = EnhancedDepthAnything(pretrained=True)
    else:
        print("Loading standard Depth Anything model...")
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
