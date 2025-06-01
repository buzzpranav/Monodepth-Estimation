#!/usr/bin/env python
# script: test_mars_depth.py
"""
Test the Mars depth estimation model on real Mars imagery.
This script evaluates the model's performance on rover, satellite, and aerial (ingenuity) perspectives,
generates visual comparisons, and reports performance metrics.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import time
import glob
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from models.model import DepthAnything
from utils.terrain_reconstruction_fixed import MarsTerrainMapper
from utils.visualization import MarsTerrainVisualizer
from layers import disp_to_depth


def parse_args():
    parser = argparse.ArgumentParser(description="Test Mars Depth Estimation on Real Imagery")
    
    # Input arguments
    parser.add_argument("--data_root", type=str, default="../assets",
                        help="Root directory containing Mars imagery")
    parser.add_argument("--output_dir", type=str, default="../test_results",
                        help="Output directory for evaluation results")
    
    # Model arguments
    parser.add_argument("--model_weights", type=str, default=None,
                        help="Path to model weights (if None, uses default pretrained weights)")
    parser.add_argument("--mars_weights", type=str, default=None,
                        help="Path to Mars-specific finetuned weights (optional)")
    
    # Evaluation options
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples per source to evaluate (0 for all)")
    parser.add_argument("--reconstruct_3d", action="store_true",
                        help="Also evaluate 3D reconstruction quality")
    parser.add_argument("--test_stitching", action="store_true",
                        help="Test terrain stitching capabilities")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose debugging output")
    
    # Performance testing
    parser.add_argument("--benchmark", action="store_true",
                        help="Run performance benchmarking")
    parser.add_argument("--repeats", type=int, default=10,
                        help="Number of repeat runs for benchmarking")
    
    # Hardware settings
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available")
    
    return parser.parse_args()


def load_model(args, device):
    """Load the depth estimation model"""
    print("Loading Depth Anything model...")
    model = DepthAnything(pretrained=True)
    
    # Load custom weights if specified
    if args.model_weights is not None and os.path.exists(args.model_weights):
        print(f"Loading custom weights from {args.model_weights}")
        model.load_state_dict(torch.load(args.model_weights, map_location=device))
    
    # Apply Mars domain adaptation if specified
    if args.mars_weights is not None and os.path.exists(args.mars_weights):
        print(f"Loading Mars-specific weights from {args.mars_weights}")
        model.convert_to_mars_domain(args.mars_weights)
    else:
        print("Using general domain adaptation for Mars terrain")
        model.convert_to_mars_domain()
    
    # Move model to device
    model.to(device)
    model.eval()
    
    return model


def collect_test_images(args):
    """Collect test images from different sources"""
    sources = ["rover", "satellite", "ingenuity"]
    test_images = {}
    
    for source in sources:
        source_dir = os.path.join(args.data_root, source)
        if not os.path.exists(source_dir):
            print(f"Warning: Source directory {source_dir} not found. Skipping.")
            continue
            
        # Get image paths
        image_paths = sorted(glob.glob(os.path.join(source_dir, "*.png")))
        if not image_paths:
            print(f"Warning: No images found in {source_dir}")
            continue
            
        # Limit to num_samples if specified
        if args.num_samples > 0 and len(image_paths) > args.num_samples:
            # Take evenly spaced samples
            indices = np.linspace(0, len(image_paths) - 1, args.num_samples, dtype=int)
            image_paths = [image_paths[i] for i in indices]
            
        test_images[source] = image_paths
        print(f"Found {len(image_paths)} {source} images for testing")
    
    return test_images


def prepare_metadata(source, image_path):
    """Prepare metadata dictionary for the image"""
    metadata = {
        "source": source,
        "path": image_path
    }
    
    # Add source-specific metadata
    if source == "satellite":
        metadata["altitude"] = 400000  # Example altitude in meters for Mars satellite
    elif source == "ingenuity":
        metadata["altitude"] = 10  # Example altitude in meters for Mars helicopter
        
    return metadata


def process_image(image_path, mapper, visualizer, metadata, output_dir, args):
    """Process a single image with the Mars terrain mapper"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_name = os.path.basename(image_path)
    
    # Create output directory
    image_output_dir = os.path.join(output_dir, os.path.splitext(image_name)[0])
    os.makedirs(image_output_dir, exist_ok=True)
    
    # Store original size for later
    original_size = image.size
    
    # Resize image to 224x224 (ViT model requirement)
    image_resized = image.resize((224, 224), Image.LANCZOS)
    
    if args.verbose:
        print(f"Processing image: {image_path}")
    
    # Start timing
    start_time = time.time()
    
    # Convert PIL image to tensor
    img_tensor = torch.from_numpy(np.array(image_resized).transpose((2, 0, 1))).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    
    # Move tensor to device
    device = next(mapper.model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    # Estimate depth
    depth = mapper.infer_depth(img_tensor, metadata)
    
    # Record processing time
    processing_time = time.time() - start_time
    
    # Save depth map visualization
    depth_path = os.path.join(image_output_dir, f"{os.path.splitext(image_name)[0]}_depth.png")
    visualizer.visualize_depth_map(image, depth, save_path=depth_path, show=False)
    
    # Results dictionary
    result = {
        "image_path": image_path,
        "depth_path": depth_path,
        "metadata": metadata,
        "processing_time": processing_time
    }
    
    # 3D reconstruction if requested
    if args.reconstruct_3d:
        recon_start_time = time.time()
        
        # Create point cloud
        pcd = mapper.process_image(img_tensor, None, metadata)
        
        # Save point cloud
        pcd_path = os.path.join(image_output_dir, f"{os.path.splitext(image_name)[0]}_pointcloud.ply")
        mapper.reconstructor.save_point_cloud(pcd, pcd_path)
        
        # Create mesh
        mesh = mapper.process_image_to_mesh(img_tensor, None, metadata)
        
        # Save mesh
        mesh_path = os.path.join(image_output_dir, f"{os.path.splitext(image_name)[0]}_mesh.obj")
        mapper.reconstructor.save_mesh(mesh, mesh_path)
        
        # Record reconstruction time
        reconstruction_time = time.time() - recon_start_time
        
        # Add to results
        result["pointcloud_path"] = pcd_path
        result["mesh_path"] = mesh_path
        result["reconstruction_time"] = reconstruction_time
    
    return result


def benchmark_performance(mapper, visualizer, test_images, args):
    """Benchmark performance on various image sources"""
    # Initialize benchmark results
    benchmark_results = {
        "rover": {"inference_times": [], "depths_per_second": [], "resolution": []},
        "satellite": {"inference_times": [], "depths_per_second": [], "resolution": []},
        "ingenuity": {"inference_times": [], "depths_per_second": [], "resolution": []}
    }
    
    # Benchmark each source
    for source, image_paths in test_images.items():
        if not image_paths:
            continue
            
        print(f"\nBenchmarking {source} imagery...")
        
        # Use first image for repeated benchmarking
        image_path = image_paths[0]
        metadata = prepare_metadata(source, image_path)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        img_width, img_height = image.size
        
        # Convert to tensor
        img_tensor = torch.from_numpy(np.array(image).transpose((2, 0, 1))).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
        # Move to device
        img_tensor = img_tensor.to(mapper.device)
        
        # Warm-up run
        with torch.no_grad():
            _ = mapper.infer_depth(img_tensor, metadata)
        
        # Benchmark runs
        times = []
        for i in tqdm(range(args.repeats), desc=f"Running {args.repeats} inference passes"):
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Time inference
            start_time = time.time()
            with torch.no_grad():
                _ = mapper.infer_depth(img_tensor, metadata)
            inference_time = time.time() - start_time
            
            times.append(inference_time)
        
        # Calculate statistics
        mean_time = np.mean(times)
        fps = 1.0 / mean_time
        
        print(f"  Resolution: {img_width}x{img_height}")
        print(f"  Average inference time: {mean_time:.4f}s")
        print(f"  Depths per second: {fps:.2f}")
        
        # Add to benchmark results
        benchmark_results[source]["inference_times"].append(mean_time)
        benchmark_results[source]["depths_per_second"].append(fps)
        benchmark_results[source]["resolution"].append(f"{img_width}x{img_height}")
    
    return benchmark_results


def test_stitching(mapper, visualizer, test_images, args):
    """Test terrain stitching capabilities"""
    stitching_results = {}
    
    # Create output directory
    stitch_dir = os.path.join(args.output_dir, "stitching")
    os.makedirs(stitch_dir, exist_ok=True)
    
    # Test stitching for each source
    for source, image_paths in test_images.items():
        # Need at least 3 images for meaningful stitching
        if len(image_paths) < 3:
            continue
            
        print(f"\nTesting terrain stitching for {source} imagery...")
        
        # Take a subset of images for stitching
        stitch_paths = image_paths[:3]  # Use first 3 images
        
        # Load images and prepare metadata
        images = []
        metadata_list = []
        
        for path in stitch_paths:
            # Load image
            image = Image.open(path).convert('RGB')
            
            # Convert to tensor
            img_tensor = torch.from_numpy(np.array(image).transpose((2, 0, 1))).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
            
            # Prepare metadata
            metadata = prepare_metadata(source, path)
            
            images.append(img_tensor)
            metadata_list.append(metadata)
        
        # Start timing
        start_time = time.time()
        
        # Stitch terrain
        combined_mesh = mapper.process_image_sequence(images, None, metadata_list)
        
        # Record stitching time
        stitch_time = time.time() - start_time
        
        # Save stitched mesh
        mesh_path = os.path.join(stitch_dir, f"{source}_stitched_terrain.obj")
        mapper.reconstructor.save_mesh(combined_mesh, mesh_path)
        
        # Create visualization
        # For simplicity, we'll just visualize input and output without interactive display
        fig = plt.figure(figsize=(12, 4))
        
        for i, path in enumerate(stitch_paths[:3]):
            img = Image.open(path).convert('RGB')
            plt.subplot(1, 3, i + 1)
            plt.imshow(img)
            plt.title(f"Input {i+1}")
            plt.axis('off')
            
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(stitch_dir, f"{source}_inputs.png")
        plt.savefig(viz_path)
        plt.close()
        
        # Add to results
        stitching_results[source] = {
            "input_images": stitch_paths,
            "stitched_mesh": mesh_path,
            "input_visualization": viz_path,
            "stitching_time": stitch_time,
            "num_images": len(stitch_paths)
        }
    
    return stitching_results


def create_summary_visualization(test_results, output_dir):
    """Create a summary visualization of test results"""
    sources = list(test_results.keys())
    
    # Create output directory
    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    for source in sources:
        # Get a subset of results for visualization
        results = test_results[source][:min(4, len(test_results[source]))]
        
        if not results:
            continue
        
        # Create a grid of input/depth pairs
        fig = plt.figure(figsize=(12, 3 * len(results)))
        
        for i, result in enumerate(results):
            # Load input image
            input_img = Image.open(result["image_path"]).convert('RGB')
            
            # Load depth visualization
            depth_img = Image.open(result["depth_path"]).convert('RGB')
            
            # Display input
            plt.subplot(len(results), 2, i*2 + 1)
            plt.imshow(input_img)
            plt.title(f"Input: {os.path.basename(result['image_path'])}")
            plt.axis('off')
            
            # Display depth
            plt.subplot(len(results), 2, i*2 + 2)
            plt.imshow(depth_img)
            plt.title(f"Depth Map")
            plt.axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(summary_dir, f"{source}_summary.png")
        plt.savefig(viz_path)
        plt.close()
        
    # Create a single composite visualization
    if all(len(test_results.get(source, [])) > 0 for source in ["rover", "satellite", "ingenuity"]):
        fig = plt.figure(figsize=(15, 12))
        
        # Display one example from each source
        for i, source in enumerate(["rover", "satellite", "ingenuity"]):
            # Get first result
            result = test_results[source][0]
            
            # Load input image
            input_img = Image.open(result["image_path"]).convert('RGB')
            
            # Load depth visualization
            depth_img = Image.open(result["depth_path"]).convert('RGB')
            
            # Display input
            plt.subplot(3, 2, i*2 + 1)
            plt.imshow(input_img)
            plt.title(f"{source.capitalize()} Input")
            plt.axis('off')
            
            # Display depth
            plt.subplot(3, 2, i*2 + 2)
            plt.imshow(depth_img)
            plt.title(f"{source.capitalize()} Depth Map")
            plt.axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(summary_dir, "all_sources_comparison.png")
        plt.savefig(viz_path)
        plt.close()


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args, device)
    
    # Initialize mapper and visualizer
    mapper = MarsTerrainMapper(model=model, use_cuda=not args.no_cuda)
    visualizer = MarsTerrainVisualizer(mars_themed=True)
    
    # Collect test images
    test_images = collect_test_images(args)
      # Process images
    print("\nProcessing test images...")
    test_results = {}
    for source, image_paths in test_images.items():
        print(f"\nProcessing {len(image_paths)} {source} images...")
        results = []
        for image_path in tqdm(image_paths, desc=f"Processing {source} images"):
            # Prepare metadata
            metadata = prepare_metadata(source, image_path)
            
            # Process image
            result = process_image(image_path, mapper, visualizer, metadata, args.output_dir, args)
            results.append(result)
            
        test_results[source] = results
    
    # Create summary visualization
    print("\nCreating summary visualizations...")
    create_summary_visualization(test_results, args.output_dir)
    
    # Run benchmarks if requested
    benchmark_results = None
    if args.benchmark:
        print("\nRunning performance benchmarks...")
        benchmark_results = benchmark_performance(mapper, visualizer, test_images, args)
    
    # Test stitching if requested
    stitching_results = None
    if args.test_stitching:
        print("\nTesting terrain stitching...")
        stitching_results = test_stitching(mapper, visualizer, test_images, args)
    
    # Save overall test results
    print("\nSaving test results...")
    overall_results = {
        "test_configuration": {
            "model_weights": args.model_weights,
            "mars_weights": args.mars_weights,
            "device": str(device),
            "num_samples": args.num_samples,
            "reconstruct_3d": args.reconstruct_3d if hasattr(args, 'reconstruct_3d') else False,
            "test_stitching": args.test_stitching if hasattr(args, 'test_stitching') else False,
            "benchmark": args.benchmark if hasattr(args, 'benchmark') else False
        },
        "test_results": {
            source: [
                {
                    "image_path": os.path.basename(result["image_path"]),
                    "depth_path": os.path.basename(result["depth_path"]),
                    "processing_time": result["processing_time"]
                }
                for result in results
            ]
            for source, results in test_results.items()
        }
    }
    
    # Add benchmark results if available
    if benchmark_results:
        overall_results["benchmark_results"] = benchmark_results
    
    # Add stitching results if available
    if stitching_results:
        overall_results["stitching_results"] = {
            source: {
                "stitched_mesh": os.path.basename(result["stitched_mesh"]),
                "input_visualization": os.path.basename(result["input_visualization"]),
                "stitching_time": result["stitching_time"],
                "num_images": result["num_images"]
            }
            for source, result in stitching_results.items()
        }
    
    # Write JSON report
    report_path = os.path.join(args.output_dir, "test_report.json")
    with open(report_path, 'w') as f:
        json.dump(overall_results, f, indent=2)
    
    print(f"\nTest completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
