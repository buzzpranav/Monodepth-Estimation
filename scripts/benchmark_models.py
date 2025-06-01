#!/usr/bin/env python
# script: benchmark_models.py
"""
Benchmark different depth estimation models on Mars terrain imagery.
This script compares the Depth Anything model against other common depth estimation
approaches, including the original monodepth2 ResNet models and other baselines.
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import time
import glob
import json
import cv2

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from models.depth_anything_model import DepthAnything
from networks.resnet_encoder import ResnetEncoder
from networks.depth_decoder import DepthDecoder
from utils.terrain_reconstruction import MarsTerrainMapper, TerrainReconstructor
from utils.visualization import MarsTerrainVisualizer
from utils.evaluation import MarsTerrainEvaluator
from layers import disp_to_depth


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark depth estimation models on Mars imagery")
    
    # Input arguments
    parser.add_argument("--data_dir", type=str, default="../assets",
                        help="Root directory containing Mars imagery")
    parser.add_argument("--output_dir", type=str, default="../benchmark_results",
                        help="Output directory for benchmark results")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples per source to evaluate (0 for all)")
    
    # Model selection
    parser.add_argument("--models", nargs="+", 
                        default=["depth_anything", "monodepth2_mono", "monodepth2_stereo"],
                        choices=["depth_anything", "monodepth2_mono", "monodepth2_stereo", 
                                 "monodepth2_mono_stereo", "midas", "dpt"],
                        help="Models to benchmark")
    
    # Custom model paths
    parser.add_argument("--depth_anything_weights", type=str, default=None,
                        help="Path to Depth Anything model weights")
    parser.add_argument("--mars_weights", type=str, default=None,
                        help="Path to Mars-specific finetuned weights")
    
    # Evaluation options
    parser.add_argument("--compare_terrain", action="store_true",
                        help="Compare 3D terrain reconstruction quality")
    parser.add_argument("--feature_preservation", action="store_true",
                        help="Analyze feature preservation in depth maps")
    
    # Hardware settings
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available")
    
    return parser.parse_args()


def load_monodepth2_model(model_type, device):
    """Load a monodepth2 model"""
    print(f"Loading Monodepth2 {model_type} model...")
    
    # Define model paths for different types
    model_paths = {
        "mono": "mono_640x192",
        "stereo": "stereo_640x192",
        "mono_stereo": "mono+stereo_640x192"
    }
    
    # Select appropriate model path
    model_name = model_paths.get(model_type, "mono+stereo_640x192")
    
    # Define model architecture
    encoder_path = os.path.join("models", model_name, "encoder.pth")
    depth_decoder_path = os.path.join("models", model_name, "depth.pth")
    
    # Check if the model files exist, otherwise suggest download
    if not os.path.exists(encoder_path) or not os.path.exists(depth_decoder_path):
        print(f"Model files not found. Please download the {model_name} model from:")
        print(f"https://storage.googleapis.com/niantic-lon-static/research/monodepth2/{model_name.replace('+', '%2B')}.zip")
        print("Extract the zip file to the 'models' directory")
        return None, None
    
    # Load encoder
    encoder = ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    encoder.load_state_dict(loaded_dict_enc)
    encoder.to(device)
    encoder.eval()
    
    # Load decoder
    depth_decoder = DepthDecoder(
        num_ch_enc=encoder.num_ch_enc,
        scales=range(4)
    )
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()
    
    return encoder, depth_decoder


def load_depth_anything_model(weights_path, mars_weights, device):
    """Load the Depth Anything model"""
    print("Loading Depth Anything model...")
    model = DepthAnything(pretrained=True)
    
    # Load custom weights if specified
    if weights_path is not None and os.path.exists(weights_path):
        print(f"Loading custom weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device))
    
    # Apply Mars domain adaptation if specified
    if mars_weights is not None and os.path.exists(mars_weights):
        print(f"Loading Mars-specific weights from {mars_weights}")
        model.convert_to_mars_domain(mars_weights)
    else:
        print("Using general domain adaptation for Mars terrain")
        model.convert_to_mars_domain()
    
    # Move model to device
    model.to(device)
    model.eval()
    
    return model


def try_load_midas_model(device):
    """Try to load MiDaS model if available"""
    try:
        import torch
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        midas.to(device)
        midas.eval()
        
        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.default_transform
        
        print("MiDaS model loaded successfully")
        return midas, transform
    except Exception as e:
        print(f"Could not load MiDaS model: {e}")
        print("Skipping MiDaS model in benchmark")
        return None, None


def try_load_dpt_model(device):
    """Try to load DPT model if available"""
    try:
        import torch
        dpt_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        dpt_model.to(device)
        dpt_model.eval()
        
        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.dpt_transform
        
        print("DPT model loaded successfully")
        return dpt_model, transform
    except Exception as e:
        print(f"Could not load DPT model: {e}")
        print("Skipping DPT model in benchmark")
        return None, None


def collect_test_images(args):
    """Collect test images from different sources"""
    sources = ["rover", "satellite", "ingenuity"]
    test_images = {}
    
    for source in sources:
        source_dir = os.path.join(args.data_dir, source)
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


def predict_monodepth2(encoder, depth_decoder, image_tensor, device):
    """Get depth prediction from Monodepth2 model"""
    with torch.no_grad():
        features = encoder(image_tensor)
        outputs = depth_decoder(features)
    
    disp = outputs[("disp", 0)]
    return disp


def predict_depth_anything(model, image_tensor, metadata, device):
    """Get depth prediction from Depth Anything model"""
    with torch.no_grad():
        outputs = model(image_tensor, metadata)
    
    disp = outputs[("disp", 0)]
    return disp


def predict_midas(model, transform, image, device):
    """Get depth prediction from MiDaS model"""
    # Apply transform
    input_batch = transform(image).to(device)
    
    # Predict
    with torch.no_grad():
        prediction = model(input_batch)
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False
        ).squeeze()
    
    # Convert to numpy
    depth = prediction.cpu().numpy()
    
    # Normalize to disparity range similar to other models
    depth_min = depth.min()
    depth_max = depth.max()
    
    if depth_max - depth_min > 0:
        depth = (depth - depth_min) / (depth_max - depth_min)
        
    # Invert to match disparity (closer is brighter)
    disp = 1.0 - depth
    
    # Add batch dimension to match other models
    disp = torch.from_numpy(disp).unsqueeze(0).unsqueeze(0).to(device)
    
    return disp


def evaluate_models(models, test_images, device, args):
    """Evaluate all models on test images"""
    results = {
        "model": [],
        "source": [],
        "image": [],
        "processing_time": [],
        "edge_preservation": [],
        "detail_preservation": [],
        "noise_level": [],
        "overall_score": []
    }
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create evaluator
    evaluator = MarsTerrainEvaluator(min_depth=0.1, max_depth=100.0)
    
    # Create visualizer
    visualizer = MarsTerrainVisualizer(mars_themed=True)
    
    # Process each source and image
    for source, image_paths in test_images.items():
        print(f"\nProcessing {len(image_paths)} {source} images...")
        
        # Create source directory
        source_dir = os.path.join(args.output_dir, source)
        os.makedirs(source_dir, exist_ok=True)
        
        for img_idx, image_path in enumerate(tqdm(image_paths, desc=f"Processing {source} images")):
            image_name = os.path.basename(image_path)
            img_output_dir = os.path.join(source_dir, os.path.splitext(image_name)[0])
            os.makedirs(img_output_dir, exist_ok=True)
            
            # Load image
            image_pil = Image.open(image_path).convert('RGB')
            
            # Prepare tensor for Monodepth2 and Depth Anything
            image_tensor = torch.from_numpy(np.array(image_pil).transpose((2, 0, 1))).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(device)
            
            # Prepare metadata for Depth Anything
            metadata = prepare_metadata(source, image_path)
            
            # Process with each model
            depth_maps = {}
            
            for model_name, model_data in models.items():
                try:
                    # Skip if model couldn't be loaded
                    if model_data is None or (isinstance(model_data, tuple) and model_data[0] is None):
                        continue
                    
                    # Process based on model type
                    start_time = time.time()
                    
                    if model_name == "depth_anything":
                        disp = predict_depth_anything(model_data, image_tensor, metadata, device)
                    elif model_name.startswith("monodepth2"):
                        encoder, decoder = model_data
                        disp = predict_monodepth2(encoder, decoder, image_tensor, device)
                    elif model_name == "midas":
                        model, transform = model_data
                        disp = predict_midas(model, transform, image_pil, device)
                    elif model_name == "dpt":
                        model, transform = model_data
                        disp = predict_midas(model, transform, image_pil, device)  # Same interface as MiDaS
                    else:
                        print(f"Unknown model: {model_name}")
                        continue
                    
                    # Record processing time
                    processing_time = time.time() - start_time
                    
                    # Convert disparity to depth
                    _, depth = disp_to_depth(disp, min_depth=0.1, max_depth=100.0)
                    
                    # Save to depth maps
                    depth_maps[model_name] = depth
                    
                    # Compute proxy metrics
                    depth_np = depth.squeeze().cpu().numpy()
                    metrics = evaluator.compute_proxy_metrics(depth_np)
                    
                    # Calculate overall score
                    overall_score = metrics["edge_preservation_score"] + metrics["detail_preservation_score"] - metrics["noise_level"]
                    
                    # Save depth visualization
                    vis_path = os.path.join(img_output_dir, f"{model_name}_depth.png")
                    visualizer.visualize_depth_map(image_pil, depth, save_path=vis_path, show=False)
                    
                    # Record results
                    results["model"].append(model_name)
                    results["source"].append(source)
                    results["image"].append(image_name)
                    results["processing_time"].append(processing_time)
                    results["edge_preservation"].append(metrics["edge_preservation_score"])
                    results["detail_preservation"].append(metrics["detail_preservation_score"])
                    results["noise_level"].append(metrics["noise_level"])
                    results["overall_score"].append(overall_score)
                    
                except Exception as e:
                    print(f"Error processing {image_path} with {model_name}: {e}")
                    continue
            
            # Create comparison visualization
            if len(depth_maps) > 1:
                comparison_path = os.path.join(img_output_dir, "depth_comparison.png")
                create_comparison_visualization(image_pil, depth_maps, comparison_path)
            
            # If terrain comparison enabled, compare 3D reconstructions
            if args.compare_terrain and len(depth_maps) > 1:
                terrain_dir = os.path.join(img_output_dir, "terrain")
                os.makedirs(terrain_dir, exist_ok=True)
                
                # Create terrain reconstructor
                reconstructor = TerrainReconstructor(min_depth=0.1, max_depth=100.0)
                
                # Generate terrain for each model
                for model_name, depth in depth_maps.items():
                    try:
                        # Convert depth to point cloud
                        points, colors = reconstructor.depth_to_point_cloud(depth, None, image_tensor)
                        
                        # Create point cloud
                        pcd = reconstructor.create_open3d_point_cloud(points, colors)
                        
                        # Save point cloud
                        pcd_path = os.path.join(terrain_dir, f"{model_name}_terrain.ply")
                        reconstructor.save_point_cloud(pcd, pcd_path)
                        
                    except Exception as e:
                        print(f"Error creating terrain for {model_name}: {e}")
                        continue
    
    # Convert results to dataframe
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(os.path.join(args.output_dir, "benchmark_results.csv"), index=False)
    
    # Create summary visualizations
    create_summary_plots(results_df, args.output_dir)
    
    return results_df


def create_comparison_visualization(image, depth_maps, output_path):
    """Create a visual comparison of different depth maps"""
    num_models = len(depth_maps) + 1  # +1 for the input image
    
    # Create figure
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 5))
    
    # Plot input image
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    
    # Plot depth maps
    for i, (model_name, depth) in enumerate(depth_maps.items(), 1):
        # Convert to numpy
        depth_np = depth.squeeze().cpu().numpy()
        
        # Show depth map
        axes[i].imshow(depth_np, cmap='magma')
        axes[i].set_title(f"{model_name}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_summary_plots(results_df, output_dir):
    """Create summary plots for benchmark results"""
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Performance comparison by model
    plt.figure(figsize=(10, 6))
    performance = results_df.groupby('model')['processing_time'].mean().sort_values()
    performance.plot(kind='bar')
    plt.title("Average Processing Time by Model")
    plt.ylabel("Time (seconds)")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "processing_time_comparison.png"))
    plt.close()
    
    # 2. Quality metrics by model
    plt.figure(figsize=(12, 8))
    metrics = ['edge_preservation', 'detail_preservation', 'overall_score']
    quality = results_df.groupby('model')[metrics].mean()
    quality.plot(kind='bar')
    plt.title("Quality Metrics by Model")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "quality_metrics_comparison.png"))
    plt.close()
    
    # 3. Performance by source type
    plt.figure(figsize=(12, 8))
    source_perf = results_df.pivot_table(
        index='model', 
        columns='source', 
        values='overall_score',
        aggfunc='mean'
    )
    source_perf.plot(kind='bar')
    plt.title("Model Performance by Source Type")
    plt.ylabel("Overall Score")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "source_type_comparison.png"))
    plt.close()
    
    # 4. Noise level comparison
    plt.figure(figsize=(10, 6))
    noise = results_df.groupby('model')['noise_level'].mean().sort_values()
    noise.plot(kind='bar')
    plt.title("Average Noise Level by Model")
    plt.ylabel("Noise Level (lower is better)")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "noise_level_comparison.png"))
    plt.close()
    
    # 5. Feature preservation score
    plt.figure(figsize=(10, 6))
    feature = results_df.groupby('model')['detail_preservation'].mean().sort_values(ascending=False)
    feature.plot(kind='bar')
    plt.title("Feature Preservation by Model")
    plt.ylabel("Detail Preservation Score")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "feature_preservation_comparison.png"))
    plt.close()


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Collect test images
    test_images = collect_test_images(args)
    
    # Load selected models
    print("\nLoading models...")
    models = {}
    
    if "depth_anything" in args.models:
        models["depth_anything"] = load_depth_anything_model(
            args.depth_anything_weights, 
            args.mars_weights, 
            device
        )
    
    if "monodepth2_mono" in args.models:
        models["monodepth2_mono"] = load_monodepth2_model("mono", device)
    
    if "monodepth2_stereo" in args.models:
        models["monodepth2_stereo"] = load_monodepth2_model("stereo", device)
    
    if "monodepth2_mono_stereo" in args.models:
        models["monodepth2_mono_stereo"] = load_monodepth2_model("mono_stereo", device)
    
    if "midas" in args.models:
        models["midas"] = try_load_midas_model(device)
    
    if "dpt" in args.models:
        models["dpt"] = try_load_dpt_model(device)
    
    # Filter out models that failed to load
    models = {k: v for k, v in models.items() if v is not None and not (isinstance(v, tuple) and v[0] is None)}
    
    if not models:
        print("No models could be loaded. Exiting.")
        return
    
    print(f"Successfully loaded {len(models)} models for evaluation")
    
    # Run evaluation
    print("\nEvaluating models...")
    results = evaluate_models(models, test_images, device, args)
    
    # Print summary of results
    print("\nEvaluation Results:")
    print("\nAverage processing time (seconds):")
    print(results.groupby('model')['processing_time'].mean())
    
    print("\nAverage quality score:")
    print(results.groupby('model')['overall_score'].mean())
    
    print("\nBenchmark completed successfully!")
    print(f"Detailed results saved to {os.path.join(args.output_dir, 'benchmark_results.csv')}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
