#!/usr/bin/env python
"""
Benchmark script to compare the performance of different depth estimation models
on Mars terrain imagery.
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import model components
from models.model import DepthAnything
from models.enhanced_model import EnhancedDepthAnything
from utils.terrain_reconstruction_fixed import MarsTerrainMapper
from utils.evaluation import MarsTerrainEvaluator
from utils.advanced_visualization import AdvancedMarsVisualizer

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Mars Terrain Depth Estimation Models"
    )
    parser.add_argument(
        "--input_dir", type=str, default=None, 
        help="Directory containing Mars images for benchmarking"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./benchmark_results",
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--max_size", type=int, default=1024,
        help="Maximum image size (preserves aspect ratio)"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Visualize comparison results"
    )
    parser.add_argument(
        "--sample_count", type=int, default=10,
        help="Number of images to sample (0 for all)"
    )
    
    return parser.parse_args()

def load_images(input_dir, max_size, sample_count=0):
    """Load images from directory"""
    image_paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        image_paths.extend([str(p) for p in Path(input_dir).glob(ext)])
    
    # Sample if requested
    if sample_count > 0 and sample_count < len(image_paths):
        import random
        image_paths = random.sample(image_paths, sample_count)
    
    print(f"Loading {len(image_paths)} images for benchmarking...")
    
    images = []
    for img_path in tqdm(image_paths):
        img = Image.open(img_path).convert("RGB")
        
        # Resize if needed
        if max(img.size) > max_size:
            scale = max_size / max(img.size)
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.LANCZOS)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(np.array(img).transpose((2, 0, 1))).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        
        images.append({
            'path': img_path,
            'tensor': img_tensor,
            'pil': img
        })
    
    return images

def benchmark_models(images, args):
    """Run benchmark comparison on different models"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize models
    print("Loading models...")
    standard_model = DepthAnything(pretrained=True)
    enhanced_model = EnhancedDepthAnything(pretrained=True)
    
    # Move models to device
    standard_model.to(device).eval()
    enhanced_model.to(device).eval()
    
    # Initialize evaluator
    evaluator = MarsTerrainEvaluator()
    
    # Initialize visualizer if needed
    if args.visualize:
        visualizer = AdvancedMarsVisualizer()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Benchmark results
    results = {
        'standard': {'time': [], 'metrics': []},
        'enhanced': {'time': [], 'metrics': []}
    }
    
    # Process each image
    for i, img_data in enumerate(tqdm(images, desc="Processing images")):
        img_name = Path(img_data['path']).stem
        img_tensor = img_data['tensor'].to(device)
        
        # Prepare metadata - try to infer source from filename
        source_type = "rover"  # Default
        path_lower = img_data['path'].lower()
        if "rover" in path_lower:
            source_type = "rover"
        elif "ingenuity" in path_lower or "aerial" in path_lower:
            source_type = "ingenuity"
        elif "satellite" in path_lower or "orbital" in path_lower:
            source_type = "satellite"
            
        metadata = {"source": source_type}
        
        # Process with standard model
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        with torch.no_grad():
            std_outputs = standard_model(img_tensor, metadata)
            std_disp = std_outputs[("disp", 0)]
            std_depth = 1.0 / std_disp.clamp(min=1e-6)
            
        torch.cuda.synchronize() if device.type == 'cuda' else None
        std_time = time.time() - start_time
        results['standard']['time'].append(std_time)
        
        # Process with enhanced model
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        with torch.no_grad():
            enh_outputs = enhanced_model(img_tensor, metadata)
            enh_disp = enh_outputs[("disp", 0)]
            enh_depth = 1.0 / enh_disp.clamp(min=1e-6)
            enh_uncertainty = enh_outputs.get(("uncertainty", 0), None)
            
        torch.cuda.synchronize() if device.type == 'cuda' else None
        enh_time = time.time() - start_time
        results['enhanced']['time'].append(enh_time)
        
        # Convert to numpy for metrics and visualization
        std_depth_np = std_depth.cpu().squeeze().numpy()
        enh_depth_np = enh_depth.cpu().squeeze().numpy()
        enh_uncertainty_np = enh_uncertainty.cpu().squeeze().numpy() if enh_uncertainty is not None else None
        
        # Calculate qualitative metrics
        # (Without ground truth, we can only use no-reference metrics)
        std_metrics = {
            'depth_range': np.ptp(std_depth_np),
            'depth_std': np.std(std_depth_np),
            'depth_entropy': evaluator._calculate_entropy(std_depth_np),
            'edge_score': evaluator._calculate_edge_preservation(std_depth_np, img_data['pil']),
        }
        
        enh_metrics = {
            'depth_range': np.ptp(enh_depth_np),
            'depth_std': np.std(enh_depth_np),
            'depth_entropy': evaluator._calculate_entropy(enh_depth_np),
            'edge_score': evaluator._calculate_edge_preservation(enh_depth_np, img_data['pil']),
        }
        
        results['standard']['metrics'].append(std_metrics)
        results['enhanced']['metrics'].append(enh_metrics)
        
        # Visualize comparison if requested
        if args.visualize:
            result_dir = os.path.join(args.output_dir, img_name)
            os.makedirs(result_dir, exist_ok=True)
            
            # Create comparison visualization
            plt.figure(figsize=(20, 10))
            
            # Original image
            plt.subplot(2, 3, 1)
            plt.imshow(img_data['pil'])
            plt.title("Original Image")
            plt.axis('off')
            
            # Standard model depth
            plt.subplot(2, 3, 2)
            plt.imshow(std_depth_np, cmap='viridis')
            plt.title(f"Standard Model Depth\nTime: {std_time:.3f}s")
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')
            
            # Enhanced model depth
            plt.subplot(2, 3, 3)
            plt.imshow(enh_depth_np, cmap='viridis')
            plt.title(f"Enhanced Model Depth\nTime: {enh_time:.3f}s")
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')
            
            # Depth difference
            plt.subplot(2, 3, 5)
            diff = np.abs(std_depth_np - enh_depth_np)
            plt.imshow(diff, cmap='hot')
            plt.title("Depth Difference")
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')
            
            # Uncertainty (if available)
            if enh_uncertainty_np is not None:
                plt.subplot(2, 3, 6)
                plt.imshow(enh_uncertainty_np, cmap='plasma')
                plt.title("Enhanced Model Uncertainty")
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, "model_comparison.png"))
            plt.close()
            
            # Create advanced visualization for the enhanced model
            visualizer.create_multi_view_visualization(
                img_data['pil'], enh_depth_np, uncertainty=enh_uncertainty_np,
                save_path=os.path.join(result_dir, "enhanced_multiview.png"),
                show=False
            )
    
    # Calculate average performance metrics
    avg_results = {
        'standard': {
            'avg_time': np.mean(results['standard']['time']),
            'avg_metrics': {k: np.mean([m[k] for m in results['standard']['metrics']]) 
                           for k in results['standard']['metrics'][0]}
        },
        'enhanced': {
            'avg_time': np.mean(results['enhanced']['time']),
            'avg_metrics': {k: np.mean([m[k] for m in results['enhanced']['metrics']]) 
                           for k in results['enhanced']['metrics'][0]}
        }
    }
    
    # Create summary report
    summary_path = os.path.join(args.output_dir, "benchmark_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== Mars Terrain Depth Estimation Benchmark Results ===\n\n")
        f.write(f"Number of images: {len(images)}\n")
        f.write(f"Image max size: {args.max_size}\n")
        f.write(f"Device: {device}\n\n")
        
        f.write("--- Standard Model Performance ---\n")
        f.write(f"Average inference time: {avg_results['standard']['avg_time']:.4f} seconds\n")
        for k, v in avg_results['standard']['avg_metrics'].items():
            f.write(f"Average {k}: {v:.4f}\n")
        f.write("\n")
        
        f.write("--- Enhanced Model Performance ---\n")
        f.write(f"Average inference time: {avg_results['enhanced']['avg_time']:.4f} seconds\n")
        for k, v in avg_results['enhanced']['avg_metrics'].items():
            f.write(f"Average {k}: {v:.4f}\n")
        f.write("\n")
        
        # Calculate improvements
        time_diff = avg_results['standard']['avg_time'] - avg_results['enhanced']['avg_time']
        time_pct = (time_diff / avg_results['standard']['avg_time']) * 100
        
        f.write("--- Performance Comparison ---\n")
        f.write(f"Time difference: {time_diff:.4f} seconds ({time_pct:.1f}%)\n")
        
        for k in avg_results['standard']['avg_metrics'].keys():
            std_val = avg_results['standard']['avg_metrics'][k]
            enh_val = avg_results['enhanced']['avg_metrics'][k]
            diff = enh_val - std_val
            pct = (diff / std_val) * 100 if std_val != 0 else 0
            f.write(f"{k} improvement: {diff:.4f} ({pct:.1f}%)\n")
    
    print(f"Benchmark completed. Results saved to {args.output_dir}")
    print(f"Summary report: {summary_path}")
    
    # Plot performance comparison
    plt.figure(figsize=(12, 8))
    
    # Plot inference time comparison
    plt.subplot(2, 2, 1)
    plt.bar(['Standard Model', 'Enhanced Model'], 
           [avg_results['standard']['avg_time'], avg_results['enhanced']['avg_time']])
    plt.title("Average Inference Time (seconds)")
    plt.grid(axis='y', alpha=0.3)
    
    # Plot key metrics
    metrics_to_plot = ['edge_score', 'depth_entropy']
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 2, i+2)
        plt.bar(['Standard Model', 'Enhanced Model'], 
               [avg_results['standard']['avg_metrics'][metric], 
                avg_results['enhanced']['avg_metrics'][metric]])
        plt.title(f"Average {metric}")
        plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "performance_comparison.png"))
    
    if args.visualize:
        plt.show()
    else:
        plt.close()
    
    return results

def main():
    args = parse_args()
    
    if args.input_dir is None:
        print("Error: Please provide an input directory.")
        return
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
        return
    
    # Load images
    images = load_images(args.input_dir, args.max_size, args.sample_count)
    
    if not images:
        print("Error: No images found in the input directory.")
        return
    
    # Run benchmark
    benchmark_models(images, args)

if __name__ == "__main__":
    main()
