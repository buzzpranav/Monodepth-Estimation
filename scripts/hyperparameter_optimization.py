#!/usr/bin/env python
# script: hyperparameter_optimization.py
"""
Hyperparameter optimization for Mars terrain depth estimation.
This script performs grid search or Bayesian optimization over hyperparameters
to find the best configuration for Mars terrain feature extraction.
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import time
from datetime import datetime
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import glob

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from models.depth_anything_model import DepthAnything
from utils.terrain_reconstruction import MarsTerrainMapper
from utils.visualization import MarsTerrainVisualizer
from utils.evaluation import MarsTerrainEvaluator
from train.mars_adapter import MarsDomainAdapter
from data.mars_dataset import get_mars_dataloaders
from layers import disp_to_depth


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization for Mars Depth Estimation")
    
    # Data arguments
    parser.add_argument("--data_root", type=str, default="../assets",
                        help="Root directory containing Mars imagery")
    parser.add_argument("--output_dir", type=str, default="../optimization_results",
                        help="Output directory for optimization results")
    parser.add_argument("--split_ratio", type=float, default=0.8,
                        help="Train/val split ratio for optimization data")
    
    # Optimization arguments
    parser.add_argument("--opt_method", type=str, default="bayesian", choices=["grid", "bayesian"],
                        help="Optimization method: grid search or Bayesian")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of trials for Bayesian optimization")
    parser.add_argument("--timeout", type=int, default=7200,  # 2 hours
                        help="Timeout for optimization in seconds")
    
    # Model arguments
    parser.add_argument("--model_weights", type=str, default=None,
                        help="Path to model weights (if None, uses default pretrained weights)")
    parser.add_argument("--save_best", action="store_true",
                        help="Save the best model weights from optimization")
    
    # Hardware settings
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available")
                        
    # Visualization
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations for optimization results")
    
    return parser.parse_args()


def load_model(args, device):
    """Load the depth estimation model"""
    print("Loading Depth Anything model...")
    model = DepthAnything(pretrained=True)
    
    # Load custom weights if specified
    if args.model_weights is not None and os.path.exists(args.model_weights):
        print(f"Loading custom weights from {args.model_weights}")
        model.load_state_dict(torch.load(args.model_weights, map_location=device))
    
    # Move model to device
    model.to(device)
    model.eval()
    
    return model


def create_optimization_dataset(args):
    """Create dataset for optimization from available Mars imagery"""
    print("Preparing optimization dataset...")
    
    sources = ["rover", "satellite", "ingenuity"]
    all_images = []
    
    # Collect images from each source
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
        
        # Add to collection with source information
        for path in image_paths:
            all_images.append({
                "path": path,
                "source": source
            })
    
    # Shuffle images
    np.random.shuffle(all_images)
    
    # Split into train and validation sets
    split_idx = int(len(all_images) * args.split_ratio)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    print(f"Created dataset with {len(train_images)} training and {len(val_images)} validation images")
    
    return train_images, val_images


def objective(trial, model, train_images, val_images, device, args):
    """Objective function for hyperparameter optimization"""
    # Define hyperparameters to optimize
    params = {
        # Depth estimation parameters
        "min_depth": trial.suggest_float("min_depth", 0.01, 1.0, log=True),
        "max_depth": trial.suggest_float("max_depth", 10.0, 200.0),
        
        # Mars-specific parameters
        "rover_scale": trial.suggest_float("rover_scale", 0.5, 2.0),
        "satellite_scale": trial.suggest_float("satellite_scale", 0.01, 0.5),
        "aerial_scale": trial.suggest_float("aerial_scale", 0.1, 1.0),
        
        # Terrain reconstruction parameters
        "voxel_size": trial.suggest_float("voxel_size", 0.01, 0.2),
        "poisson_depth": trial.suggest_int("poisson_depth", 6, 10),
    }
    
    # Create mapper with these parameters
    mapper = MarsTerrainMapper(
        model=model,
        min_depth=params["min_depth"],
        max_depth=params["max_depth"],
        use_cuda=not args.no_cuda
    )
    
    # Create evaluator
    evaluator = MarsTerrainEvaluator(
        min_depth=params["min_depth"],
        max_depth=params["max_depth"]
    )
    
    # Process validation images and compute metrics
    val_metrics = []
    
    for img_info in tqdm(val_images, desc="Evaluating validation images"):
        try:
            # Load image
            image_path = img_info["path"]
            source = img_info["source"]
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Convert to tensor
            img_tensor = torch.from_numpy(np.array(image).transpose((2, 0, 1))).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension
            
            # Prepare metadata with source-specific scale
            metadata = {
                "source": source,
                "path": image_path
            }
            
            if source == "rover":
                metadata["scale_factor"] = params["rover_scale"]
            elif source == "satellite":
                metadata["scale_factor"] = params["satellite_scale"]
                metadata["altitude"] = 400000  # Example altitude
            elif source == "ingenuity":
                metadata["scale_factor"] = params["aerial_scale"]
                metadata["altitude"] = 10  # Example altitude
            
            # Predict depth
            with torch.no_grad():
                outputs = model(img_tensor, metadata)
                
            # Get depth from disparity
            disp = outputs[("disp", 0)]
            _, depth = disp_to_depth(disp, min_depth=params["min_depth"], max_depth=params["max_depth"])
            
            # Compute quality metrics (we don't have ground truth, so use proxy metrics)
            metrics = evaluator.compute_proxy_metrics(depth.squeeze().cpu().numpy())
            metrics["source"] = source
            val_metrics.append(metrics)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    # Compute average metrics
    if not val_metrics:
        return float('inf')  # Return high loss if all images failed
    
    # Calculate average scores per source
    source_scores = {}
    for source in ["rover", "satellite", "ingenuity"]:
        source_metrics = [m for m in val_metrics if m["source"] == source]
        if source_metrics:
            # Combine metrics into a single score
            scores = [
                m["edge_preservation_score"] + 
                m["detail_preservation_score"] -
                m["noise_level"] * 2.0  # Penalize noise more heavily
                for m in source_metrics
            ]
            source_scores[source] = np.mean(scores) if scores else 0.0
    
    # Combine scores across sources with equal weighting
    total_score = sum(source_scores.values()) / max(1, len(source_scores))
    
    # We want to maximize the score, but Optuna minimizes the objective
    return -total_score


def grid_search(model, train_images, val_images, device, args):
    """Perform grid search over hyperparameters"""
    # Define grid search parameters
    param_grid = {
        "min_depth": [0.01, 0.05, 0.1, 0.5],
        "max_depth": [50.0, 100.0, 150.0],
        "rover_scale": [0.5, 1.0, 1.5],
        "satellite_scale": [0.01, 0.05, 0.1],
        "aerial_scale": [0.1, 0.5, 1.0],
        "voxel_size": [0.01, 0.05, 0.1],
        "poisson_depth": [7, 8, 9]
    }
    
    # Calculate total combinations
    total_combinations = np.prod([len(values) for values in param_grid.values()])
    print(f"Grid search with {total_combinations} combinations")
    
    # Initialize best parameters and score
    best_params = None
    best_score = float('-inf')
    all_results = []
    
    # Track start time for timeout
    start_time = time.time()
    timeout = args.timeout
    
    # Generate parameter combinations
    from itertools import product
    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for idx, values in enumerate(tqdm(product(*param_values), total=total_combinations)):
        # Check for timeout
        if timeout and time.time() - start_time > timeout:
            print(f"Grid search timeout after {timeout} seconds")
            break
        
        # Create parameter dictionary
        params = dict(zip(param_keys, values))
        
        # Create mapper with these parameters
        mapper = MarsTerrainMapper(
            model=model,
            min_depth=params["min_depth"],
            max_depth=params["max_depth"],
            use_cuda=not args.no_cuda
        )
        
        # Create evaluator
        evaluator = MarsTerrainEvaluator(
            min_depth=params["min_depth"],
            max_depth=params["max_depth"]
        )
        
        # Process validation images and compute metrics
        val_metrics = []
        
        # Use a subset of validation images for efficiency
        subset_val = val_images[:min(len(val_images), 5)]
        
        for img_info in subset_val:
            try:
                # Load image
                image_path = img_info["path"]
                source = img_info["source"]
                
                # Load image
                image = Image.open(image_path).convert('RGB')
                
                # Convert to tensor
                img_tensor = torch.from_numpy(np.array(image).transpose((2, 0, 1))).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension
                
                # Prepare metadata with source-specific scale
                metadata = {
                    "source": source,
                    "path": image_path
                }
                
                if source == "rover":
                    metadata["scale_factor"] = params["rover_scale"]
                elif source == "satellite":
                    metadata["scale_factor"] = params["satellite_scale"]
                    metadata["altitude"] = 400000  # Example altitude
                elif source == "ingenuity":
                    metadata["scale_factor"] = params["aerial_scale"]
                    metadata["altitude"] = 10  # Example altitude
                
                # Predict depth
                with torch.no_grad():
                    outputs = model(img_tensor, metadata)
                    
                # Get depth from disparity
                disp = outputs[("disp", 0)]
                _, depth = disp_to_depth(disp, min_depth=params["min_depth"], max_depth=params["max_depth"])
                
                # Compute quality metrics (we don't have ground truth, so use proxy metrics)
                metrics = evaluator.compute_proxy_metrics(depth.squeeze().cpu().numpy())
                metrics["source"] = source
                val_metrics.append(metrics)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        # Compute average metrics
        if not val_metrics:
            continue  # Skip this combination if all images failed
        
        # Calculate average scores per source
        source_scores = {}
        for source in ["rover", "satellite", "ingenuity"]:
            source_metrics = [m for m in val_metrics if m["source"] == source]
            if source_metrics:
                # Combine metrics into a single score
                scores = [
                    m["edge_preservation_score"] + 
                    m["detail_preservation_score"] -
                    m["noise_level"] * 2.0  # Penalize noise more heavily
                    for m in source_metrics
                ]
                source_scores[source] = np.mean(scores) if scores else 0.0
        
        # Combine scores across sources with equal weighting
        total_score = sum(source_scores.values()) / max(1, len(source_scores))
        
        # Record results
        result = {
            "params": params,
            "score": total_score,
            "source_scores": source_scores
        }
        all_results.append(result)
        
        # Update best parameters if better
        if total_score > best_score:
            best_score = total_score
            best_params = params
            print(f"New best score: {best_score:.4f} with params: {best_params}")
    
    return best_params, best_score, all_results


def bayesian_optimization(model, train_images, val_images, device, args):
    """Run Bayesian optimization using Optuna"""
    # Create study
    study = optuna.create_study(direction="maximize")
    
    # Create objective function wrapper
    def objective_wrapper(trial):
        return -objective(trial, model, train_images, val_images, device, args)
    
    # Run optimization
    study.optimize(
        objective_wrapper,
        n_trials=args.n_trials,
        timeout=args.timeout
    )
    
    # Get best parameters
    best_params = study.best_params
    best_score = -study.best_value  # Negate since we negated in the wrapper
    
    # Generate visualizations if requested
    if args.visualize:
        viz_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Plot optimization history
        fig = plot_optimization_history(study)
        fig.write_image(os.path.join(viz_dir, "optimization_history.png"))
        
        # Plot parameter importances
        fig = plot_param_importances(study)
        fig.write_image(os.path.join(viz_dir, "param_importances.png"))
    
    return best_params, best_score, study.trials_dataframe()


def visualize_results(best_params, model, val_images, device, args):
    """Visualize results with the best parameters"""
    # Create output directory
    viz_dir = os.path.join(args.output_dir, "result_visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create mapper with best parameters
    mapper = MarsTerrainMapper(
        model=model,
        min_depth=best_params["min_depth"],
        max_depth=best_params["max_depth"],
        use_cuda=not args.no_cuda
    )
    
    # Create visualizer
    visualizer = MarsTerrainVisualizer(
        min_depth=best_params["min_depth"],
        max_depth=best_params["max_depth"],
        mars_themed=True
    )
    
    # Process subset of validation images from each source
    source_images = {}
    for source in ["rover", "satellite", "ingenuity"]:
        source_imgs = [img for img in val_images if img["source"] == source]
        if source_imgs:
            # Take first 3 images of each source
            source_images[source] = source_imgs[:min(3, len(source_imgs))]
    
    # Process and visualize each source
    for source, images in source_images.items():
        print(f"Visualizing results for {source} images...")
        
        # Create a figure for this source
        fig, axes = plt.subplots(len(images), 2, figsize=(12, 4 * len(images)))
        if len(images) == 1:
            axes = np.array([axes])
        
        for i, img_info in enumerate(images):
            try:
                # Load image
                image_path = img_info["path"]
                
                # Load image
                image = Image.open(image_path).convert('RGB')
                
                # Convert to tensor
                img_tensor = torch.from_numpy(np.array(image).transpose((2, 0, 1))).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension
                
                # Prepare metadata with source-specific scale
                metadata = {
                    "source": source,
                    "path": image_path
                }
                
                if source == "rover":
                    metadata["scale_factor"] = best_params["rover_scale"]
                elif source == "satellite":
                    metadata["scale_factor"] = best_params["satellite_scale"]
                    metadata["altitude"] = 400000  # Example altitude
                elif source == "ingenuity":
                    metadata["scale_factor"] = best_params["aerial_scale"]
                    metadata["altitude"] = 10  # Example altitude
                
                # Predict depth
                with torch.no_grad():
                    outputs = model(img_tensor, metadata)
                    
                # Get depth from disparity
                disp = outputs[("disp", 0)]
                _, depth = disp_to_depth(disp, min_depth=best_params["min_depth"], max_depth=best_params["max_depth"])
                
                # Visualize
                # Display input
                axes[i, 0].imshow(image)
                axes[i, 0].set_title(f"Input: {os.path.basename(image_path)}")
                axes[i, 0].axis('off')
                
                # Display depth with optimized parameters
                depth_np = depth.squeeze().cpu().numpy()
                axes[i, 1].imshow(depth_np, cmap='magma')
                axes[i, 1].set_title(f"Optimized Depth")
                axes[i, 1].axis('off')
                
            except Exception as e:
                print(f"Error visualizing {image_path}: {e}")
                continue
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"{source}_optimized_results.png"))
        plt.close()
    
    # Create composite visualization with one example per source
    if len(source_images.keys()) > 1:
        fig, axes = plt.subplots(len(source_images), 2, figsize=(12, 5 * len(source_images)))
        
        for i, (source, images) in enumerate(source_images.items()):
            if not images:
                continue
                
            # Use first image from each source
            img_info = images[0]
            
            # Load image
            image_path = img_info["path"]
            image = Image.open(image_path).convert('RGB')
            
            # Convert to tensor
            img_tensor = torch.from_numpy(np.array(image).transpose((2, 0, 1))).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            # Prepare metadata
            metadata = {
                "source": source,
                "path": image_path
            }
            
            # Apply source-specific scale
            if source == "rover":
                metadata["scale_factor"] = best_params["rover_scale"]
            elif source == "satellite":
                metadata["scale_factor"] = best_params["satellite_scale"]
                metadata["altitude"] = 400000
            elif source == "ingenuity":
                metadata["scale_factor"] = best_params["aerial_scale"]
                metadata["altitude"] = 10
            
            # Predict depth
            with torch.no_grad():
                outputs = model(img_tensor, metadata)
                
            # Get depth
            disp = outputs[("disp", 0)]
            _, depth = disp_to_depth(disp, min_depth=best_params["min_depth"], max_depth=best_params["max_depth"])
            
            # Display input
            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f"{source.capitalize()} Input")
            axes[i, 0].axis('off')
            
            # Display depth
            depth_np = depth.squeeze().cpu().numpy()
            axes[i, 1].imshow(depth_np, cmap='magma')
            axes[i, 1].set_title(f"{source.capitalize()} Depth")
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "all_sources_optimized.png"))
        plt.close()


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args, device)
    
    # Create dataset for optimization
    train_images, val_images = create_optimization_dataset(args)
    
    # Run optimization based on selected method
    print(f"\nRunning {args.opt_method} optimization...")
    
    start_time = time.time()
    
    if args.opt_method == "grid":
        best_params, best_score, all_results = grid_search(
            model, train_images, val_images, device, args
        )
    else:  # bayesian
        best_params, best_score, trial_results = bayesian_optimization(
            model, train_images, val_images, device, args
        )
        all_results = trial_results
    
    optimization_time = time.time() - start_time
    
    print(f"\nOptimization completed in {optimization_time:.2f} seconds")
    print(f"Best score: {best_score:.4f}")
    print("Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Save optimization results
    results_file = os.path.join(args.output_dir, "optimization_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "best_params": best_params,
            "best_score": best_score,
            "optimization_time": optimization_time,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    # Save detailed results
    if args.opt_method == "grid":
        detailed_file = os.path.join(args.output_dir, "all_grid_results.json")
        with open(detailed_file, 'w') as f:
            json.dump(all_results, f, indent=2)
    else:
        detailed_file = os.path.join(args.output_dir, "trials_dataframe.csv")
        all_results.to_csv(detailed_file)
    
    # Visualize results if requested
    if args.visualize:
        print("\nGenerating result visualizations...")
        visualize_results(best_params, model, val_images, device, args)
    
    # Save best model parameters if requested
    if args.save_best:
        # Create custom weight file with optimized Mars parameters
        model_state = model.state_dict()
        
        # Save model with Mars-specific adaptation
        model_file = os.path.join(args.output_dir, "mars_optimized_model.pth")
        torch.save(model_state, model_file)
        
        # Also save hyperparameters for loading
        hyper_file = os.path.join(args.output_dir, "mars_hyperparams.json")
        with open(hyper_file, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        print(f"Best model saved to {model_file}")
    
    print("\nOptimization completed successfully!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
