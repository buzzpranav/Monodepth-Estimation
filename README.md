# Mars Terrain Depth Estimation

A robust implementation of depth estimation for Mars terrain imagery based on the [Depth Anything](https://github.com/LiheYoung/Depth-Anything) architecture, with enhanced accuracy and advanced visualizations.

## Overview

This repository provides depth estimation for Mars terrain images from various sources (rover, aerial/ingenuity, satellite). The implementation has been optimized to handle the specific challenges of Mars terrain and produces accurate depth maps without gradient artifacts.

## Key Features

- Robust depth estimation that works with various image sizes and sources
- Enhanced model with self-attention and uncertainty estimation
- Multi-scale depth fusion for improved accuracy
- Texture-aware depth refinement specific to Mars terrain
- Source-specific depth scaling for rover, ingenuity/aerial, and satellite imagery
- Advanced visualization tools including interactive 3D, terrain features, and flyover animations
- Comprehensive terrain reconstruction utilities
- Uncertainty visualization and confidence metrics
- No gradient pattern fallbacks - ensures valid depth maps or explicit errors

## Installation

```bash
# Clone the repository
git clone https://github.com/buzzpranav/monodepth-estimation.git
cd monodepth-estimation

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run depth estimation on Mars images:

```bash
# Standard depth estimation
python run_mars_depth.py --input path/to/image_or_folder --source auto

# Enhanced model with uncertainty estimation
python run_mars_depth.py --input path/to/image_or_folder --source auto --enhanced_model

# Advanced visualizations
python run_mars_depth.py --input path/to/image_or_folder --enhanced_model --visualizations advanced

# Generate 3D terrain flyover animation
python run_mars_depth.py --input path/to/image_or_folder --enhanced_model --visualizations all --flyover
```

### Command Line Arguments

- `--input`: Path to an image or directory containing images
- `--output_dir`: Directory to save results (default: ./outputs)
- `--source`: Source type (auto, rover, ingenuity, satellite)
- `--max_size`: Maximum image size (default: 1024, preserves aspect ratio)
- `--show`: Show results interactively
- `--terrain_reconstruction`: Generate 3D terrain reconstruction

### Enhanced Features

- `--enhanced_model`: Use the enhanced depth model with self-attention and uncertainty
- `--visualizations`: Visualization type to generate (standard, advanced, all, interactive, anaglyph, terrain_features)
- `--flyover`: Generate terrain flyover animation
- `--multi_scale_fusion`: Enable multi-scale fusion for improved accuracy
- `--benchmark`: Run benchmark to compare standard and enhanced models

## Project Structure

The project has a clean, modular structure:

```
models/
  ├── encoder.py                # Vision Transformer based encoder
  ├── decoder.py                # Depth decoder with skip connections
  ├── model.py                  # Standard depth estimation model
  ├── enhanced_model.py         # Enhanced model with self-attention and uncertainty
  └── refinement.py             # Texture-aware refinement and multi-scale fusion
  
utils/
  ├── terrain_reconstruction_fixed.py  # Terrain reconstruction utilities
  ├── advanced_visualization.py        # Advanced visualization tools
  └── evaluation.py                    # Metrics and evaluation utilities
  
scripts/
  └── benchmark_depth_models.py        # Benchmarking script for model comparison
  
run_mars_depth.py        # Main script for running depth estimation
requirements.txt         # Dependencies
README.md                # This file
```

## How It Works

### Standard Model
1. The model uses a Vision Transformer (ViT) encoder to extract features from Mars images
2. A specialized decoder processes these features with skip connections to generate depth maps
3. Source-specific scaling is applied based on metadata (rover, ingenuity, satellite)
4. Optional terrain reconstruction provides 3D visualization

### Enhanced Model
1. Self-attention blocks enhance feature extraction for better detail preservation
2. Multi-scale feature fusion combines information from multiple resolutions
3. Uncertainty estimation provides confidence maps for depth predictions
4. Texture-aware refinement improves depth accuracy using Mars terrain cues
5. Advanced visualization tools provide detailed terrain analysis and interactive 3D views

## Example Results

### Standard Model
Input | Depth Map | 3D Reconstruction
:-------------------------:|:-------------------------:|:-------------------------:
![](assets/example_input.png) | ![](assets/example_depth.png) | ![](assets/example_terrain.png)

### Enhanced Model with Advanced Visualizations

#### Multi-view Visualization
![Multi-view visualization showing depth, uncertainty, and 3D terrain](assets/enhanced_multiview.png)

#### Terrain Feature Analysis
![Terrain feature analysis showing slope, roughness, and curvature](assets/terrain_features.png)

#### Interactive 3D Visualization
Interactive 3D visualizations and flyover animations provide detailed terrain exploration capabilities.

## Benchmarks

The enhanced model provides significant improvements in depth quality:
- Higher accuracy at depth discontinuities
- Better preservation of fine terrain details
- Confidence metrics through uncertainty estimation
- Faster processing in multi-scale fusion mode
- More detailed 3D reconstructions

## License

[MIT License](LICENSE)

## Acknowledgements

- The [Depth Anything](https://github.com/LiheYoung/Depth-Anything) project for the core architecture
- NASA and JPL for Mars imagery used in testing
- Contributors to open-source visualization libraries
