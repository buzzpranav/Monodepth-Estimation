# Mars Terrain Depth Estimation

A robust implementation of depth estimation for Mars terrain imagery based on the [Depth Anything](https://github.com/LiheYoung/Depth-Anything) architecture.

## Overview

This repository provides depth estimation for Mars terrain images from various sources (rover, aerial/ingenuity, satellite). The implementation has been optimized to handle the specific challenges of Mars terrain and produces accurate depth maps without gradient artifacts.

## Key Features

- Robust depth estimation that works with various image sizes and sources
- Source-specific depth scaling for rover, ingenuity/aerial, and satellite imagery
- Terrain reconstruction utilities for 3D visualization
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
python run_mars_depth.py --input path/to/image_or_folder --source auto
```

### Command Line Arguments

- `--input`: Path to an image or directory containing images
- `--output_dir`: Directory to save results (default: ./outputs)
- `--source`: Source type (auto, rover, ingenuity, satellite)
- `--max_size`: Maximum image size (default: 1024, preserves aspect ratio)
- `--show`: Show results interactively
- `--terrain_reconstruction`: Generate 3D terrain reconstruction

## Project Structure

The project has a clean, modular structure:

```
models/
  ├── encoder.py         # Vision Transformer based encoder
  ├── decoder.py         # Depth decoder with skip connections
  └── model.py           # Full depth estimation model
  
utils/
  └── terrain_reconstruction_fixed.py  # Terrain reconstruction utilities
  
run_mars_depth.py        # Main script for running depth estimation
requirements.txt         # Dependencies
README.md                # This file
```

## How It Works

1. The model uses a Vision Transformer (ViT) encoder to extract features from Mars images
2. A specialized decoder processes these features with skip connections to generate depth maps
3. Source-specific scaling is applied based on metadata (rover, ingenuity, satellite)
4. Optional terrain reconstruction provides 3D visualization and terrain type classification

## Example Results

Input | Depth Map | 3D Reconstruction
:-------------------------:|:-------------------------:|:-------------------------:
![](assets/example_input.png) | ![](assets/example_depth.png) | ![](assets/example_terrain.png)

## License

[MIT License](LICENSE)

## Acknowledgements

- The [Depth Anything](https://github.com/LiheYoung/Depth-Anything) project for the core architecture
- NASA and JPL for Mars imagery used in testing
