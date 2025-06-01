from __future__ import absolute_import, division, print_function

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.encoder import DepthAnythingEncoder
from models.decoder import DepthAnythingDecoder

class DepthAnything(nn.Module):
    """
    Depth Anything model for Mars terrain depth estimation
    Based on the Depth Anything architecture (https://github.com/LiheYoung/Depth-Anything)
    Adapted for Mars surface mapping and planetary terrain reconstruction
    """
    def __init__(self, pretrained=True, mars_finetuned=False):
        super(DepthAnything, self).__init__()
        
        self.encoder = DepthAnythingEncoder(pretrained=pretrained)
        self.decoder = DepthAnythingDecoder(
            num_ch_enc=self.encoder.num_ch_enc,
            scales=range(4),  # Multi-scale output
            num_output_channels=1,
            use_skips=True
        )
        
        self.mars_finetuned = mars_finetuned
        self.input_size_warning_shown = False

    def forward(self, x, metadata=None):
        """
        Forward pass for depth prediction
        
        Args:
            x: Input image tensor [B, 3, H, W]
            metadata: Optional dictionary with metadata about the image
                     e.g., {'source': 'rover', 'altitude': 1.5}
        """
        # Get original input dimensions
        original_size = x.shape[2:]
        
        # Check if the dimensions are appropriate
        if min(original_size) < 160 and not self.input_size_warning_shown:
            print("Warning: Input image is very small. For best results, use images of at least 224x224 pixels.")
            self.input_size_warning_shown = True
            
        # Handle different input sizes
        # ViT works best with dimensions divisible by 16
        if x.shape[2] % 16 != 0 or x.shape[3] % 16 != 0:
            h = ((x.shape[2] // 16) + 1) * 16
            w = ((x.shape[3] // 16) + 1) * 16
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        
        # Extract features from encoder
        features = self.encoder(x)
        
        # Decode to depth maps
        outputs = self.decoder(features)
        
        # Apply source-specific depth scaling based on metadata
        if metadata is not None and 'source' in metadata:
            source_type = metadata['source']
            
            if source_type == 'rover':
                # Scale based on rover camera height (typically 1-2m)
                disp_scale_factor = 1.0
                if 'height' in metadata:
                    disp_scale_factor = 2.0 / max(metadata['height'], 0.1)
                    
            elif source_type == 'ingenuity':
                # Scale based on flight altitude
                if 'altitude' in metadata:
                    altitude = metadata['altitude']
                    # Scale factor decreases with altitude
                    disp_scale_factor = 10.0 / max(altitude, 1.0)
                else:
                    disp_scale_factor = 0.5  # Default for aerial
                    
            elif source_type == 'satellite':
                # Much smaller disparity for satellite imagery
                disp_scale_factor = 0.01
                
            else:
                # Default scaling
                disp_scale_factor = 1.0
                
            # Apply scaling to all disparity maps
            for key in outputs.keys():
                if key[0] == "disp":
                    outputs[key] = outputs[key] * disp_scale_factor
        
        # Resize outputs to match original input if needed
        for key in outputs.keys():
            if key[0] == "disp" and outputs[key].shape[2:] != original_size:
                outputs[key] = F.interpolate(
                    outputs[key], size=original_size,
                    mode='bilinear', align_corners=False
                )
        
        return outputs
