from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3x3(nn.Module):
    """Layer to pad and convolve input"""
    def __init__(self, in_channels, out_channels):
        super(Conv3x3, self).__init__()

        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3)
        )

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by activation"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            Conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DepthAnythingDecoder(nn.Module):
    """
    Depth Anything decoder for Mars terrain depth estimation with fixed channel handling
    """
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthAnythingDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.scales = scales

        # Channel dimensions for decoder - progressive reduction
        self.num_ch_dec = np.array([256, 128, 64, 32, 16])
        
        # Debug mode helps trace dimensions
        self.debug = False

        # Convolutions dictionary
        self.convs = nn.ModuleDict()
        
        # Initial convolution to process encoder output
        self.convs["init_conv"] = ConvBlock(num_ch_enc[-1], self.num_ch_dec[0])
        
        # Create decoder blocks
        for i in range(4):
            # First upconv in each decoder block
            input_ch = self.num_ch_dec[i] if i == 0 else self.num_ch_dec[i]
            output_ch = self.num_ch_dec[i+1]
            
            self.convs[f"upconv_{i}_0"] = ConvBlock(input_ch, output_ch)
            
            # Skip connection handlers - fixed to avoid dimension mismatches
            if self.use_skips and i < len(num_ch_enc)-1:
                skip_ch = num_ch_enc[-(i+2)] if i+2 <= len(num_ch_enc) else num_ch_enc[0]
                self.convs[f"skip_{i}"] = ConvBlock(skip_ch, output_ch)
            
            # Second upconv after potential skip connection
            input_ch = output_ch
            if self.use_skips and i < len(num_ch_enc)-1:
                input_ch *= 2  # Double channels after skip connection
                
            self.convs[f"upconv_{i}_1"] = ConvBlock(input_ch, output_ch)
        
        # Final adaptive convolution for Mars terrain
        self.convs["mars_adapt"] = ConvBlock(self.num_ch_dec[-1], self.num_ch_dec[-1])
        
        # Output convolutions for each scale
        for s in self.scales:
            if s < len(self.num_ch_dec):  # Ensure scale is valid
                self.convs[f"dispconv_{s}"] = nn.Conv2d(
                    self.num_ch_dec[s+1], self.num_output_channels, 3, padding=1
                )

    def forward(self, input_features):
        """
        Forward pass for depth decoder
        """
        outputs = {}
        
        # Start with encoder features
        x = input_features[-1]
        
        if self.debug:
            print(f"Initial x shape: {x.shape}")
            
        # Initial convolution
        x = self.convs["init_conv"](x)
        
        # Process through decoder blocks
        for i in range(4):
            if self.debug:
                print(f"\nLevel {i}")
                print(f"Pre-upconv shape: {x.shape}")
                
            # First upconv
            x = self.convs[f"upconv_{i}_0"](x)
            
            if self.debug:
                print(f"After first upconv: {x.shape}")
            
            # Upsample for next level
            x_size = x.shape[-2:]
            target_size = (x_size[0] * 2, x_size[1] * 2)
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
            
            if self.debug:
                print(f"After upsampling: {x.shape}")
            
            # Add skip connection if applicable
            if self.use_skips and i < len(input_features)-1:
                # Get skip features - safely
                skip_idx = min(i+2, len(input_features))
                skip_feats = input_features[-skip_idx]
                
                if self.debug:
                    print(f"Skip features shape: {skip_feats.shape}")
                
                # Process skip connection
                skip_processed = self.convs[f"skip_{i}"](skip_feats)
                
                # Ensure dimensions match
                if skip_processed.shape[2:] != x.shape[2:]:
                    skip_processed = F.interpolate(
                        skip_processed, size=x.shape[2:], 
                        mode='bilinear', align_corners=True
                    )
                    
                if self.debug:
                    print(f"Processed skip shape: {skip_processed.shape}")
                
                # Concatenate along channel dimension
                x = torch.cat([x, skip_processed], dim=1)
                
                if self.debug:
                    print(f"After concatenation: {x.shape}")
            
            # Second upconv
            x = self.convs[f"upconv_{i}_1"](x)
            
            if self.debug:
                print(f"After second upconv: {x.shape}")
            
            # Apply Mars terrain enhancement at final level
            if i == 3:
                x = self.convs["mars_adapt"](x)
                
            # Generate outputs at required scales
            if i in self.scales and i < len(self.num_ch_dec)-1:
                # Get disparity prediction
                disp = self.convs[f"dispconv_{i}"](x)
                disp = torch.sigmoid(disp)  # Constrain to [0,1]
                outputs[("disp", i)] = disp
                
                if self.debug:
                    print(f"Output disp_{i} shape: {disp.shape}")
        
        return outputs
