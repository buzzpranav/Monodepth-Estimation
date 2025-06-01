from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import DepthAnythingEncoder
from models.decoder import DepthAnythingDecoder
from layers import disp_to_depth

class SelfAttentionBlock(nn.Module):
    """Self-attention block for enhancing feature extraction"""
    def __init__(self, in_channels):
        super(SelfAttentionBlock, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, height, width = x.size()
        
        # Reshape for matrix multiplication
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B x HW x C'
        key = self.key(x).view(batch_size, -1, height * width)  # B x C' x HW
        
        # Calculate attention map
        energy = torch.bmm(query, key)  # B x HW x HW
        attention = F.softmax(energy, dim=2)
        
        # Apply attention to value
        value = self.value(x).view(batch_size, -1, height * width)  # B x C x HW
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x HW
        out = out.view(batch_size, C, height, width)
        
        # Add residual connection with learnable weight
        out = self.gamma * out + x
        
        return out

class UncertaintyModule(nn.Module):
    """Module to estimate uncertainty in depth predictions"""
    def __init__(self, in_channels):
        super(UncertaintyModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        self.activation = nn.ELU()
        
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        uncertainty = torch.sigmoid(self.conv3(x))  # Normalize to [0, 1]
        return uncertainty

class MultiScaleFeatureFusion(nn.Module):
    """Fuses features from multiple scales for better detail preservation"""
    def __init__(self, scales=[0, 1, 2, 3]):
        super(MultiScaleFeatureFusion, self).__init__()
        self.scales = scales
        self.weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
        
    def forward(self, features_dict):
        # Get target size from the highest resolution features
        target_scale = min(self.scales)
        target_size = features_dict[target_scale].shape[2:]
        
        # Resize and weigh features from different scales
        weighted_features = []
        softmax_weights = F.softmax(self.weights, dim=0)
        
        for i, scale in enumerate(self.scales):
            if scale in features_dict:
                features = features_dict[scale]
                # Resize if needed
                if features.shape[2:] != target_size:
                    features = F.interpolate(features, size=target_size, mode='bilinear', align_corners=False)
                weighted_features.append(softmax_weights[i] * features)
        
        # Sum all weighted features
        fused_features = sum(weighted_features)
        return fused_features

class TextureAwareRefinement(nn.Module):
    """Refines depth maps using texture cues from input image"""
    def __init__(self, in_channels):
        super(TextureAwareRefinement, self).__init__()
        self.conv1 = nn.Conv2d(in_channels + 3, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        self.activation = nn.ELU()
    def forward(self, depth_features, rgb_image):
        # Ensure RGB input is correctly sized
        if rgb_image.shape[2:] != depth_features.shape[2:]:
            rgb_image = F.interpolate(rgb_image, size=depth_features.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate features
        x = torch.cat([depth_features, rgb_image], dim=1)
        
        # Process through refinement network
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        refinement = self.conv3(x)
        
        # Ensure refinement output has the right dimensions
        if hasattr(self, 'target_size') and self.target_size is not None:
            if refinement.shape[2:] != self.target_size:
                refinement = F.interpolate(
                    refinement, size=self.target_size,
                    mode='bilinear', align_corners=False
                )
        
        return refinement

class EnhancedDepthAnything(nn.Module):
    """
    Enhanced Depth Anything model for Mars terrain depth estimation
    Includes uncertainty estimation, self-attention, and multi-scale fusion
    """
    def __init__(self, pretrained=True, mars_finetuned=False):
        super(EnhancedDepthAnything, self).__init__()
        
        # Base encoder-decoder architecture
        self.encoder = DepthAnythingEncoder(pretrained=pretrained)
        self.decoder = DepthAnythingDecoder(
            num_ch_enc=self.encoder.num_ch_enc,
            scales=range(4),  # Multi-scale output
            num_output_channels=1,
            use_skips=True
        )
        
        # Enhanced components
        
        # Self-attention blocks for selected feature levels
        self.attention_blocks = nn.ModuleDict({
            '1': SelfAttentionBlock(self.encoder.num_ch_enc[1]),  # Mid-level features
            '2': SelfAttentionBlock(self.encoder.num_ch_enc[2])   # Higher-level features
        })
        
        # Multi-scale feature fusion
        self.feature_fusion = MultiScaleFeatureFusion(scales=[0, 1, 2])
        
        # Uncertainty estimation module
        self.uncertainty_module = UncertaintyModule(self.encoder.num_ch_enc[0])
        
        # Texture-aware refinement
        self.refinement = TextureAwareRefinement(self.encoder.num_ch_enc[0])
        
        self.mars_finetuned = mars_finetuned
        self.input_size_warning_shown = False

    def forward(self, x, metadata=None):
        """
        Forward pass for depth and uncertainty prediction
        
        Args:
            x: Input image tensor [B, 3, H, W]
            metadata: Optional dictionary with metadata about the image
                     e.g., {'source': 'rover', 'altitude': 1.5}
                     
        Returns:
            outputs: Dictionary with depth, uncertainty and other outputs
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
        encoder_features = self.encoder(x)
        
        # Apply self-attention to selected feature levels
        enhanced_features = encoder_features.copy()
        for i in [1, 2]:  # Apply to mid and high level features
            if str(i) in self.attention_blocks:
                enhanced_features[i] = self.attention_blocks[str(i)](encoder_features[i])
        
        # Decode to depth maps
        outputs = self.decoder(enhanced_features)
          # Estimate uncertainty
        highest_res_features = encoder_features[0]
        uncertainty = self.uncertainty_module(highest_res_features)
        outputs[("uncertainty", 0)] = uncertainty
        
        # Apply texture-aware refinement
        refinement = self.refinement(highest_res_features, x)
        
        # Apply refinement to the highest resolution disparity map
        if ("disp", 0) in outputs:
            # Ensure refinement matches the size of disparity output
            disp_size = outputs[("disp", 0)].shape[2:]
            refinement_size = refinement.shape[2:]
            if disp_size != refinement_size:
                # Resize refinement to match disparity output
                refinement = F.interpolate(
                    refinement, size=disp_size, 
                    mode='bilinear', align_corners=False
                )
            
            
            # Save unrefined disparity before adding refinement
            outputs[("disp_unrefined", 0)] = outputs[("disp", 0)].clone()
            
            # Apply refinement
            outputs[("disp", 0)] = outputs[("disp", 0)] + refinement
        
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
        
        # Multi-scale fusion for final disparity map
        if all(("disp", scale) in outputs for scale in [0, 1, 2]):
            # Create dictionary of features for fusion
            disp_features = {}
            for scale in range(3):  # Use scales 0, 1, 2
                disp_features[scale] = outputs[("disp", scale)]
                
            # Fuse features
            fused_disp = self.feature_fusion(disp_features)
            
            # Store the fused disparity
            outputs[("disp_fused", 0)] = fused_disp
            
            # Use fused disparity as the final output
            outputs[("disp", 0)] = fused_disp
        
        # Resize outputs to match original input if needed
        for key in outputs.keys():
            if outputs[key].shape[2:] != original_size:
                outputs[key] = F.interpolate(
                    outputs[key], size=original_size,
                    mode='bilinear', align_corners=False
                )
        
        # Convert disparity to depth for convenience
        for key in list(outputs.keys()):
            if key[0] == "disp":
                _, depth = disp_to_depth(outputs[key], min_depth=0.1, max_depth=100.0)
                outputs[("depth", key[1])] = depth
        
        return outputs
