from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

try:
    from torchvision.models.vision_transformer import ViT_B_16_Weights
except ImportError:
    print("Warning: Could not import ViT_B_16_Weights, using older torchvision version")
    ViT_B_16_Weights = None

class DepthAnythingEncoder(nn.Module):
    """
    Encoder based on Vision Transformer (ViT) architecture
    Adapted from the Depth Anything paper for Mars terrain depth estimation
    """
    def __init__(self, pretrained=True):
        super(DepthAnythingEncoder, self).__init__()
        
        # Define consistent channel dimensions for all layers
        # Using 256 channels throughout for compatibility with decoder
        self.num_ch_enc = np.array([256, 256, 256, 256])
        
        # Import the ViT model architecture
        from torchvision.models.vision_transformer import vit_b_16
        
        # Initialize the ViT encoder
        if pretrained:
            self.encoder = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.encoder = vit_b_16(weights=None)
            
        # Remove the classification head - we only need the feature extractor
        self.encoder.heads = nn.Identity()
        
        # Create hooks to get intermediate features
        self.hooks = []
        self.features = []
        # Extract features from early, middle, and late transformer blocks
        self.hook_layer_indices = [2, 5, 8, 11]
        
        for i in self.hook_layer_indices:
            hook = self.encoder.encoder.layers[i].register_forward_hook(
                lambda module, input, output, idx=i: self._feature_hook(output, idx)
            )
            self.hooks.append(hook)
            
        # Channel reduction layers to match decoder input dimensions
        self.channel_reduce = nn.ModuleList([
            nn.Conv2d(768, 256, 1) for _ in range(4)
        ])
        
        # Add input normalization
        self.normalize = nn.BatchNorm2d(3)
            
    def _feature_hook(self, output, layer_idx):
        """Hook to capture intermediate features"""
        if isinstance(output, tuple):
            feature_tensor = output[0]
        else:
            feature_tensor = output
            
        self.features.append((layer_idx, feature_tensor))
    
    def reshape_features_to_grid(self, features, h, w):
        """
        Reshape transformer token features back to 2D grid format
        
        Args:
            features: Tensor of shape [B, num_tokens, C]
            h, w: Height and width of the reshaped grid
            
        Returns:
            Tensor of shape [B, C, h, w]
        """
        b, tokens, c = features.shape
        
        # Remove CLS token if present
        if tokens == h * w + 1:
            features = features[:, 1:, :]
            tokens = tokens - 1
            
        # Check if dimensions match
        if tokens != h * w:
            # If dimensions don't match, adapt as best we can
            new_h = int(np.sqrt(tokens))
            new_w = tokens // new_h
            
            # Make sure we account for all tokens
            if new_h * new_w < tokens:
                new_w += 1
                
            # Pad or truncate as needed
            if new_h * new_w > tokens:
                padding = new_h * new_w - tokens
                features = torch.cat([features, torch.zeros(b, padding, c, device=features.device)], dim=1)
            
            h, w = new_h, new_w
            
        # Reshape to [B, h, w, C]
        features = features.reshape(b, h, w, c)
        
        # Permute to [B, C, h, w] for convolution operations
        features = features.permute(0, 3, 1, 2)
        
        return features
        
    def forward(self, x):
        """
        Forward pass of the encoder
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            list of feature maps at different scales
        """
        # Clear previous features
        self.features = []
        
        # Get input dimensions
        b, c, h, w = x.shape
        
        # Apply input normalization
        x = self.normalize(x)
        
        # Ensure input is in expected range for ViT [-1, 1]
        if x.max() > 1.0:
            x = x / 255.0
        if x.min() >= 0 and x.max() <= 1.0:
            x = 2.0 * x - 1.0
        
        # Forward pass through ViT
        # Handle different input sizes - ViT requires 224x224 inputs
        original_size = x.shape[2:]
        if original_size[0] != 224 or original_size[1] != 224:
            x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            self.encoder(x_resized)
        else:
            self.encoder(x)
        
        # Sort and process features
        self.features.sort(key=lambda x: x[0])
        feature_tensors = [feat for _, feat in self.features]
        
        # Process feature tensors into feature maps
        patch_size = self.encoder.patch_size if hasattr(self.encoder, 'patch_size') else 16
        feature_h = h // patch_size
        feature_w = w // patch_size
        
        output_features = []
        for i, feat in enumerate(feature_tensors):
            # Only process if we have a valid tensor
            if isinstance(feat, torch.Tensor):
                if len(feat.shape) == 3:  # [B, tokens, C]
                    # Reshape from token sequence to feature grid
                    grid_feat = self.reshape_features_to_grid(feat, feature_h, feature_w)
                    
                    # Apply channel reduction to match decoder
                    reduced_feat = self.channel_reduce[i](grid_feat)
                    
                    # Apply feature normalization
                    reduced_feat = F.instance_norm(reduced_feat)
                    output_features.append(reduced_feat)
                    
                elif len(feat.shape) == 4:  # Already [B, C, H, W]
                    # Apply channel reduction
                    reduced_feat = self.channel_reduce[i](feat)
                    reduced_feat = F.instance_norm(reduced_feat)
                    output_features.append(reduced_feat)
            
        # If we don't have enough features, duplicate the last one
        # But this should never happen with our defined hooks
        while len(output_features) < 4:
            if output_features:
                output_features.append(output_features[-1])
            else:
                # Create initialized feature map if none exist
                feat = torch.zeros(b, self.num_ch_enc[0], feature_h, feature_w, device=x.device)
                nn.init.xavier_normal_(feat)
                output_features.append(feat)
                
        return output_features
