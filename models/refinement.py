from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TextureAwareRefinementModule(nn.Module):
    """
    Texture-aware refinement module for improving depth map quality
    using Mars surface texture cues.
    
    This module:
    1. Takes initial depth map and RGB image as input
    2. Extracts texture features from the RGB image
    3. Compares texture consistency with depth predictions
    4. Refines depth map edges and details
    """
    def __init__(self, input_channels=3, depth_channels=1, features=64):
        super(TextureAwareRefinementModule, self).__init__()
        
        # RGB texture feature extraction
        self.rgb_conv1 = nn.Conv2d(input_channels, features, kernel_size=3, padding=1)
        self.rgb_conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        
        # Depth feature extraction
        self.depth_conv1 = nn.Conv2d(depth_channels, features, kernel_size=3, padding=1)
        self.depth_conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        
        # Combined feature processing
        self.combined_conv1 = nn.Conv2d(features*2, features, kernel_size=3, padding=1)
        self.combined_conv2 = nn.Conv2d(features, features//2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(features//2, depth_channels, kernel_size=1)
        
        # Edge awareness features
        self.edge_detect_x = nn.Conv2d(input_channels, 1, kernel_size=3, padding=1, bias=False)
        self.edge_detect_y = nn.Conv2d(input_channels, 1, kernel_size=3, padding=1, bias=False)
        
        # Initialize edge detection kernels (Sobel operators)
        with torch.no_grad():
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            
            # Expand to 3-channel input
            sobel_x = sobel_x.expand(1, input_channels, 3, 3) / input_channels
            sobel_y = sobel_y.expand(1, input_channels, 3, 3) / input_channels
            
            self.edge_detect_x.weight.data = sobel_x
            self.edge_detect_y.weight.data = sobel_y
            
            # Freeze the edge detection weights
            self.edge_detect_x.requires_grad_(False)
            self.edge_detect_y.requires_grad_(False)
        
        self.activation = nn.ELU()
    
    def forward(self, depth_map, rgb_image):
        """
        Forward pass for refining the depth map
        
        Args:
            depth_map: Initial depth map [B, 1, H, W]
            rgb_image: RGB image [B, 3, H, W]
            
        Returns:
            refined_depth: Refined depth map [B, 1, H, W]
        """
        batch_size, _, height, width = depth_map.shape
        
        # Ensure input dimensions match
        if rgb_image.shape[2:] != depth_map.shape[2:]:
            rgb_image = F.interpolate(rgb_image, size=(height, width), mode='bilinear', align_corners=False)
        
        # Extract texture features from RGB
        rgb_feat = self.activation(self.rgb_conv1(rgb_image))
        rgb_feat = self.activation(self.rgb_conv2(rgb_feat))
        
        # Extract features from depth map
        depth_feat = self.activation(self.depth_conv1(depth_map))
        depth_feat = self.activation(self.depth_conv2(depth_feat))
        
        # Detect edges in RGB image
        edge_x = self.edge_detect_x(rgb_image)
        edge_y = self.edge_detect_y(rgb_image)
        edge_magnitude = torch.sqrt(edge_x.pow(2) + edge_y.pow(2))
        
        # Detect edges in depth map
        depth_edge_x = F.conv2d(depth_map, self.edge_detect_x.weight, padding=1)
        depth_edge_y = F.conv2d(depth_map, self.edge_detect_y.weight, padding=1)
        depth_edge_magnitude = torch.sqrt(depth_edge_x.pow(2) + depth_edge_y.pow(2))
        
        # Combine RGB and depth features
        combined_feat = torch.cat([rgb_feat, depth_feat], dim=1)
        combined_feat = self.activation(self.combined_conv1(combined_feat))
        combined_feat = self.activation(self.combined_conv2(combined_feat))
        
        # Generate depth map refinement
        depth_residual = self.output_conv(combined_feat)
        
        # Edge-aware weighting: apply more refinement at edges
        edge_weight = torch.sigmoid(edge_magnitude * 5.0)  # Emphasize edges
        weighted_residual = depth_residual * edge_weight
        
        # Apply refinement to original depth map
        refined_depth = depth_map + weighted_residual
        
        return refined_depth

class MultiScaleDepthFusion(nn.Module):
    """
    Fuses depth predictions from multiple scales to improve accuracy.
    - Uses confidence weighting to prioritize reliable predictions
    - Preserves fine details while maintaining global structure
    - Outputs final depth map with reduced artifacts
    """
    def __init__(self, scales=[0, 1, 2, 3], num_channels=1):
        super(MultiScaleDepthFusion, self).__init__()
        self.scales = scales
        self.num_scales = len(scales)
        
        # Confidence estimator for each scale
        self.confidence_conv = nn.ModuleList([
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
            for _ in range(self.num_scales)
        ])
        
        # Final fusion convolution
        self.fusion_conv = nn.Conv2d(num_channels * self.num_scales, num_channels, kernel_size=1)
        self.activation = nn.Sigmoid()
        
    def forward(self, depth_scales):
        """
        Forward pass for fusing multiple depth scales
        
        Args:
            depth_scales: Dictionary of depth maps at different scales
                         keys are ("disp", scale_idx) or ("depth", scale_idx)
                         
        Returns:
            fused_depth: Fused depth map at highest resolution
        """
        # Determine target size from the highest resolution depth map
        target_scale = min(self.scales)
        target_key = ("disp", target_scale) if ("disp", target_scale) in depth_scales else ("depth", target_scale)
        target_size = depth_scales[target_key].shape[2:]
        
        # Prepare features for fusion
        scaled_features = []
        confidence_weights = []
        
        for i, scale in enumerate(self.scales):
            # Get feature map for this scale
            key = ("disp", scale) if ("disp", scale) in depth_scales else ("depth", scale)
            if key not in depth_scales:
                continue
                
            features = depth_scales[key]
            
            # Resize to target resolution if needed
            if features.shape[2:] != target_size:
                features = F.interpolate(features, size=target_size, mode='bilinear', align_corners=False)
            
            # Compute confidence for this scale
            confidence = self.activation(self.confidence_conv[i](features))
            
            # Store for fusion
            scaled_features.append(features)
            confidence_weights.append(confidence)
        
        # Stack features along channel dimension
        stacked_features = torch.cat(scaled_features, dim=1)
        stacked_confidence = torch.cat(confidence_weights, dim=1)
        
        # Normalize confidence weights across scales
        normalized_confidence = stacked_confidence / (torch.sum(stacked_confidence, dim=1, keepdim=True) + 1e-8)
        
        # Apply confidence weighting
        weighted_features = stacked_features * normalized_confidence
        
        # Final fusion
        fused_depth = self.fusion_conv(weighted_features)
        
        return fused_depth

class DepthQualityEnhancement:
    """
    A collection of post-processing methods to enhance depth map quality
    optimized for Mars terrain characteristics.
    """
    @staticmethod
    def bilateral_depth_filter(depth_map, rgb_image=None, sigma_space=15, sigma_color=0.05, sigma_depth=0.05):
        """
        Apply bilateral filtering to smooth depth while preserving edges
        
        Args:
            depth_map: Numpy depth map
            rgb_image: RGB image (optional)
            sigma_space: Spatial standard deviation
            sigma_color: Color standard deviation
            sigma_depth: Depth standard deviation
            
        Returns:
            filtered_depth: Filtered depth map
        """
        import cv2
        
        # Convert to single-channel float32
        depth_map = np.squeeze(depth_map).astype(np.float32)
        
        # Normalize depth to [0,1] for filtering
        depth_min = np.min(depth_map)
        depth_max = np.max(depth_map)
        depth_normalized = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
        
        # If RGB is provided, use joint bilateral filtering
        if rgb_image is not None:
            rgb = np.squeeze(rgb_image)
            if rgb.dtype != np.uint8:
                rgb = (rgb * 255).astype(np.uint8)
                
            # Apply joint bilateral filtering using RGB as guide
            filtered_depth = cv2.ximgproc.jointBilateralFilter(
                guide=rgb, 
                src=depth_normalized, 
                d=sigma_space, 
                sigmaColor=sigma_color, 
                sigmaSpace=sigma_space
            )
        else:
            # Apply regular bilateral filtering
            filtered_depth = cv2.bilateralFilter(
                depth_normalized, 
                d=sigma_space, 
                sigmaColor=sigma_depth, 
                sigmaSpace=sigma_space
            )
        
        # Rescale back to original range
        filtered_depth = filtered_depth * (depth_max - depth_min) + depth_min
        
        return filtered_depth
    
    @staticmethod
    def edge_preserving_smoothing(depth_map, rgb_image=None, lambda_param=0.1):
        """
        Apply edge-preserving smoothing (guided filter) to the depth map
        
        Args:
            depth_map: Numpy depth map
            rgb_image: RGB image (optional)
            lambda_param: Regularization parameter
            
        Returns:
            smoothed_depth: Smoothed depth map
        """
        import cv2
        
        # Convert to single-channel float32
        depth_map = np.squeeze(depth_map).astype(np.float32)
        
        # Normalize depth to [0,1] for filtering
        depth_min = np.min(depth_map)
        depth_max = np.max(depth_map)
        depth_normalized = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
        
        # Apply guided filter
        if rgb_image is not None:
            rgb = np.squeeze(rgb_image)
            if rgb.dtype != np.uint8:
                rgb = (rgb * 255).astype(np.uint8)
                
            # Convert RGB to grayscale if needed as guide
            if len(rgb.shape) == 3:
                guide = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            else:
                guide = rgb
                
            radius = min(depth_map.shape) // 16  # Adaptive radius
            smoothed_depth = cv2.ximgproc.guidedFilter(
                guide=guide,
                src=depth_normalized,
                radius=radius,
                eps=lambda_param
            )
        else:
            # Self-guided filtering
            radius = min(depth_map.shape) // 16
            smoothed_depth = cv2.ximgproc.guidedFilter(
                guide=depth_normalized,
                src=depth_normalized,
                radius=radius,
                eps=lambda_param
            )
        
        # Rescale back to original range
        smoothed_depth = smoothed_depth * (depth_max - depth_min) + depth_min
        
        return smoothed_depth
    
    @staticmethod
    def fill_depth_holes(depth_map, max_hole_size=10):
        """
        Fill holes (invalid or missing regions) in depth maps
        
        Args:
            depth_map: Numpy depth map
            max_hole_size: Maximum hole size to fill
            
        Returns:
            filled_depth: Depth map with holes filled
        """
        import cv2
        import scipy.ndimage as ndimage
        
        # Convert to single-channel float32
        depth_map = np.squeeze(depth_map).astype(np.float32)
        
        # Create a mask of invalid/missing values
        invalid_mask = ~np.isfinite(depth_map) | (depth_map <= 0)
        
        # Make a copy of the depth map
        filled_depth = depth_map.copy()
        
        # Label connected components of invalid regions
        labeled, num_features = ndimage.label(invalid_mask)
        
        # Process each connected component
        for i in range(1, num_features+1):
            # Get this component's mask
            component = labeled == i
            component_size = np.sum(component)
            
            # Skip large holes
            if component_size > max_hole_size * max_hole_size:
                continue
                
            # Get dilated mask to capture the nearby valid values
            dilated = ndimage.binary_dilation(component, iterations=3)
            neighbor_mask = dilated & ~component
            
            # Get valid neighbors
            if np.any(neighbor_mask):
                # Fill hole with the median of neighbors
                filled_depth[component] = np.median(depth_map[neighbor_mask])
        
        # Apply a gentle smoothing to the filled regions
        final_filled = filled_depth.copy()
        final_filled = cv2.medianBlur(final_filled, 3)
        
        # Only use the smoothed values where the original had holes
        filled_depth[invalid_mask] = final_filled[invalid_mask]
        
        return filled_depth
    
    @staticmethod
    def temporal_consistency_filter(depth_maps, weights=None):
        """
        Enhance depth quality using multiple frames for temporal consistency
        
        Args:
            depth_maps: List of sequential depth maps
            weights: Optional list of weights for each map
            
        Returns:
            filtered_depth: Temporally filtered depth map
        """
        if len(depth_maps) == 0:
            return None
            
        if len(depth_maps) == 1:
            return depth_maps[0]
            
        # Convert all depth maps to numpy
        depth_maps = [np.squeeze(d) for d in depth_maps]
        
        # Use equal weights if not provided
        if weights is None:
            weights = np.ones(len(depth_maps)) / len(depth_maps)
        else:
            weights = np.array(weights) / np.sum(weights)
            
        # Stack maps into a single array
        stacked = np.stack(depth_maps, axis=0)
        
        # Calculate weighted average
        filtered_depth = np.sum(stacked * weights[:, np.newaxis, np.newaxis], axis=0)
        
        return filtered_depth
