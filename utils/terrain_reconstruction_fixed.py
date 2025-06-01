import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from layers import disp_to_depth

class TerrainReconstructor:
    """
    Base class for terrain reconstruction from depth maps
    """
    def __init__(self, min_depth=0.1, max_depth=100.0):
        self.min_depth = min_depth
        self.max_depth = max_depth
        
    def create_point_cloud(self, depth_map, intrinsics=None):
        """
        Create a 3D point cloud from a depth map
        
        Args:
            depth_map: Depth map tensor of shape [H, W] or [1, H, W]
            intrinsics: Camera intrinsic parameters as 3x3 matrix
            
        Returns:
            points3D: 3D points of shape [N, 3]
        """
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.detach().cpu().numpy()
            
        # Ensure depth_map is 2D
        if depth_map.ndim == 3:
            depth_map = depth_map.squeeze(0)
            
        h, w = depth_map.shape
        
        # Default intrinsics if none provided
        if intrinsics is None:
            # Use reasonable defaults for Mars rover cameras
            f = 0.7 * w  # focal length
            cx, cy = w / 2, h / 2  # principal point
            intrinsics = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0, 1]
            ])
        
        # Create coordinate grid
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        u = u.reshape(-1).astype(np.float32)
        v = v.reshape(-1).astype(np.float32)
        z = depth_map.reshape(-1).astype(np.float32)
        
        # Remove points with invalid depth
        valid = (z > self.min_depth) & (z < self.max_depth)
        u = u[valid]
        v = v[valid]
        z = z[valid]
        
        # Convert image coordinates to normalized camera coordinates
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Create point cloud
        points3D = np.stack((x, y, z), axis=1)
        
        return points3D
    
    def reconstruct_terrain(self, depth_map, rgb_image=None, mask=None, intrinsics=None):
        """
        Reconstruct terrain as a colored point cloud or mesh
        
        Args:
            depth_map: Depth map tensor [H, W] or [1, H, W]
            rgb_image: Optional RGB image [H, W, 3]
            mask: Optional binary mask for valid terrain regions [H, W]
            intrinsics: Camera intrinsic parameters
            
        Returns:
            points3D: 3D points
            colors: Point colors if rgb_image is provided
        """
        points3D = self.create_point_cloud(depth_map, intrinsics)
        
        if rgb_image is not None:
            if isinstance(rgb_image, torch.Tensor):
                rgb_image = rgb_image.detach().cpu().numpy()
                
            # Ensure rgb_image is [H, W, 3]
            if rgb_image.shape[0] == 3:
                rgb_image = np.moveaxis(rgb_image, 0, -1)
                
            h, w = depth_map.shape[-2:] if depth_map.ndim > 2 else depth_map.shape
            
            # Ensure rgb_image has the right dimensions
            if rgb_image.shape[:2] != (h, w):
                rgb_image = cv2.resize(rgb_image, (w, h))
                
            # Extract colors for the valid points
            depth_map_flat = depth_map.reshape(-1) if depth_map.ndim <= 2 else depth_map.squeeze(0).reshape(-1)
            valid = (depth_map_flat > self.min_depth) & (depth_map_flat < self.max_depth)
            
            colors = rgb_image.reshape(-1, 3)[valid]
            
            return points3D, colors
        else:
            return points3D, None

class MarsTerrainMapper:
    """
    Mars terrain depth mapping and reconstruction
    with robust handling of different image sources
    """
    def __init__(self, model, min_depth=0.1, max_depth=100.0, use_cuda=True):
        """
        Initialize the Mars terrain mapper
        
        Args:
            model: Depth prediction model
            min_depth: Minimum depth value
            max_depth: Maximum depth value
            use_cuda: Whether to use GPU for inference
        """
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.model.to(self.device)
        self.reconstructor = TerrainReconstructor(min_depth=min_depth, max_depth=max_depth)
        
    def infer_depth(self, image, metadata=None):
        """
        Infer depth from an image
        
        Args:
            image: Image tensor or numpy array [B, 3, H, W]
            metadata: Dictionary with metadata about the image source
            
        Returns:
            depth: Depth map tensor [B, 1, H, W]
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Ensure image is on the right device
        if isinstance(image, torch.Tensor):
            image = image.to(self.device)
        else:
            # Convert numpy array to tensor
            image = torch.from_numpy(image).float().to(self.device)
            
            # Add batch dimension if needed
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            # Normalize if needed
            if image.max() > 1.0:
                image = image / 255.0
                
        # Store original size for resizing back
        original_size = image.shape[2:]
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image, metadata)
            
            # Get disparity and convert to depth
            disp = outputs[("disp", 0)]
            _, depth = disp_to_depth(disp, min_depth=self.reconstructor.min_depth, 
                                     max_depth=self.reconstructor.max_depth)
            
            # Resize back to original dimensions if needed
            if depth.shape[2:] != original_size:
                depth = F.interpolate(
                    depth, size=original_size, mode='bilinear', align_corners=False
                )
                
        return depth
    
    def process_image(self, image_path, output_path=None, metadata=None):
        """
        Process a single image to create depth map and point cloud
        
        Args:
            image_path: Path to image file
            output_path: Path to save visualization (optional)
            metadata: Dictionary with metadata about the image
            
        Returns:
            depth: Depth map tensor
            points3D: 3D points from depth map
            colors: Point colors from image
        """
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Set default metadata if none provided
        if metadata is None:
            # Try to infer source from path
            if 'rover' in image_path.lower():
                source = 'rover'
            elif 'ingenuity' in image_path.lower():
                source = 'ingenuity'
            elif 'satellite' in image_path.lower():
                source = 'satellite'
            else:
                source = 'unknown'
                
            metadata = {'source': source, 'path': image_path}
        
        # Get depth map
        depth = self.infer_depth(image_tensor, metadata)
        
        # Create point cloud
        points3D, colors = self.reconstructor.reconstruct_terrain(
            depth.squeeze().cpu().numpy(), image
        )
        
        # Visualize if output path is provided
        if output_path:
            plt.figure(figsize=(15, 10))
            
            # Show original image
            plt.subplot(2, 2, 1)
            plt.imshow(image)
            plt.title('Original Image')
            plt.axis('off')
            
            # Show depth map
            plt.subplot(2, 2, 2)
            plt.imshow(depth.squeeze().cpu().numpy(), cmap='plasma')
            plt.colorbar(label='Depth')
            plt.title('Depth Map')
            plt.axis('off')
            
            # Show point cloud if available
            if len(points3D) > 0:
                plt.subplot(2, 2, 3, projection='3d')
                ax = plt.gca()
                
                # Downsample points for visualization
                if len(points3D) > 10000:
                    idx = np.random.choice(len(points3D), 10000, replace=False)
                    pts = points3D[idx]
                    cols = colors[idx] if colors is not None else None
                else:
                    pts = points3D
                    cols = colors
                
                # Plot points
                if cols is not None:
                    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols/255, s=1, alpha=0.5)
                else:
                    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], cmap='viridis', s=1, alpha=0.5)
                
                ax.set_title('3D Point Cloud')
                
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
        return depth, points3D, colors
    
    def process_batch(self, image_paths, output_dir=None, metadata_list=None):
        """
        Process a batch of images
        
        Args:
            image_paths: List of image paths
            output_dir: Directory to save visualizations
            metadata_list: List of metadata dictionaries
            
        Returns:
            depths: List of depth maps
            point_clouds: List of point clouds
        """
        depths = []
        point_clouds = []
        
        for i, image_path in enumerate(image_paths):
            metadata = metadata_list[i] if metadata_list else None
            
            if output_dir:
                # Create output filename
                image_name = os.path.basename(image_path)
                output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_depth.png")
            else:
                output_path = None
                
            # Process image
            depth, points3D, _ = self.process_image(image_path, output_path, metadata)
            
            depths.append(depth)
            point_clouds.append(points3D)
            
        return depths, point_clouds
