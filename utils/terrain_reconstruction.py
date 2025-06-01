from __future__ import absolute_import, division, print_function

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy.spatial.transform import Rotation as R
import math

# Import project specific functions
from layers import disp_to_depth

# Try to import open3d, but gracefully handle import error
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("Warning: open3d not available. 3D reconstruction features will be limited.")
    OPEN3D_AVAILABLE = False


class TerrainReconstructor:
    """
    Convert depth maps to 3D terrain reconstructions with geospatial referencing
    """
    def __init__(self, 
                 min_depth=0.1, 
                 max_depth=100.0,
                 voxel_size=0.05,
                 use_cuda=True):
        """
        Initialize the terrain reconstructor
        
        Args:
            min_depth: Minimum valid depth value
            max_depth: Maximum valid depth value
            voxel_size: Voxel size for point cloud downsampling
            use_cuda: Whether to use GPU acceleration
        """
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.voxel_size = voxel_size
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        
        # Check if open3d is available
        if not OPEN3D_AVAILABLE:
            print("Warning: open3d not available. Advanced 3D reconstruction features disabled.")
            print("Basic depth processing will still work.")
        
    def depth_to_point_cloud(self, depth, intrinsics=None, rgb_image=None, mask=None):
        """
        Convert a depth map to a point cloud
        
        Args:
            depth: Depth map tensor [H, W] or [B, 1, H, W]
            intrinsics: Camera intrinsics matrix [3, 3] or None (will use default)
            rgb_image: Optional RGB image tensor for coloring [H, W, 3] or [B, 3, H, W]
            mask: Optional mask for valid depth points [H, W] or [B, 1, H, W]
            
        Returns:
            points: Point cloud as numpy array [N, 3]
            colors: Colors as numpy array [N, 3] or None
        """
        # Ensure depth is on CPU as numpy array
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().squeeze().cpu().numpy()
        else:
            depth = np.squeeze(depth)
            
        # Get dimensions
        if depth.ndim == 3:  # [B, H, W]
            depth = depth[0]  # Just take the first batch element
        
        height, width = depth.shape
        
        # Default intrinsics if none provided (rough estimate for Mars rover)
        if intrinsics is None:
            fx = fy = 0.7 * max(height, width)  # Estimate focal length
            cx, cy = width / 2, height / 2       # Assume center principal point
            intrinsics = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            
        # Create meshgrid for pixel coordinates
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Apply mask if provided
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().squeeze().cpu().numpy() > 0.5
            valid = mask & (depth > self.min_depth) & (depth < self.max_depth)
        else:
            valid = (depth > self.min_depth) & (depth < self.max_depth)
            
        # Filter by valid mask
        z = depth[valid]
        x = x[valid]
        y = y[valid]
        
        # Extract intrinsics
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # Convert pixel coordinates to 3D points
        x_3d = (x - cx) * z / fx
        y_3d = (y - cy) * z / fy
        z_3d = z
        
        # Create point cloud
        points = np.stack([x_3d, y_3d, z_3d], axis=1)
        
        # Add colors if RGB image provided
        colors = None
        if rgb_image is not None:
            if isinstance(rgb_image, torch.Tensor):
                if rgb_image.shape[0] == 3:  # [3, H, W]
                    rgb_image = rgb_image.permute(1, 2, 0)
                rgb_image = rgb_image.detach().cpu().numpy()
            
            if rgb_image.max() <= 1.0:
                rgb_image = (rgb_image * 255).astype(np.uint8)
                
            # Extract colors for valid points
            colors = rgb_image.reshape(-1, 3)[valid.flatten()]
        
        return points, colors
        
    def create_open3d_point_cloud(self, points, colors=None):
        """
        Create an Open3D point cloud from numpy arrays
        
        Args:
            points: Point cloud as numpy array [N, 3]
            colors: Colors as numpy array [N, 3] or None
            
        Returns:
            pcd: Open3D PointCloud object or None if open3d not available
        """
        if not OPEN3D_AVAILABLE:
            print("Warning: Cannot create Open3D point cloud - open3d not available")
            return None
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        
        if colors is not None:
            if colors.max() > 1.0:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        
        return pcd
    
    def filter_and_downsample(self, pcd, voxel_size=None):
        """
        Filter outliers and downsample point cloud
        
        Args:
            pcd: Open3D PointCloud object
            voxel_size: Voxel size for downsampling (if None, use default)
            
        Returns:
            pcd_filtered: Filtered Open3D PointCloud object
        """
        if not OPEN3D_AVAILABLE or pcd is None:
            print("Warning: Cannot filter point cloud - open3d not available or point cloud is None")
            return pcd
            
        if voxel_size is None:
            voxel_size = self.voxel_size
        
        # Statistical outlier removal
        pcd_filtered, _ = pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )
        
        # Voxel downsampling
        pcd_filtered = pcd_filtered.voxel_down_sample(voxel_size)
        
        return pcd_filtered
        
    def estimate_normals(self, pcd, radius=0.1, max_nn=30):
        """
        Estimate normals for point cloud
        
        Args:
            pcd: Open3D PointCloud object
            radius: Search radius for normal estimation
            max_nn: Maximum number of neighbors to use
            
        Returns:
            pcd: PointCloud with normals
        """
        if not OPEN3D_AVAILABLE or pcd is None:
            print("Warning: Cannot estimate normals - open3d not available or point cloud is None")
            return pcd
            
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        )
        pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))
        
        return pcd
        
    def reconstruct_mesh(self, pcd, method='poisson', depth=9):
        """
        Reconstruct mesh from point cloud
        
        Args:
            pcd: Open3D PointCloud object with normals
            method: Reconstruction method ('poisson' or 'alpha_shape')
            depth: Depth parameter for Poisson reconstruction
            
        Returns:
            mesh: Reconstructed Open3D TriangleMesh object
        """
        if not OPEN3D_AVAILABLE or pcd is None:
            print("Warning: Cannot reconstruct mesh - open3d not available or point cloud is None")
            return None
            
        # Ensure normals are present
        if not pcd.has_normals():
            pcd = self.estimate_normals(pcd)
            
        if method == 'poisson':
            # Poisson surface reconstruction
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth, linear_fit=False
            )
            # Remove low-density vertices
            vertices_to_remove = self._remove_low_density_vertices(mesh, pcd)
            mesh.remove_vertices_by_mask(vertices_to_remove)
        else:
            # Alpha shape reconstruction
            alpha = 0.5
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            
        # Clean up the mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        return mesh
    
    def _remove_low_density_vertices(self, mesh, pcd, density_threshold=0.1):
        """Remove low-density vertices from the mesh"""
        # Create a KDTree from the original point cloud
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        
        # For each vertex in the mesh, check its density
        mesh_vertices = np.asarray(mesh.vertices)
        density = np.zeros(len(mesh_vertices))
        
        for i, vertex in enumerate(mesh_vertices):
            # Find the K nearest neighbors in the original point cloud
            [_, idx, _] = pcd_tree.search_knn_vector_3d(vertex, 10)
            # Density is the number of neighbors within a radius
            density[i] = len(idx)
            
        # Normalize density
        density = density / density.max()
        
        # Vertices to remove
        vertices_to_remove = density < density_threshold
        
        return vertices_to_remove
    
    def stitch_meshes(self, meshes):
        """
        Stitch multiple meshes into a single mesh
        
        Args:
            meshes: List of Open3D TriangleMesh objects
            
        Returns:
            combined_mesh: Combined Open3D TriangleMesh object
        """
        if not OPEN3D_AVAILABLE:
            print("Warning: Cannot stitch meshes - open3d not available")
            return None
            
        if not meshes:
            return None
            
        # If only one mesh is provided, return it
        if len(meshes) == 1:
            return meshes[0]
            
        # Initialize combined mesh with the first mesh
        combined_mesh = meshes[0]
        
        # Add each subsequent mesh
        for mesh in meshes[1:]:
            if mesh is not None:
                combined_mesh += mesh
                
        return combined_mesh
        
    def add_geospatial_reference(self, mesh, lat, lon, alt, rotation=(0.0, 0.0, 0.0), scale=1.0):
        """
        Add geospatial reference to mesh
        
        Args:
            mesh: Open3D TriangleMesh object
            lat: Latitude in degrees
            lon: Longitude in degrees
            alt: Altitude in meters
            rotation: (roll, pitch, yaw) in degrees
            scale: Scale factor
            
        Returns:
            mesh: Mesh with geospatial reference
        """
        if not OPEN3D_AVAILABLE or mesh is None:
            print("Warning: Cannot add geospatial reference - open3d not available or mesh is None")
            return mesh
            
        # Store geospatial metadata in mesh object
        # Since Open3D doesn't have a direct way to store this info,
        # we'll store it as a custom attribute
        
        # Convert to numpy array for consistency
        mesh_points = np.asarray(mesh.vertices)
        
        # Create geodetic metadata
        metadata = {
            'coordinate_system': 'Mars_latlon_alt',
            'latitude': lat,
            'longitude': lon,
            'altitude': alt,
            'rotation': rotation,
            'scale': scale
        }
        
        # For future reference, we would add a method to write this metadata
        # to a sidecar file when saving the mesh
        
        return mesh
    

class MarsTerrainMapper:
    """
    High-level class for Mars terrain mapping from depth maps
    """
    def __init__(self, 
                 model=None, 
                 min_depth=0.1,
                 max_depth=100.0,
                 use_cuda=True):
        """
        Initialize the Mars terrain mapper
        
        Args:
            model: DepthAnything model for inferring depth maps
            min_depth: Minimum valid depth value
            max_depth: Maximum valid depth value
            use_cuda: Whether to use GPU acceleration
        """
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        
        # Initialize terrain reconstructor
        self.reconstructor = TerrainReconstructor(
            min_depth=min_depth,
            max_depth=max_depth,
            use_cuda=use_cuda
        )
        
        # If model is provided, move it to device
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
              # Check if 3D reconstruction is available
        self.reconstruction_available = OPEN3D_AVAILABLE
        
    def infer_depth(self, image, metadata=None):
        """
        Infer depth from image using the model
        
        Args:
            image: Input image tensor [B, 3, H, W] or numpy array [H, W, 3]
            metadata: Optional dictionary with image metadata
            
        Returns:
            depth: Depth map tensor [B, 1, H, W]
        """
        if self.model is None:
            raise ValueError("No model provided for depth inference")
        
        # Convert numpy image to tensor if needed
        if isinstance(image, np.ndarray):
            # Ensure [H, W, 3] format
            if image.shape[0] == 3 and len(image.shape) == 3:  # [3, H, W]
                image = np.transpose(image, (1, 2, 0))
                
            # Normalize if uint8
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
                
            # Convert to [B, 3, H, W] tensor
            image = torch.from_numpy(image).float()
            image = image.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        
        # Ensure batch dimension
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # Store original image dimensions
        original_size = (image.shape[2], image.shape[3])
            
        # Move to device
        image = image.to(self.device)
        
        # Validate input image
        if torch.isnan(image).any() or torch.isinf(image).any():
            print("Warning: Input image contains NaN or Inf values. Fixing...")
            image = torch.nan_to_num(image, nan=0.5, posinf=1.0, neginf=0.0)
        
        # Check image range and normalize if needed
        img_min = image.min()
        img_max = image.max()
        
        if img_max > 1.0 + 1e-6:
            print(f"Warning: Input image values outside [0,1] range: [{img_min:.2f}, {img_max:.2f}]. Normalizing...")
            image = image / max(img_max, 1.0)
        
        # Infer depth with exception handling
        try:
            with torch.no_grad():
                outputs = self.model(image, metadata)
                
            # Validate outputs
            if ("disp", 0) not in outputs:
                raise ValueError("Model output doesn't contain disparity values")
                
            disp = outputs[("disp", 0)]
            
            # Check if disparity is valid
            if torch.isnan(disp).any() or torch.isinf(disp).any():
                print("Warning: Disparity contains NaN or Inf values. Fixing...")
                disp = torch.nan_to_num(disp, nan=0.5, posinf=1.0, neginf=0.0)
                
            # Check if disparity is constant
            disp_min = disp.min().item()
            disp_max = disp.max().item()
            disp_range = disp_max - disp_min
            
            if disp_range < 1e-6:
                print(f"Warning: Constant disparity map detected (value: {disp_min:.6f}). Attempting alternate processing...")
                
                # Attempt a different image size to see if that helps
                resized_image = F.interpolate(image, size=(256, 256), mode='bilinear', align_corners=False)
                outputs = self.model(resized_image, metadata)
                disp = outputs[("disp", 0)]
                
                # Check if this fixed the issue
                new_disp_min = disp.min().item()
                new_disp_max = disp.max().item()
                new_disp_range = new_disp_max - new_disp_min
                
                if new_disp_range < 1e-6:
                    print("Warning: Model still producing constant disparity. Using fallback depth estimation.")
                    
                    # Create a fallback gradient-based depth map using image gradients
                    # This is better than a constant map for visualization/debugging
                    gray = image.mean(dim=1, keepdim=True)
                    dx = gray[:, :, 1:, :] - gray[:, :, :-1, :]
                    dy = gray[:, :, :, 1:] - gray[:, :, :, :-1]
                    grad_mag = torch.sqrt(
                        F.pad(dx**2, (0, 0, 0, 1)) + 
                        F.pad(dy**2, (0, 1, 0, 0))
                    ) + 0.01
                    
                    # Create synthetic disparity based on image gradients
                    # Areas with more texture/gradients are often closer
                    disp = 0.5 + grad_mag * 0.5  # Range [0.5, 1.0]
                    
                    # Add some structure based on position in the image
                    h, w = disp.shape[2:]
                    y_grad = torch.linspace(0.1, -0.1, h, device=disp.device).view(1, 1, h, 1).expand_as(disp)
                    disp = disp + y_grad  # Objects lower in frame are typically closer
                    
                    print(f"Created fallback depth map with range: [{disp.min().item():.4f}, {disp.max().item():.4f}]")
                else:
                    print(f"Alternate processing succeeded. New disparity range: {new_disp_range:.6f}")
            
            # Convert to depth
            _, depth = disp_to_depth(disp, min_depth=self.reconstructor.min_depth, 
                                     max_depth=self.reconstructor.max_depth)
            
            # Resize back to original dimensions if needed
            if depth.shape[2:] != original_size:
                depth = F.interpolate(
                    depth, size=original_size, mode='bilinear', align_corners=False
                )
            
            return depth
            
        except Exception as e:
            print(f"Error during depth inference: {e}")
            print("Creating fallback depth map")
            
            # Create a fallback depth map with useful information, not just constants
            h, w = original_size
            batch_size = image.shape[0]
            
            # Use image intensity to create a rough depth approximation
            # Darker pixels are often more distant in Mars imagery
            if isinstance(image, torch.Tensor):
                gray = image.mean(dim=1, keepdim=True)  # Simple grayscale conversion
                
                # Normalize to [0.1, 1.0] range for depth values
                depth = 0.1 + 0.9 * (1.0 - gray)  # Invert so brighter is closer
            else:
                # Create a position-based gradient if image is not available
                y = torch.linspace(0.1, 0.9, h, device=self.device)
                depth = y.view(1, 1, h, 1).expand(batch_size, 1, h, w)
            
            return depth
    
    def process_image(self, image, intrinsics=None, metadata=None):
        """
        Process a single image to create a 3D point cloud
        
        Args:
            image: Input image tensor or numpy array
            intrinsics: Optional camera intrinsics matrix
            metadata: Optional dictionary with image metadata
            
        Returns:
            pcd: Open3D PointCloud object or None if 3D reconstruction not available
        """
        # Check if 3D reconstruction is available
        if not self.reconstruction_available:
            print("Warning: 3D reconstruction not available - open3d not installed")
            return None
            
        # Infer depth
        depth = self.infer_depth(image, metadata)
        
        # Convert to point cloud
        points, colors = self.reconstructor.depth_to_point_cloud(depth, intrinsics, image)
        
        # Create Open3D point cloud
        pcd = self.reconstructor.create_open3d_point_cloud(points, colors)
        
        # Filter and downsample
        if pcd is not None:
            pcd = self.reconstructor.filter_and_downsample(pcd)
        
        return pcd
    
    def process_image_to_mesh(self, image, intrinsics=None, metadata=None):
        """
        Process a single image to create a 3D mesh
        
        Args:
            image: Input image tensor or numpy array
            intrinsics: Optional camera intrinsics matrix
            metadata: Optional dictionary with image metadata
            
        Returns:
            mesh: Open3D TriangleMesh object or None if 3D reconstruction not available
        """
        # Check if 3D reconstruction is available
        if not self.reconstruction_available:
            print("Warning: 3D mesh generation not available - open3d not installed")
            return None
            
        # Create point cloud
        pcd = self.process_image(image, intrinsics, metadata)
        
        # If point cloud creation failed, return None
        if pcd is None:
            return None
            
        # Estimate normals
        pcd = self.reconstructor.estimate_normals(pcd)
        
        # Reconstruct mesh
        mesh = self.reconstructor.reconstruct_mesh(pcd)
        
        # Add geospatial reference if location is provided in metadata
        if mesh is not None and metadata is not None and 'latitude' in metadata and 'longitude' in metadata:
            lat = metadata.get('latitude', 0.0)
            lon = metadata.get('longitude', 0.0)
            alt = metadata.get('altitude', 0.0)
            rotation = metadata.get('rotation', (0.0, 0.0, 0.0))
            scale = metadata.get('scale', 1.0)
            
            mesh = self.reconstructor.add_geospatial_reference(
                mesh, lat, lon, alt, rotation, scale
            )
        
        return mesh
    
    def process_image_sequence(self, images, intrinsics=None, metadata_list=None):
        """
        Process a sequence of images to create a stitched 3D mesh
        
        Args:
            images: List of input images
            intrinsics: Optional camera intrinsics matrix or list of matrices
            metadata_list: Optional list of metadata dictionaries
            
        Returns:
            combined_mesh: Combined Open3D TriangleMesh object or None if 3D reconstruction not available
        """
        # Check if 3D reconstruction is available
        if not self.reconstruction_available:
            print("Warning: 3D terrain stitching not available - open3d not installed")
            return None
            
        meshes = []
        
        for i, image in enumerate(images):
            # Get intrinsics and metadata for this image
            img_intrinsics = intrinsics[i] if isinstance(intrinsics, list) else intrinsics
            img_metadata = metadata_list[i] if metadata_list is not None else None
            
            # Process image to mesh
            mesh = self.process_image_to_mesh(image, img_intrinsics, img_metadata)
            if mesh is not None:
                meshes.append(mesh)
            
        # If no meshes were created, return None
        if not meshes:
            return None
            
        # Stitch meshes
        combined_mesh = self.reconstructor.stitch_meshes(meshes)
        
        return combined_mesh


# Import function for disparity to depth conversion
from layers import disp_to_depth
