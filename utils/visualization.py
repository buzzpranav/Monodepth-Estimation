from __future__ import absolute_import, division, print_function

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import cv2
import PIL.Image as pil
from tqdm import tqdm

# Try to import open3d, but gracefully handle import error
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("Warning: open3d not available. 3D visualization features will be limited.")
    OPEN3D_AVAILABLE = False


class MarsTerrainVisualizer:
    """
    Visualization tools for Mars terrain maps with depth colorization
    """
    def __init__(self, min_depth=0.1, max_depth=100.0, mars_themed=True):
        """
        Initialize the Mars terrain visualizer
        
        Args:
            min_depth: Minimum depth value for colorization
            max_depth: Maximum depth value for colorization
            mars_themed: Whether to use Mars-themed colormaps
        """
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.mars_themed = mars_themed
        
        # Define Mars-themed colormaps
        self._setup_mars_colormaps()
        
    def _setup_mars_colormaps(self):
        """Define Mars-themed colormaps"""
        # Mars surface colormap - reddish-brown to light tan
        mars_colors = [
            (0.42, 0.2, 0.04),   # Dark reddish-brown
            (0.55, 0.3, 0.15),   # Medium reddish-brown
            (0.67, 0.42, 0.27),  # Light reddish-brown
            (0.78, 0.57, 0.39),  # Tan
            (0.88, 0.71, 0.53)   # Light tan
        ]
        self.mars_cmap = LinearSegmentedColormap.from_list("mars_terrain", mars_colors)
        
        # Mars elevation colormap - deep canyons (blue) to high mountains (white)
        mars_elev_colors = [
            (0.01, 0.01, 0.35),  # Deep blue (Valles Marineris-like canyons)
            (0.3, 0.3, 0.5),     # Slate blue (low terrain)
            (0.6, 0.22, 0.12),   # Reddish-brown (average terrain)
            (0.8, 0.5, 0.3),     # Light brown (highlands)
            (1.0, 0.95, 0.9)     # Near-white (Olympus Mons-like peaks)
        ]
        self.mars_elev_cmap = LinearSegmentedColormap.from_list("mars_elevation", mars_elev_colors)
        
    def colorize_depth(self, depth, cmap_name=None, percentile=95):
        """
        Colorize depth map
        
        Args:
            depth: Depth map as tensor or numpy array
            cmap_name: Name of colormap to use
            percentile: Percentile for max depth normalization
            
        Returns:
            colored_depth: RGB image with colormapped depth
        """
        # Convert to numpy if tensor
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().squeeze().cpu().numpy()
        
        # Ensure depth is 2D
        depth = np.squeeze(depth)
        
        # Determine colormap
        if cmap_name is None:
            if self.mars_themed:
                cmap = self.mars_elev_cmap
            else:
                cmap = 'magma'
        else:
            cmap = cmap_name
            
        # Normalize depth values
        vmin = depth.min()
        # Use percentile to avoid outliers affecting the colormap
        vmax = np.percentile(depth, percentile)
        
        # Apply colormap
        normalizer = plt.Normalize(vmin=vmin, vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
        colored_depth = mapper.to_rgba(depth)[:, :, :3]
        
        # Convert to uint8 image
        colored_depth = (colored_depth * 255).astype(np.uint8)
        
        return colored_depth
    
    def visualize_depth_map(self, image, depth, save_path=None, show=True, figsize=(12, 5)):
        """
        Visualize input image and its corresponding depth map
        
        Args:
            image: Input image tensor or numpy array
            depth: Depth map tensor or numpy array
            save_path: Path to save the visualization image
            show: Whether to display the visualization
            figsize: Figure size for the plot
            
        Returns:
            fig: Matplotlib figure
        """
        # Convert to numpy if tensors
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()
            
        # Ensure proper dimensions
        image = np.squeeze(image)
        if image.shape[0] == 3 and len(image.shape) == 3:  # [3, H, W]
            image = np.transpose(image, (1, 2, 0))
        depth = np.squeeze(depth)
            
        # Normalize image if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
            
        # Colorize depth
        colored_depth = self.colorize_depth(depth)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot input image
        axes[0].imshow(image)
        axes[0].set_title("Input Image")
        axes[0].axis('off')
        
        # Plot depth map
        im = axes[1].imshow(depth, cmap=self.mars_elev_cmap if self.mars_themed else 'magma')
        axes[1].set_title("Depth Map")
        axes[1].axis('off')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=axes[1], shrink=0.7)
        cbar.set_label('Depth')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            
        # Show or close
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def visualize_3d_terrain(self, points, colors=None, save_path=None):
        """
        Visualize 3D point cloud of terrain
        
        Args:
            points: Point cloud as numpy array [N, 3]
            colors: Optional colors for points [N, 3]
            save_path: Optional path to save the visualization
            
        Returns:
            pcd: Open3D point cloud object or None if open3d is not available
        """
        if not OPEN3D_AVAILABLE:
            print("Open3D not available. Cannot visualize 3D terrain.")
            # Fall back to matplotlib 3D plot if available
            try:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                if colors is not None:
                    if colors.max() > 1.0:
                        colors = colors / 255.0
                    
                    # Matplotlib needs RGB tuples for each point
                    color_tuples = colors.reshape(-1, 3)
                else:
                    # Color by height (Z coordinate)
                    z_values = points[:, 2]
                    norm = plt.Normalize(z_values.min(), z_values.max())
                    cmap = self.mars_elev_cmap if self.mars_themed else plt.cm.viridis
                    color_tuples = cmap(norm(z_values))[:, :3]
                
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c=color_tuples, s=1)
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title('Mars Terrain 3D Visualization')
                
                if save_path is not None:
                    plt.savefig(save_path.replace('.ply', '.png'))
                    
                plt.show()
                return None
                
            except Exception as e:
                print(f"Could not create fallback 3D plot: {e}")
                return None
                
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        
        if colors is not None:
            if colors.max() > 1.0:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        else:
            # Color by height (Z coordinate)
            z_values = np.asarray(points)[:, 2]
            colors = self.colorize_depth(z_values.reshape(-1, 1))
            
            # Normalize to [0, 1]
            colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
        # Optional: Estimate normals for better visualization
        pcd.estimate_normals()
        
        # Save if path provided
        if save_path is not None:
            o3d.io.write_point_cloud(save_path, pcd)
        
        # Visualize
        o3d.visualization.draw_geometries([pcd])
        return pcd
    
    def visualize_point_cloud(self, depth, intrinsics=None, save_path=None):
        """
        Visualize depth map as point cloud
        
        Args:
            depth: Depth map (HxW)
            intrinsics: Optional camera intrinsics matrix
            save_path: Optional path to save the visualization
            
        Returns:
            pcd: Open3D point cloud object or None if open3d is not available
        """
        if not OPEN3D_AVAILABLE:
            print("Open3D not available. Cannot visualize point cloud.")
            # Create a simple matplotlib 3D point cloud visualization as fallback
            try:
                # Convert depth to point cloud using simple projection
                h, w = depth.shape
                
                # Create x,y grid
                y, x = np.mgrid[0:h, 0:w]
                
                # If intrinsics not provided, use default
                if intrinsics is None:
                    # Default for Mars cameras (approximation)
                    fx, fy = 350, 350  # focal length
                    cx, cy = w // 2, h // 2  # center point
                else:
                    fx = intrinsics[0, 0]
                    fy = intrinsics[1, 1]
                    cx = intrinsics[0, 2]
                    cy = intrinsics[1, 2]
                
                # Convert to normalized coordinates
                x_norm = (x - cx) / fx
                y_norm = (y - cy) / fy
                
                # Generate 3D points (subsample for speed)
                step = 10  # Subsample by taking every nth point
                points = np.zeros((h//step * w//step, 3))
                idx = 0
                for i in range(0, h, step):
                    for j in range(0, w, step):
                        z = depth[i, j]
                        points[idx] = [x_norm[i, j] * z, y_norm[i, j] * z, z]
                        idx += 1
                
                # Create matplotlib 3D plot
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Color points by depth
                norm = plt.Normalize(depth.min(), depth.max())
                cmap = self.mars_elev_cmap if self.mars_themed else plt.cm.viridis
                
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c=points[:, 2], cmap=cmap, s=1)
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title('Depth Point Cloud')
                
                if save_path is not None:
                    plt.savefig(save_path.replace('.ply', '.png'))
                    
                plt.show()
                return None
                
            except Exception as e:
                print(f"Could not create fallback depth visualization: {e}")
                return None
        
        # If intrinsics not provided, use default
        if intrinsics is None:
            # Default for Mars cameras (approximation)
            fx, fy = 350, 350  # focal length
            cx, cy = depth.shape[1] // 2, depth.shape[0] // 2  # center point
            
            intrinsics = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
        
        # Create depth image for Open3D
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
        
        # Create intrinsic matrix for Open3D
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.intrinsic_matrix = intrinsics
        
        # Create point cloud from depth image
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d, 
            intrinsic
        )
        
        # Optional: Estimate normals for better visualization
        pcd.estimate_normals()
        
        # Save if path provided
        if save_path is not None:
            o3d.io.write_point_cloud(save_path, pcd)
        
        # Visualize
        o3d.visualization.draw_geometries([pcd])
        return pcd
        
    def visualize_mesh(self, mesh, save_path=None):
        """
        Visualize 3D mesh of terrain
        
        Args:
            mesh: Open3D TriangleMesh object
            save_path: Optional path to save the visualization
            
        Returns:
            mesh: The input mesh or None if visualization failed
        """
        if not OPEN3D_AVAILABLE:
            print("Open3D not available. Cannot visualize mesh.")
            return None
            
        # Save if path provided
        if save_path is not None:
            o3d.io.write_triangle_mesh(save_path, mesh)
            
        # Visualize
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
        return mesh
    
    def create_depth_video(self, image_list, depth_list, output_path, fps=30, colormap=None):
        """
        Create a video of depth maps
        
        Args:
            image_list: List of input images
            depth_list: List of depth maps
            output_path: Path to save the output video
            fps: Frames per second
            colormap: Optional colormap name
            
        Returns:
            output_path: Path to the saved video
        """
        if len(image_list) != len(depth_list):
            raise ValueError("Number of images and depth maps must match")
            
        if len(image_list) == 0:
            raise ValueError("Empty lists provided")
            
        # Get dimensions from first image
        if isinstance(image_list[0], torch.Tensor):
            height, width = image_list[0].shape[-2], image_list[0].shape[-1]
        else:
            height, width = image_list[0].shape[:2]
            
        # Double width to show image and depth side by side
        out_width = width * 2
        out_height = height
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
        
        # Process each frame
        for i in tqdm(range(len(image_list)), desc="Creating depth video"):
            # Get image and depth
            image = image_list[i]
            depth = depth_list[i]
            
            # Convert to numpy if tensors
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()
            if isinstance(depth, torch.Tensor):
                depth = depth.detach().cpu().numpy()
                
            # Ensure proper dimensions
            image = np.squeeze(image)
            if image.shape[0] == 3 and len(image.shape) == 3:  # [3, H, W]
                image = np.transpose(image, (1, 2, 0))
            depth = np.squeeze(depth)
                
            # Normalize image if needed
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
                
            # Colorize depth
            colored_depth = self.colorize_depth(depth, cmap_name=colormap)
            
            # Ensure image is RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)
            
            # Resize if necessary
            if image.shape[0] != height or image.shape[1] != width:
                image = cv2.resize(image, (width, height))
            if colored_depth.shape[0] != height or colored_depth.shape[1] != width:
                colored_depth = cv2.resize(colored_depth, (width, height))
                
            # Combine image and depth side by side
            combined = np.hstack([image, colored_depth])
            
            # Write to video
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            video_out.write(combined_bgr)
            
        # Release video writer
        video_out.release()
        
        return output_path
        
    def create_flythrough_video(self, mesh, trajectory, output_path, fps=30, resolution=(1280, 720)):
        """
        Create a flythrough video of 3D terrain
        
        Args:
            mesh: Open3D TriangleMesh object
            trajectory: List of camera poses (4x4 matrices)
            output_path: Path to save the output video
            fps: Frames per second
            resolution: Video resolution (width, height)
            
        Returns:
            output_path: Path to the saved video or None if open3d is not available
        """
        if not OPEN3D_AVAILABLE:
            print("Open3D not available. Cannot create flythrough video.")
            return None
            
        width, height = resolution
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Set up Open3D visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height, visible=False)
        vis.add_geometry(mesh)
        
        # Add default lighting
        vis.get_render_option().load_from_json({
            "light_on": True,
            "background_color": [0.0, 0.0, 0.0],
            "point_size": 1.0,
            "light_ambient_intensity": 0.3,
            "light_diffuse_intensity": 0.8,
            "light_specular_intensity": 0.5,
            "mesh_shade_option": 2
        })
        
        # Get view control
        ctr = vis.get_view_control()
        
        # Render each frame along trajectory
        for i, pose in enumerate(tqdm(trajectory, desc="Creating flythrough video")):
            # Set camera pose
            param = ctr.convert_to_pinhole_camera_parameters()
            extrinsic = np.linalg.inv(pose)  # Camera extrinsic is inverse of pose
            param.extrinsic = extrinsic
            ctr.convert_from_pinhole_camera_parameters(param)
            
            # Update and render
            vis.update_geometry(mesh)
            vis.poll_events()
            vis.update_renderer()
            
            # Capture image
            img = np.asarray(vis.capture_screen_float_buffer())
            
            # Convert to uint8
            img = (img * 255).astype(np.uint8)
            
            # Convert to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Write to video
            video_out.write(img_bgr)
            
        # Clean up
        vis.destroy_window()
        video_out.release()
        
        return output_path
        
    def generate_flythrough_trajectory(self, mesh, n_frames=300):
        """
        Generate a camera trajectory for a flythrough video
        
        Args:
            mesh: Open3D TriangleMesh object
            n_frames: Number of frames in the trajectory
            
        Returns:
            trajectory: List of 4x4 camera pose matrices or None if open3d is not available
        """
        if not OPEN3D_AVAILABLE:
            print("Open3D not available. Cannot generate flythrough trajectory.")
            return None
            
        # Get mesh bounds
        bbox = mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_extent()
        
        # Calculate trajectory radius
        radius = np.linalg.norm(extent) * 1.5
        
        # Generate circular trajectory around mesh
        trajectory = []
        
        for i in range(n_frames):
            # Angle (full circle plus some extra)
            angle = i * 2.0 * np.pi / (n_frames - 1) * 1.2
            
            # Position on circle plus some height variation
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            # Height varies in a sinusoidal pattern
            z = center[2] + extent[2] * (0.5 + 0.3 * np.sin(angle * 2))
            
            # Create a camera pose matrix (looking at center)
            pos = np.array([x, y, z])
            look_at = center
            up = np.array([0, 0, 1])  # Z-up
            
            # Create rotation matrix to look at center
            z_axis = look_at - pos
            z_axis = z_axis / np.linalg.norm(z_axis)
            
            x_axis = np.cross(up, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
            
            y_axis = np.cross(z_axis, x_axis)
            
            # Create pose matrix
            pose = np.eye(4)
            pose[:3, 0] = x_axis
            pose[:3, 1] = y_axis
            pose[:3, 2] = z_axis
            pose[:3, 3] = pos
            
            trajectory.append(pose)
        
        return trajectory
        
    def create_depth_comparison_grid(self, images, depths, captions=None, save_path=None, grid_size=None):
        """
        Create a grid comparison of multiple depth maps
        
        Args:
            images: List of input images
            depths: List of depth maps
            captions: Optional list of captions
            save_path: Path to save the visualization
            grid_size: Optional tuple (rows, cols) for grid layout
            
        Returns:
            fig: Matplotlib figure
        """
        n = len(images)
        assert n == len(depths), "Number of images and depth maps must match"
        
        # Determine grid size
        if grid_size is None:
            cols = min(3, n)
            rows = (n + cols - 1) // cols
        else:
            rows, cols = grid_size
            assert rows * cols >= n, "Grid size too small for the number of images"
            
        # Create figure
        fig = plt.figure(figsize=(4 * cols, 4 * rows))
        
        for i in range(n):
            # Get image and depth
            image = images[i]
            depth = depths[i]
            
            # Convert to numpy if tensors
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()
            if isinstance(depth, torch.Tensor):
                depth = depth.detach().cpu().numpy()
                
            # Ensure proper dimensions
            image = np.squeeze(image)
            if image.shape[0] == 3 and len(image.shape) == 3:  # [3, H, W]
                image = np.transpose(image, (1, 2, 0))
            depth = np.squeeze(depth)
                
            # Normalize image if needed
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
                
            # Image subplot
            plt.subplot(rows, cols * 2, i * 2 + 1)
            plt.imshow(image)
            if captions is not None and i < len(captions):
                plt.title(captions[i])
            plt.axis('off')
            
            # Depth subplot
            plt.subplot(rows, cols * 2, i * 2 + 2)
            im = plt.imshow(depth, cmap=self.mars_elev_cmap if self.mars_themed else 'magma')
            plt.axis('off')
            
        plt.tight_layout()
        
        # Add a single colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Depth')
        
        # Save if path provided
        if save_path is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            
        return fig
