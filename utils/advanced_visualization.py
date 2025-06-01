from __future__ import absolute_import, division, print_function

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import cv2
from PIL import Image
from tqdm import tqdm
import math

# Try to import open3d for better 3D visualization
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("Warning: open3d not available. 3D visualization features will be limited.")
    OPEN3D_AVAILABLE = False

# Try to import plotly for interactive visualizations
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Warning: plotly not available. Interactive visualizations will be disabled.")
    PLOTLY_AVAILABLE = False

class AdvancedMarsVisualizer:
    """
    Advanced visualization tools for Mars terrain analysis with multiple view types
    and interactive visualizations.
    """
    def __init__(self, min_depth=0.1, max_depth=100.0):
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        # Define Mars-themed colormaps
        self._setup_mars_colormaps()
    def _setup_mars_colormaps(self):
        """Define custom Mars-themed colormaps"""
        # Mars surface colormap - various terrain types
        self.mars_terrain_colors = [
            (0.42, 0.2, 0.04),   # Dark reddish-brown (lowlands)
            (0.55, 0.3, 0.15),   # Medium reddish-brown
            (0.67, 0.42, 0.27),  # Light reddish-brown
            (0.78, 0.57, 0.39),  # Tan
            (0.88, 0.71, 0.53)   # Light tan (highlands)
        ]
        self.mars_terrain_cmap = LinearSegmentedColormap.from_list("mars_terrain", self.mars_terrain_colors)
        
        # Olympus Mons style elevation map (blue to white via red)
        self.mars_elevation_colors = [
            (0.01, 0.01, 0.35),  # Deep blue (deep canyons)
            (0.3, 0.15, 0.2),    # Deep red (midlands)
            (0.6, 0.22, 0.12),   # Reddish-brown (average terrain)
            (0.8, 0.5, 0.3),     # Light brown (highlands)
            (1.0, 0.95, 0.9)     # Near-white (mountain peaks)
        ]
        self.mars_elevation_cmap = LinearSegmentedColormap.from_list("mars_elevation", self.mars_elevation_colors)
        
        # Scientific thermal map style (for uncertainty visualization)
        self.thermal_colors = [
            (0, 0, 0.7),        # Deep blue (low uncertainty)
            (0, 0.5, 1),        # Light blue
            (0, 0.8, 0.3),      # Green
            (1, 0.8, 0),        # Yellow
            (1, 0, 0)           # Red (high uncertainty)
        ]
        self.thermal_cmap = LinearSegmentedColormap.from_list("thermal", self.thermal_colors)
        
    def _import_fallback_methods(self):
        """Import fallback visualization methods"""
        # Check if fallbacks file exists
        fallback_file = os.path.join(os.path.dirname(__file__), "advanced_visualization_fallbacks.py")
        if os.path.exists(fallback_file):
            with open(fallback_file, 'r') as f:
                fallback_code = f.read()
                
            # Extract method definitions and add them to the class
            # Safer than exec, this just adds the methods as string attributes
            # which will be properly added to the class during runtime
            import types
            
            # Extract the methods from the file content
            method_lines = fallback_code.strip().split('\n')
            
            # Define the methods directly
            exec(fallback_code.strip(), globals(), self.__dict__)
            
            print("Fallback visualization methods loaded successfully.")
        else:
            # Define minimal fallbacks inline
            def _create_static_fallback_visualization(self, rgb_image, depth_map, uncertainty=None, save_path=None):
                """Simple fallback for interactive visualizations"""
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                
                # Simple depth visualization
                if isinstance(depth_map, torch.Tensor):
                    depth_map = depth_map.detach().cpu().numpy()
                depth_map = np.squeeze(depth_map)
                
                axes[0].imshow(depth_map, cmap='viridis')
                axes[0].set_title("Depth Map")
                axes[1].set_title("Interactive visualization not available")
                axes[1].text(0.5, 0.5, "Install plotly for interactive visualizations", 
                            ha='center', va='center', transform=axes[1].transAxes)
                
                if save_path:
                    plt.savefig(save_path)
                
                return fig
                
            def _create_static_3d_view(self, depth_map, rgb_image=None, save_path=None):
                """Simple fallback for 3D visualization"""
                fig, ax = plt.subplots(figsize=(8, 6))
                
                if isinstance(depth_map, torch.Tensor):
                    depth_map = depth_map.detach().cpu().numpy()
                depth_map = np.squeeze(depth_map)
                
                im = ax.imshow(depth_map, cmap='viridis')
                ax.set_title("3D visualization not available")
                ax.text(0.5, 0.5, "Install plotly and open3d for 3D visualizations", 
                       ha='center', va='center', transform=ax.transAxes)
                plt.colorbar(im, ax=ax, label="Depth")
                
                if save_path:
                    plt.savefig(save_path)
                
                return fig
                
            # Add methods to the class
            self._create_static_fallback_visualization = types.MethodType(_create_static_fallback_visualization, self)
            self._create_static_3d_view = types.MethodType(_create_static_3d_view, self)
        
    def create_multi_view_visualization(self, rgb_image, depth_map, uncertainty=None, normals=None, 
                                       save_path=None, show=True, view_3d=True, interactive=False):
        """
        Create a multi-view visualization of the Mars terrain
        
        Args:
            rgb_image: Original RGB image
            depth_map: Depth map
            uncertainty: Optional uncertainty map
            normals: Optional surface normals map
            save_path: Path to save the visualization
            show: Whether to display the visualization
            view_3d: Whether to create 3D visualization
            interactive: Whether to create interactive visualization (requires plotly)
            
        Returns:
            fig: Matplotlib figure
        """
        # Convert tensors to numpy arrays if needed
        if isinstance(rgb_image, torch.Tensor):
            rgb_image = rgb_image.detach().cpu().numpy()
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.detach().cpu().numpy()
        if isinstance(uncertainty, torch.Tensor) and uncertainty is not None:
            uncertainty = uncertainty.detach().cpu().numpy()
        if isinstance(normals, torch.Tensor) and normals is not None:
            normals = normals.detach().cpu().numpy()
            
        # Ensure proper dimensions
        rgb_image = np.squeeze(rgb_image)
        if rgb_image.shape[0] == 3 and len(rgb_image.shape) == 3:  # [3, H, W]
            rgb_image = np.transpose(rgb_image, (1, 2, 0))
        depth_map = np.squeeze(depth_map)
        
        # Normalize RGB image if needed
        if rgb_image.max() <= 1.0:
            rgb_image = (rgb_image * 255).astype(np.uint8)
            
        # Create interactive visualization if requested
        if interactive and PLOTLY_AVAILABLE:
            return self.create_interactive_visualization(
                rgb_image, depth_map, uncertainty, normals, save_path
            )
            
        # Determine layout based on available data
        n_cols = 2  # Always RGB and depth
        if uncertainty is not None:
            n_cols += 1
        if normals is not None:
            n_cols += 1
        
        # Create figure
        fig = plt.figure(figsize=(n_cols * 5, 8))
        
        # Add RGB image
        ax1 = fig.add_subplot(2, n_cols, 1)
        ax1.imshow(rgb_image)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Add depth map with colormap
        ax2 = fig.add_subplot(2, n_cols, 2)
        depth_vis = ax2.imshow(depth_map, cmap=self.mars_elevation_cmap)
        ax2.set_title("Depth Map")
        ax2.axis('off')
        plt.colorbar(depth_vis, ax=ax2, label="Depth", fraction=0.046, pad=0.04)
        
        # Add uncertainty if available
        curr_idx = 3
        if uncertainty is not None:
            ax_unc = fig.add_subplot(2, n_cols, curr_idx)
            unc_vis = ax_unc.imshow(uncertainty, cmap=self.thermal_cmap)
            ax_unc.set_title("Uncertainty")
            ax_unc.axis('off')
            plt.colorbar(unc_vis, ax=ax_unc, label="Uncertainty", fraction=0.046, pad=0.04)
            curr_idx += 1
            
        # Add surface normals if available
        if normals is not None:
            ax_norm = fig.add_subplot(2, n_cols, curr_idx)
            # Normalize and rescale normals for visualization
            normals_vis = (normals + 1) / 2  # Convert from [-1,1] to [0,1]
            ax_norm.imshow(normals_vis)
            ax_norm.set_title("Surface Normals")
            ax_norm.axis('off')
            curr_idx += 1
            
        # Add 3D visualization if requested
        if view_3d:
            # Create point cloud visualization using matplotlib
            ax3d = fig.add_subplot(2, n_cols, n_cols + 1, projection='3d')
            
            # Downsample for better visualization
            step = max(1, min(depth_map.shape) // 100)  # Adaptive step size
            y, x = np.mgrid[0:depth_map.shape[0]:step, 0:depth_map.shape[1]:step]
            
            # Get subset of depth values
            z = depth_map[::step, ::step]
            
            # Get subset of colors
            rgb_sub = rgb_image[::step, ::step] / 255.0 if rgb_image is not None else None
            
            # Normalize Z range to avoid extreme values
            z_vis = np.clip(z, self.min_depth, np.percentile(z, 95))
            
            # Create surface plot
            if rgb_sub is not None and rgb_sub.ndim == 3:
                colors = rgb_sub.reshape(-1, 3)
                ax3d.plot_surface(x, y, z_vis, facecolors=rgb_sub, rstride=1, cstride=1, 
                                 shade=False, alpha=0.8)
            else:
                ax3d.plot_surface(x, y, z_vis, cmap=self.mars_terrain_cmap, alpha=0.8)
                
            ax3d.set_title("3D Terrain Visualization")
            
            # Set consistent aspect ratio
            max_range = max(np.ptp(x), np.ptp(y), np.ptp(z_vis))
            mid_x = np.mean([x.min(), x.max()])
            mid_y = np.mean([y.min(), y.max()])
            mid_z = np.mean([z_vis.min(), z_vis.max()])
            ax3d.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
            ax3d.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
            ax3d.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
            
            # Equal aspect ratio
            ax3d.set_box_aspect([1, 1, 0.3])
            
        # Additional visualizations for special features
        
        # Depth contour map (2 pixels apart in the plot)
        if n_cols >= 2:
            ax_contour = fig.add_subplot(2, n_cols, n_cols + 2)
            contour = ax_contour.contour(depth_map, cmap=self.mars_elevation_cmap, levels=15)
            ax_contour.contourf(depth_map, cmap=self.mars_elevation_cmap, levels=15, alpha=0.5)
            ax_contour.set_title("Depth Contours")
            plt.colorbar(contour, ax=ax_contour, label="Depth", fraction=0.046, pad=0.04)
            ax_contour.axis('off')
        
        # Save figure if requested
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        # Show or close
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    def create_interactive_visualization(self, rgb_image, depth_map, uncertainty=None, 
                                       normals=None, save_path=None):
        """
        Create an interactive visualization using Plotly
        
        Args:
            rgb_image: Original RGB image
            depth_map: Depth map
            uncertainty: Optional uncertainty map
            normals: Optional surface normals map
            save_path: Path to save the visualization
            
        Returns:
            fig: Plotly figure or static matplotlib figure as fallback
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly is not available. Creating static visualization as fallback.")
            # Create a static fallback visualization instead
            return self._create_static_fallback_visualization(rgb_image, depth_map, uncertainty, save_path)
            
        # Create subplots layout
        n_rows = 1
        n_cols = 3  # Always RGB, depth and 3D view
        if uncertainty is not None:
            n_cols += 1
        
        # Create figure
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            specs=[[{"type": "image"}, {"type": "image"}, 
                  {"type": "scene"}] + ([{"type": "image"}] if uncertainty is not None else [])],
            subplot_titles=["Original Image", "Depth Map", "3D Terrain"] + 
                        (["Uncertainty"] if uncertainty is not None else [])
        )
        
        # Add RGB image
        fig.add_trace(
            go.Image(z=rgb_image),
            row=1, col=1
        )
        
        # Create depth heatmap 
        fig.add_trace(
            go.Heatmap(z=depth_map, colorscale="Viridis", zmin=self.min_depth, 
                     zmax=np.percentile(depth_map, 95)),
            row=1, col=2
        )
        
        # Add uncertainty if available
        if uncertainty is not None:
            fig.add_trace(
                go.Heatmap(z=uncertainty, colorscale="Thermal", zmin=0, 
                         zmax=np.percentile(uncertainty, 95)),
                row=1, col=4
            )
            
        # Create 3D surface plot
        # Downsample for performance
        step = max(1, min(depth_map.shape) // 100)
        y, x = np.mgrid[0:depth_map.shape[0]:step, 0:depth_map.shape[1]:step]
        z = depth_map[::step, ::step]
        
        # Normalize Z range to avoid extreme values
        z_vis = np.clip(z, self.min_depth, np.percentile(z, 95))
        
        # Create the surface
        fig.add_trace(
            go.Surface(z=z_vis, x=x, y=y, colorscale="Earth", opacity=0.9),
            row=1, col=3
        )
        
        # Update layout
        fig.update_layout(
            title_text="Interactive Mars Terrain Visualization",
            height=600,
            width=n_cols * 400,
            scene=dict(
                aspectratio=dict(x=1, y=1, z=0.3)
            )
        )
        
        # Save if path provided
        if save_path:
            # Change extension to html for interactive plot
            html_path = os.path.splitext(save_path)[0] + '.html'
            fig.write_html(html_path)
            print(f"Interactive visualization saved to {html_path}")
            
        return fig    
    def create_terrain_flyover_animation(self, depth_map, rgb_image, save_path, frames=60, 
                                       elevation_range=(20, 70), azimuth_range=(0, 360)):
        """
        Create a flyover animation of the terrain
        
        Args:
            depth_map: Depth map
            rgb_image: RGB image for texturing
            save_path: Path to save animation
            frames: Number of frames
            elevation_range: Range of elevation angles
            azimuth_range: Range of azimuth angles
            
        Returns:
            Animation object or static figure as fallback
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly is not available. Creating static 3D view as fallback.")
            # Create a static 3D view instead
            return self._create_static_3d_view(depth_map, rgb_image, save_path)
            
        # Downsample for performance
        step = max(1, min(depth_map.shape) // 80)
        y, x = np.mgrid[0:depth_map.shape[0]:step, 0:depth_map.shape[1]:step]
        z = depth_map[::step, ::step]
        z_vis = np.clip(z, self.min_depth, np.percentile(z, 95))
        
        # Get downsampled RGB for texturing
        rgb_sub = None
        if rgb_image is not None:
            if isinstance(rgb_image, torch.Tensor):
                rgb_image = rgb_image.detach().cpu().numpy()
            rgb_image = np.squeeze(rgb_image)
            if rgb_image.shape[0] == 3 and len(rgb_image.shape) == 3:
                rgb_image = np.transpose(rgb_image, (1, 2, 0))
            if rgb_image.max() <= 1.0:
                rgb_image = (rgb_image * 255).astype(np.uint8)
                
            rgb_sub = rgb_image[::step, ::step]
            
        # Create frames for the animation
        frames_list = []
        for i in range(frames):
            # Calculate camera position
            elev_progress = i / frames
            azim_progress = i / frames
            elevation = elevation_range[0] + elev_progress * (elevation_range[1] - elevation_range[0])
            azimuth = azimuth_range[0] + azim_progress * (azimuth_range[1] - azimuth_range[0])
            
            # Create frame
            frame = go.Frame(
                data=[go.Surface(
                    z=z_vis, 
                    x=x, 
                    y=y, 
                    colorscale="Earth" if rgb_sub is None else None,
                    surfacecolor=None if rgb_sub is None else rgb_sub,
                    opacity=0.9
                )],
                layout=go.Layout(
                    scene_camera=dict(
                        eye=dict(
                            x=math.cos(math.radians(azimuth)) * math.cos(math.radians(elevation)),
                            y=math.sin(math.radians(azimuth)) * math.cos(math.radians(elevation)),
                            z=math.sin(math.radians(elevation))
                        ),
                        up=dict(x=0, y=0, z=1)
                    )
                )
            )
            frames_list.append(frame)
            
        # Create base figure
        fig = go.Figure(
            data=[go.Surface(
                z=z_vis, 
                x=x, 
                y=y, 
                colorscale="Earth" if rgb_sub is None else None,
                surfacecolor=None if rgb_sub is None else rgb_sub,
                opacity=0.9
            )],
            layout=go.Layout(
                title="Mars Terrain Flyover Animation",
                updatemenus=[{
                    "buttons": [
                        {
                            "args": [None, {"frame": {"duration": 50, "redraw": True},
                                         "fromcurrent": True}],
                            "label": "Play",
                            "method": "animate"
                        },
                        {
                            "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                          "mode": "immediate"}],
                            "label": "Pause",
                            "method": "animate"
                        }
                    ],
                    "type": "buttons"
                }],
                scene=dict(
                    aspectratio=dict(x=1, y=1, z=0.3)
                )
            ),
            frames=frames_list
        )
        
        if save_path:
            html_path = os.path.splitext(save_path)[0] + '_flyover.html'
            fig.write_html(html_path)
            print(f"Flyover animation saved to {html_path}")
            
        return fig
        
    def create_depth_uncertainty_visualization(self, depth, uncertainty, rgb_image=None, 
                                            save_path=None, show=True):
        """
        Create visualization of depth with uncertainty overlay
        
        Args:
            depth: Depth map
            uncertainty: Uncertainty map
            rgb_image: Optional RGB image
            save_path: Path to save visualization
            show: Whether to show the visualization
            
        Returns:
            fig: Matplotlib figure
        """
        # Convert to numpy arrays
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()
        if isinstance(uncertainty, torch.Tensor):
            uncertainty = uncertainty.detach().cpu().numpy()
        if isinstance(rgb_image, torch.Tensor) and rgb_image is not None:
            rgb_image = rgb_image.detach().cpu().numpy()
            
        depth = np.squeeze(depth)
        uncertainty = np.squeeze(uncertainty)
        
        # Scale uncertainty to [0, 1]
        norm_uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-8)
        
        # Create figure with different views
        n_cols = 3 if rgb_image is not None else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 5, 5))
        
        # Plot RGB if available
        col_idx = 0
        if rgb_image is not None:
            rgb_image = np.squeeze(rgb_image)
            if rgb_image.shape[0] == 3 and len(rgb_image.shape) == 3:
                rgb_image = np.transpose(rgb_image, (1, 2, 0))
            if rgb_image.max() <= 1.0:
                rgb_image = (rgb_image * 255).astype(np.uint8)
                
            axes[col_idx].imshow(rgb_image)
            axes[col_idx].set_title("Input Image")
            axes[col_idx].axis('off')
            col_idx += 1
            
        # Plot depth map
        depth_vis = axes[col_idx].imshow(depth, cmap=self.mars_elevation_cmap)
        axes[col_idx].set_title("Depth Map")
        plt.colorbar(depth_vis, ax=axes[col_idx], label="Depth", fraction=0.046, pad=0.04)
        axes[col_idx].axis('off')
        col_idx += 1
        
        # Plot depth with uncertainty overlay
        # Create RGBA image with uncertainty as alpha
        depth_colored = plt.cm.plasma(plt.Normalize()(depth))
        uncertainty_colored = plt.cm.hot(plt.Normalize()(uncertainty))
        
        # Set alpha based on uncertainty - higher uncertainty is more visible
        uncertainty_overlay = uncertainty_colored.copy()
        uncertainty_overlay[..., 3] = norm_uncertainty * 0.7  # Semi-transparent
        
        # Plot both
        axes[col_idx].imshow(depth_colored)
        axes[col_idx].imshow(uncertainty_overlay)
        axes[col_idx].set_title("Depth with Uncertainty Overlay")
        
        # Create custom colorbar for uncertainty
        sm = plt.cm.ScalarMappable(cmap=plt.cm.hot, norm=plt.Normalize(vmin=uncertainty.min(), vmax=uncertainty.max()))
        cbar = plt.colorbar(sm, ax=axes[col_idx], label="Uncertainty", fraction=0.046, pad=0.04)
        axes[col_idx].axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        # Show or close
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
        
    def visualize_terrain_features(self, depth_map, rgb_image=None, save_path=None, show=True):
        """
        Visualize terrain features like slopes, roughness, etc.
        
        Args:
            depth_map: Depth map
            rgb_image: Optional RGB image
            save_path: Path to save visualization
            show: Whether to show the visualization
            
        Returns:
            fig: Matplotlib figure
        """
        # Convert tensors to numpy
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.detach().cpu().numpy()
        if isinstance(rgb_image, torch.Tensor) and rgb_image is not None:
            rgb_image = rgb_image.detach().cpu().numpy()
            
        depth_map = np.squeeze(depth_map)
        
        # Calculate surface features
        
        # 1. Calculate gradients for slope
        gy, gx = np.gradient(depth_map)
        slope = np.sqrt(gx**2 + gy**2)
        
        # 2. Calculate roughness (local variance)
        from scipy.ndimage import uniform_filter
        
        # Local mean
        mean = uniform_filter(depth_map, size=5)
        # Local squared mean
        mean_sq = uniform_filter(depth_map**2, size=5)
        # Variance
        roughness = mean_sq - mean**2
        
        # 3. Calculate local curvature (for detecting craters, hills)
        # Laplacian approximation
        from scipy.ndimage import laplace
        curvature = laplace(depth_map)
        
        # Normalize features for visualization
        norm_slope = np.clip((slope - slope.min()) / (slope.max() - slope.min() + 1e-8), 0, 1)
        norm_roughness = np.clip((roughness - roughness.min()) / (roughness.max() - roughness.min() + 1e-8), 0, 1)
        norm_curvature = np.clip((curvature - curvature.min()) / (curvature.max() - curvature.min() + 1e-8), 0, 1)
        
        # Create figure
        n_cols = 4 if rgb_image is not None else 3
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Plot RGB if available or depth
        if rgb_image is not None:
            rgb_image = np.squeeze(rgb_image)
            if rgb_image.shape[0] == 3 and len(rgb_image.shape) == 3:
                rgb_image = np.transpose(rgb_image, (1, 2, 0))
            if rgb_image.max() <= 1.0:
                rgb_image = (rgb_image * 255).astype(np.uint8)
                
            axes[0].imshow(rgb_image)
            axes[0].set_title("Original Image")
        else:
            axes[0].imshow(depth_map, cmap='viridis')
            axes[0].set_title("Depth Map")
        axes[0].axis('off')
        
        # Plot slope
        slope_vis = axes[1].imshow(norm_slope, cmap='hot')
        axes[1].set_title("Terrain Slope")
        plt.colorbar(slope_vis, ax=axes[1], label="Relative Slope", fraction=0.046, pad=0.04)
        axes[1].axis('off')
        
        # Plot roughness
        rough_vis = axes[2].imshow(norm_roughness, cmap='viridis')
        axes[2].set_title("Terrain Roughness")
        plt.colorbar(rough_vis, ax=axes[2], label="Relative Roughness", fraction=0.046, pad=0.04)
        axes[2].axis('off')
        
        # Plot curvature
        curve_vis = axes[3].imshow(norm_curvature, cmap='seismic')
        axes[3].set_title("Surface Curvature")
        plt.colorbar(curve_vis, ax=axes[3], label="Relative Curvature", fraction=0.046, pad=0.04)
        axes[3].axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        # Show or close
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig

    def create_anaglyph_3d(self, rgb_image, depth_map, shift_strength=0.05, save_path=None, show=True):
        """
        Create a red-cyan anaglyph for 3D viewing with red-cyan glasses
        
        Args:
            rgb_image: RGB image
            depth_map: Depth map for parallax calculation
            shift_strength: Strength of the depth-based shift
            save_path: Path to save the anaglyph
            show: Whether to show the anaglyph
            
        Returns:
            anaglyph: Red-cyan anaglyph image
        """
        # Convert tensors to numpy arrays
        if isinstance(rgb_image, torch.Tensor):
            rgb_image = rgb_image.detach().cpu().numpy()
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.detach().cpu().numpy()
            
        # Ensure proper dimensions
        rgb_image = np.squeeze(rgb_image)
        if rgb_image.shape[0] == 3 and len(rgb_image.shape) == 3:  # [3, H, W]
            rgb_image = np.transpose(rgb_image, (1, 2, 0))
        depth_map = np.squeeze(depth_map)
        
        # Normalize RGB image if needed
        if rgb_image.max() <= 1.0:
            rgb_image = (rgb_image * 255).astype(np.uint8)
            
        # Convert RGB to grayscale for better anaglyph
        if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3:
            gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = rgb_image
            
        # Calculate shift amount based on depth
        # Normalize depth map to [0, 1]
        norm_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        shift = (norm_depth * shift_strength * gray_image.shape[1]).astype(np.int32)
        
        # Create left and right images
        height, width = gray_image.shape[:2]
        left = np.zeros_like(gray_image)
        right = np.zeros_like(gray_image)
        
        # Apply depth-based shift
        for y in range(height):
            for x in range(width):
                # Calculate shift for this pixel
                offset = shift[y, x]
                
                # Left shift
                left_x = max(0, x - offset)
                left[y, left_x] = gray_image[y, x]
                
                # Right shift
                right_x = min(width - 1, x + offset)
                right[y, right_x] = gray_image[y, x]
        
        # Create anaglyph (left=red, right=cyan)
        anaglyph = np.zeros((height, width, 3), dtype=np.uint8)
        anaglyph[..., 0] = left  # Red channel from left image
        anaglyph[..., 1] = right  # Green channel from right image
        anaglyph[..., 2] = right  # Blue channel from right image
        
        # Display and save if requested
        if show or save_path:
            plt.figure(figsize=(12, 8))
            plt.imshow(anaglyph)
            plt.title("Red-Cyan 3D Anaglyph (Use Red-Cyan 3D Glasses)")
            plt.axis('off')
            
            if save_path:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                
                # Also save as image file
                cv2.imwrite(os.path.splitext(save_path)[0] + '_anaglyph.png', 
                          cv2.cvtColor(anaglyph, cv2.COLOR_RGB2BGR))
            
            if show:
                plt.show()
            else:
                plt.close()
                
        return anaglyph
