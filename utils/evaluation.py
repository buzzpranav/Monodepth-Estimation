from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import matplotlib.pyplot as plt
import skimage.measure
import cv2
import os
import json
from tqdm import tqdm


class MarsTerrainEvaluator:
    """
    Evaluation metrics specific to planetary terrain reconstruction
    """
    def __init__(self, min_depth=0.1, max_depth=100.0, use_cuda=True):
        """
        Initialize the Mars terrain evaluator
        
        Args:
            min_depth: Minimum valid depth value
            max_depth: Maximum valid depth value
            use_cuda: Whether to use GPU acceleration
        """
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    
    def evaluate_depth_prediction(self, pred_depth, gt_depth, mask=None):
        """
        Evaluate depth prediction against ground truth
        
        Args:
            pred_depth: Predicted depth tensor or array
            gt_depth: Ground truth depth tensor or array
            mask: Optional mask for valid depth values
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        # Convert to numpy arrays
        if isinstance(pred_depth, torch.Tensor):
            pred_depth = pred_depth.detach().cpu().numpy()
        if isinstance(gt_depth, torch.Tensor):
            gt_depth = gt_depth.detach().cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
            
        # Ensure single-channel depth maps
        pred_depth = np.squeeze(pred_depth)
        gt_depth = np.squeeze(gt_depth)
        
        # Apply mask if provided, otherwise create one for valid depth ranges
        if mask is None:
            mask = (gt_depth > self.min_depth) & (gt_depth < self.max_depth)
        else:
            mask = mask & (gt_depth > self.min_depth) & (gt_depth < self.max_depth)
            
        # Apply mask to depth maps
        pred_masked = pred_depth[mask]
        gt_masked = gt_depth[mask]
        
        # Skip if no valid pixels
        if pred_masked.size == 0:
            return None
            
        # Calculate standard depth metrics
        abs_rel = np.mean(np.abs(pred_masked - gt_masked) / gt_masked)
        sq_rel = np.mean(((pred_masked - gt_masked) ** 2) / gt_masked)
        rmse = np.sqrt(np.mean((pred_masked - gt_masked) ** 2))
        rmse_log = np.sqrt(np.mean((np.log(pred_masked) - np.log(gt_masked)) ** 2))
        
        # Calculate thresholded accuracy
        thresh = np.maximum((gt_masked / pred_masked), (pred_masked / gt_masked))
        a1 = np.mean((thresh < 1.25)).astype(np.float32)
        a2 = np.mean((thresh < 1.25 ** 2)).astype(np.float32)
        a3 = np.mean((thresh < 1.25 ** 3)).astype(np.float32)
        
        # Mars-specific metrics
        
        # 1. Terrain Roughness Preservation (TRP)
        # Measures how well the depth prediction preserves terrain roughness
        gt_rough = self._calculate_roughness(gt_masked)
        pred_rough = self._calculate_roughness(pred_masked)
        trp = np.abs(gt_rough - pred_rough) / gt_rough
        
        # 2. Small Feature Detection Rate (SFDR)
        # Measures how well small terrain features are captured
        # This is a simplified version that looks at high-frequency content preservation
        gt_edges = self._detect_edges(gt_depth)
        pred_edges = self._detect_edges(pred_depth)
        sfdr = np.sum(gt_edges & pred_edges) / (np.sum(gt_edges) + 1e-6)
        
        # 3. Crater Detection Accuracy (CDA)
        # This would typically use a crater detection algorithm
        # Here we use a simplified version based on concave surface detection
        # cda = self._calculate_crater_accuracy(pred_depth, gt_depth, mask)
        # Simplifying for now
        cda = 0.0
        
        # Compile all metrics
        metrics = {
            'abs_rel': abs_rel,
            'sq_rel': sq_rel,
            'rmse': rmse,
            'rmse_log': rmse_log,
            'a1': a1,
            'a2': a2,
            'a3': a3,
            'trp': 1.0 - np.mean(trp),  # Higher is better
            'sfdr': sfdr,
            'cda': cda
        }
        
        return metrics
    
    def _calculate_roughness(self, depth_values):
        """Calculate terrain roughness as standard deviation in local neighborhoods"""
        if len(depth_values) < 9:  # Need at least a small neighborhood
            return 0.0
            
        # Reshape to 2D if possible
        n = int(np.sqrt(len(depth_values)))
        if n**2 == len(depth_values):
            depth_2d = depth_values.reshape(n, n)
            
            # Calculate local standard deviation
            local_std = skimage.measure.block_reduce(
                depth_2d, (max(2, n//10), max(2, n//10)), np.std
            )
            
            # Return mean roughness
            return np.mean(local_std)
        else:
            # If reshaping not possible, just return global std
            return np.std(depth_values)
    
    def _detect_edges(self, depth):
        """Detect edges in depth map as proxy for terrain features"""
        # Apply Sobel edge detection
        sobelx = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        
        # Combine edges
        edges = np.sqrt(sobelx**2 + sobely**2)
        
        # Threshold to get binary edge map
        threshold = np.percentile(edges, 90)  # Top 10% of gradient magnitudes
        binary_edges = edges > threshold
        
        return binary_edges
        
    def evaluate_3d_reconstruction(self, pred_mesh, gt_mesh):
        """
        Evaluate 3D mesh reconstruction against ground truth
        
        Args:
            pred_mesh: Predicted mesh (Open3D TriangleMesh)
            gt_mesh: Ground truth mesh (Open3D TriangleMesh)
            
        Returns:
            metrics: Dictionary of 3D reconstruction metrics
        """
        # Convert meshes to point clouds for comparison
        pred_pcd = pred_mesh.sample_points_uniformly(number_of_points=10000)
        gt_pcd = gt_mesh.sample_points_uniformly(number_of_points=10000)
        
        # Calculate Chamfer distance
        chamfer_dist = self._calculate_chamfer_distance(pred_pcd, gt_pcd)
        
        # Calculate Hausdorff distance
        hausdorff_dist = self._calculate_hausdorff_distance(pred_pcd, gt_pcd)
        
        # Calculate normal consistency
        normal_consistency = self._calculate_normal_consistency(pred_mesh, gt_mesh)
        
        # Calculate volumetric similarity
        vol_similarity = self._calculate_volumetric_similarity(pred_mesh, gt_mesh)
        
        metrics = {
            'chamfer_distance': chamfer_dist,
            'hausdorff_distance': hausdorff_dist,
            'normal_consistency': normal_consistency,
            'volumetric_similarity': vol_similarity
        }
        
        return metrics
    
    def _calculate_chamfer_distance(self, pred_pcd, gt_pcd):
        """Calculate Chamfer distance between two point clouds"""
        import open3d as o3d
        
        # Set up KD-Trees for fast nearest neighbor queries
        pred_tree = o3d.geometry.KDTreeFlann(pred_pcd)
        gt_tree = o3d.geometry.KDTreeFlann(gt_pcd)
        
        # Pred to GT distance
        pred_to_gt = 0
        pred_points = np.asarray(pred_pcd.points)
        for point in pred_points:
            _, idx, _ = gt_tree.search_knn_vector_3d(point, 1)
            nearest_gt = np.asarray(gt_pcd.points)[idx[0]]
            pred_to_gt += np.linalg.norm(point - nearest_gt)
        
        # GT to pred distance
        gt_to_pred = 0
        gt_points = np.asarray(gt_pcd.points)
        for point in gt_points:
            _, idx, _ = pred_tree.search_knn_vector_3d(point, 1)
            nearest_pred = np.asarray(pred_pcd.points)[idx[0]]
            gt_to_pred += np.linalg.norm(point - nearest_pred)
        
        # Average the two distances
        chamfer_dist = (pred_to_gt / len(pred_points) + gt_to_pred / len(gt_points)) / 2
        
        return chamfer_dist
    
    def _calculate_hausdorff_distance(self, pred_pcd, gt_pcd):
        """Calculate Hausdorff distance between two point clouds"""
        import open3d as o3d
        
        # Set up KD-Trees for fast nearest neighbor queries
        pred_tree = o3d.geometry.KDTreeFlann(pred_pcd)
        gt_tree = o3d.geometry.KDTreeFlann(gt_pcd)
        
        # Pred to GT max distance
        pred_to_gt_max = 0
        pred_points = np.asarray(pred_pcd.points)
        for point in pred_points:
            _, idx, _ = gt_tree.search_knn_vector_3d(point, 1)
            nearest_gt = np.asarray(gt_pcd.points)[idx[0]]
            dist = np.linalg.norm(point - nearest_gt)
            pred_to_gt_max = max(pred_to_gt_max, dist)
        
        # GT to pred max distance
        gt_to_pred_max = 0
        gt_points = np.asarray(gt_pcd.points)
        for point in gt_points:
            _, idx, _ = pred_tree.search_knn_vector_3d(point, 1)
            nearest_pred = np.asarray(pred_pcd.points)[idx[0]]
            dist = np.linalg.norm(point - nearest_pred)
            gt_to_pred_max = max(gt_to_pred_max, dist)
        
        # Hausdorff distance is the maximum of the two max distances
        hausdorff_dist = max(pred_to_gt_max, gt_to_pred_max)
        
        return hausdorff_dist
    
    def _calculate_normal_consistency(self, pred_mesh, gt_mesh):
        """Calculate normal consistency between two meshes"""
        # Sample points with normals from both meshes
        pred_pcd = pred_mesh.sample_points_uniformly(number_of_points=5000)
        gt_pcd = gt_mesh.sample_points_uniformly(number_of_points=5000)
        
        # Estimate normals if not present
        if not pred_pcd.has_normals():
            pred_pcd.estimate_normals()
        if not gt_pcd.has_normals():
            gt_pcd.estimate_normals()
            
        import open3d as o3d
        
        # Set up KD-Trees for fast nearest neighbor queries
        pred_tree = o3d.geometry.KDTreeFlann(pred_pcd)
        
        # Calculate normal consistency
        consistency_sum = 0.0
        gt_points = np.asarray(gt_pcd.points)
        gt_normals = np.asarray(gt_pcd.normals)
        pred_normals = np.asarray(pred_pcd.normals)
        
        for i, point in enumerate(gt_points):
            _, idx, _ = pred_tree.search_knn_vector_3d(point, 1)
            pred_normal = pred_normals[idx[0]]
            gt_normal = gt_normals[i]
            
            # Calculate dot product of normals (cosine similarity)
            dot_product = np.abs(np.dot(gt_normal, pred_normal))
            consistency_sum += dot_product
        
        normal_consistency = consistency_sum / len(gt_points)
        
        return normal_consistency
    
    def _calculate_volumetric_similarity(self, pred_mesh, gt_mesh):
        """Calculate volumetric similarity between two meshes"""
        pred_vol = pred_mesh.get_volume()
        gt_vol = gt_mesh.get_volume()
        
        # Avoid division by zero
        max_vol = max(pred_vol, gt_vol)
        if max_vol == 0:
            return 0.0
            
        # Calculate similarity (1.0 is perfect match)
        vol_similarity = 1.0 - abs(pred_vol - gt_vol) / max_vol
        
        return vol_similarity
    
    def evaluate_batch(self, model, dataloader, save_dir=None):
        """
        Evaluate model on a batch of data
        
        Args:
            model: DepthAnything model
            dataloader: DataLoader for evaluation data
            save_dir: Directory to save evaluation results
            
        Returns:
            avg_metrics: Dictionary of average evaluation metrics
        """
        if model is not None:
            model.eval()
            
        all_metrics = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                # Get data
                images = batch['image'].to(self.device)
                metadata = batch['metadata']
                
                # If ground truth depth is available
                if 'depth' in batch:
                    gt_depth = batch['depth'].to(self.device)
                    
                    # Forward pass
                    outputs = model(images, metadata)
                    pred_disp = outputs[("disp", 0)]
                    
                    # Convert disparity to depth
                    _, pred_depth = disp_to_depth(pred_disp, min_depth=self.min_depth, 
                                                max_depth=self.max_depth)
                    
                    # Evaluate
                    for b in range(images.shape[0]):
                        metrics = self.evaluate_depth_prediction(
                            pred_depth[b], gt_depth[b], None)
                            
                        if metrics is not None:
                            all_metrics.append(metrics)
                            
                            # Save visualization
                            if save_dir is not None:
                                os.makedirs(save_dir, exist_ok=True)
                                self.save_depth_visualization(
                                    images[b], pred_depth[b], gt_depth[b],
                                    os.path.join(save_dir, f"sample_{batch_idx}_{b}.png")
                                )
        
        # Calculate average metrics
        if not all_metrics:
            return None
            
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
            
        # Save metrics to JSON
        if save_dir is not None:
            with open(os.path.join(save_dir, "metrics.json"), 'w') as f:
                json.dump(avg_metrics, f, indent=2)
                
        return avg_metrics
    
    def save_depth_visualization(self, image, pred_depth, gt_depth, filename):
        """
        Save a visualization of depth prediction
        
        Args:
            image: Input image tensor
            pred_depth: Predicted depth tensor
            gt_depth: Ground truth depth tensor
            filename: Output filename
        """
        # Convert to numpy
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if isinstance(pred_depth, torch.Tensor):
            pred_depth = pred_depth.detach().cpu().numpy()
        if isinstance(gt_depth, torch.Tensor):
            gt_depth = gt_depth.detach().cpu().numpy()
            
        # Ensure proper dimensions
        image = np.squeeze(image)
        if image.shape[0] == 3:  # [3, H, W]
            image = np.transpose(image, (1, 2, 0))
        pred_depth = np.squeeze(pred_depth)
        gt_depth = np.squeeze(gt_depth)
            
        # Normalize image if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
            
        # Create visualization
        plt.figure(figsize=(15, 5))
        
        # Input image
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Input Image")
        plt.axis('off')
        
        # Predicted depth
        plt.subplot(1, 3, 2)
        plt.imshow(pred_depth, cmap='magma')
        plt.title("Predicted Depth")
        plt.colorbar(label='Depth')
        plt.axis('off')
        
        # Ground truth depth
        plt.subplot(1, 3, 3)
        plt.imshow(gt_depth, cmap='magma')
        plt.title("Ground Truth Depth")
        plt.colorbar(label='Depth')
        plt.axis('off')
        
        # Save and close
        plt.tight_layout()
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close()
    
    def compute_proxy_metrics(self, depth):
        """
        Compute proxy metrics for depth map quality when ground truth is not available
        
        Args:
            depth: Depth map as numpy array
            
        Returns:
            metrics: Dictionary of proxy quality metrics
        """
        # Ensure depth is a numpy array
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()
            
        # Ensure single-channel depth map
        depth = np.squeeze(depth)
        
        # 1. Edge preservation score - detects how well-defined the edges are
        edges = self._detect_edges(depth)
        edge_score = np.sum(edges) / (depth.shape[0] * depth.shape[1])
        edge_score = min(edge_score * 20.0, 1.0)  # Normalize and cap
        
        # 2. Detail preservation - measures local variance which indicates detail
        roughness = self._calculate_roughness(depth.flatten())
        # Higher roughness generally means more detail, but we want a normalized score
        detail_score = min(roughness * 5.0, 1.0)  # Normalize and cap
        
        # 3. Noise level - estimate using local variation compared to neighborhood
        # Calculate noise using Laplacian filter which enhances noise
        laplacian = cv2.Laplacian(depth.astype(np.float32), cv2.CV_32F)
        noise_level = np.mean(np.abs(laplacian)) / (np.max(depth) - np.min(depth) + 1e-6)
        noise_level = min(noise_level * 10.0, 1.0)  # Normalize and cap
        
        # 4. Depth distribution - a good depth map should have a reasonable distribution
        # Very sparse or concentrated depth values are often problematic
        hist, _ = np.histogram(depth, bins=20)
        hist_normalized = hist / np.sum(hist)
        depth_distribution_score = 1.0 - np.max(hist_normalized)  # Penalize concentrated values
        
        # Combine metrics
        metrics = {
            "edge_preservation_score": edge_score,
            "detail_preservation_score": detail_score,
            "noise_level": noise_level,
            "depth_distribution_score": depth_distribution_score
        }
        
        return metrics


# Import function for disparity to depth conversion
from layers import disp_to_depth
