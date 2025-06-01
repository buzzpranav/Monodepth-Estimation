from __future__ import absolute_import, division, print_function

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from models.depth_anything_model import DepthAnything
from data.mars_dataset import get_mars_dataloaders
from layers import disp_to_depth, compute_depth_errors, get_smooth_loss


class MarsDomainAdapter:
    """
    Fine-tuning and domain adaptation for Mars terrain
    """
    def __init__(self, config):
        """
        Initialize the Mars domain adapter
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create the model
        self.model = DepthAnything(pretrained=True)
        self.model.to(self.device)
        
        # Load Mars dataloaders
        self.dataloaders = get_mars_dataloaders(
            data_path=config.get('data_path', './mars_data'),
            batch_size=config.get('batch_size', 8),
            height=config.get('height', 384),
            width=config.get('width', 512),
            source_type=config.get('source_type', 'mixed')
        )
        
        # Setup optimizer
        self.learning_rate = config.get('learning_rate', 1e-4)
        
        # Use different learning rates for encoder and decoder
        encoder_params = list(self.model.encoder.parameters())
        decoder_params = list(self.model.decoder.parameters())
        
        self.optimizer = optim.Adam([
            {'params': encoder_params, 'lr': self.learning_rate / 10},
            {'params': decoder_params, 'lr': self.learning_rate}
        ])
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=config.get('lr_scheduler_step_size', 10),
            gamma=config.get('lr_scheduler_gamma', 0.5)
        )
        
        # Loss weights
        self.loss_weights = {
            'depth': config.get('depth_loss_weight', 1.0),
            'smoothness': config.get('smoothness_loss_weight', 0.001)
        }
        
        # Save directories
        self.save_dir = config.get('save_dir', './trained_models')
        os.makedirs(self.save_dir, exist_ok=True)
        
    def finetune(self, num_epochs=20):
        """
        Fine-tune the model on Mars terrain data
        
        Args:
            num_epochs: Number of training epochs
        """
        # Print training configuration
        print(f"Starting Mars domain adaptation...")
        print(f"  Model: Depth Anything")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Batch size: {self.config.get('batch_size', 8)}")
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_losses = self.train_one_epoch()
            
            # Validate
            val_losses, val_metrics = self.validate()
            
            # Step the scheduler
            self.scheduler.step()
            
            # Log results
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train loss: {train_losses['total']:.4f}")
            print(f"  Val loss: {val_losses['total']:.4f}")
            print(f"  Val metrics:")
            for k, v in val_metrics.items():
                print(f"    {k}: {v:.4f}")
            print(f"  Time: {time.time() - epoch_start_time:.2f}s")
            
            # Save model checkpoint
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                self.save_model(f"mars_depth_anything_epoch_{epoch+1}.pth")
                
        print("Mars domain adaptation completed!")
        
    def train_one_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_losses = {'total': 0.0, 'depth': 0.0, 'smoothness': 0.0}
        
        train_loader = self.dataloaders['train']
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader)
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)
            metadata = batch['metadata']
            
            # Forward pass
            outputs = self.model(images, metadata)
            
            # Calculate losses
            losses = self.compute_losses(images, outputs)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            self.optimizer.step()
            
            # Update loss totals
            for k, v in losses.items():
                total_losses[k] += v.item()
                
            # Update progress bar
            pbar.set_description(f"Train loss: {losses['total'].item():.4f}")
            
        # Average losses over batches
        for k in total_losses.keys():
            total_losses[k] /= num_batches
            
        return total_losses
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_losses = {'total': 0.0, 'depth': 0.0, 'smoothness': 0.0}
        metrics = {'abs_rel': 0.0, 'sq_rel': 0.0, 'rmse': 0.0, 'a1': 0.0, 'a2': 0.0, 'a3': 0.0}
        
        val_loader = self.dataloaders['val']
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Move data to device
                images = batch['image'].to(self.device)
                metadata = batch['metadata']
                
                # Forward pass
                outputs = self.model(images, metadata)
                
                # Calculate losses
                losses = self.compute_losses(images, outputs)
                
                # Update loss totals
                for k, v in losses.items():
                    total_losses[k] += v.item()
                    
                # Compute depth metrics (simulating ground truth for this example)
                # In a real scenario, you'd use actual ground truth depth
                disp = outputs[("disp", 0)]
                
                # Convert disparity to depth
                _, depth = disp_to_depth(disp, min_depth=0.1, max_depth=100.0)
                
                # For validation, we simulate ground truth using a modified version of prediction
                # In a real scenario, replace this with actual ground truth
                # This is just a placeholder for the structure
                fake_gt = depth * (0.9 + 0.2 * torch.rand_like(depth))
                
                # Compute metrics
                batch_metrics = compute_depth_errors(fake_gt, depth)
                
                # Update metrics
                for i, metric_name in enumerate(['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']):
                    if i < len(batch_metrics) and metric_name in metrics:
                        metrics[metric_name] += batch_metrics[i].item()
        
        # Average losses and metrics over batches
        for k in total_losses.keys():
            total_losses[k] /= num_batches
            
        for k in metrics.keys():
            metrics[k] /= num_batches
            
        return total_losses, metrics
    
    def compute_losses(self, images, outputs):
        """
        Compute training losses
        
        Args:
            images: Input images [B, 3, H, W]
            outputs: Model outputs dictionary
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Get disparity from finest scale
        disp = outputs[("disp", 0)]
        
        # Convert disparity to depth
        scaled_disp, depth = disp_to_depth(disp, min_depth=0.1, max_depth=100.0)
        
        # Self-supervised loss based on image smoothness
        # Encourage disparity to be locally smooth with an edge-aware term
        smoothness_loss = get_smooth_loss(disp, images)
        losses['smoothness'] = self.loss_weights['smoothness'] * smoothness_loss
        
        # For fully supervised training with ground truth depth, you would add:
        # depth_loss = compute_depth_loss(gt_depth, depth)
        # losses['depth'] = self.loss_weights['depth'] * depth_loss
        
        # For this example, we'll use a simple L1 loss on the disparity
        # This simulates a reconstruction-based loss
        # In a real scenario, replace this with your actual supervision signal
        target_disp = disp.detach() * (0.9 + 0.2 * torch.rand_like(disp))  # Fake target 
        depth_loss = torch.abs(disp - target_disp).mean()
        losses['depth'] = self.loss_weights['depth'] * depth_loss
        
        # Total loss
        losses['total'] = losses['depth'] + losses['smoothness']
        
        return losses
    
    def save_model(self, filename):
        """Save the model"""
        save_path = os.path.join(self.save_dir, filename)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        
    def load_model(self, filename):
        """Load the model"""
        load_path = os.path.join(self.save_dir, filename)
        if os.path.exists(load_path):
            self.model.load_state_dict(torch.load(load_path))
            print(f"Model loaded from {load_path}")
        else:
            print(f"No model found at {load_path}")
            
    @staticmethod
    def visualize_depth(depth, cmap='magma'):
        """Visualize depth prediction"""
        plt.figure(figsize=(10, 5))
        
        # Convert to numpy and normalize
        depth_np = depth.squeeze().cpu().numpy()
        vmax = np.percentile(depth_np, 95)
        plt.imshow(depth_np, cmap=cmap, vmax=vmax)
        plt.colorbar(label='Depth')
        plt.title('Depth Prediction')
        plt.axis('off')
        
        return plt
        
# Example usage
if __name__ == "__main__":
    config = {
        'data_path': './mars_data',
        'batch_size': 8,
        'learning_rate': 1e-4,
        'height': 384,
        'width': 512,
        'save_dir': './trained_models',
        'source_type': 'mixed'  # 'rover', 'satellite', or 'mixed'
    }
    
    adapter = MarsDomainAdapter(config)
    adapter.finetune(num_epochs=20)
