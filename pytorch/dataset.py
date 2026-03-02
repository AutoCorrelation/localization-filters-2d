import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os


class SimulationDataset(Dataset):
    """
    PyTorch Dataset for loading simulation data from HDF5 file
    """
    def __init__(self, h5_filepath='../data/simulation_data.h5'):
        """
        Parameters:
        -----------
        h5_filepath : str
            Path to the HDF5 file
        """
        self.h5_filepath = h5_filepath
        
        if not os.path.exists(h5_filepath):
            raise FileNotFoundError(f"HDF5 file not found: {h5_filepath}")
        
        # Load only toaPos and realPos from HDF5
        with h5py.File(h5_filepath, 'r') as f:
            self.toaPos = torch.from_numpy(f['toaPos'][:]).float()
            self.realPos = torch.from_numpy(f['realPos'][:]).float()

        # Determine dimensions from toaPos: coords x iterations x points x noise_levels
        self.num_iterations = self.toaPos.shape[1]
        self.num_points = self.toaPos.shape[2]
        self.num_noise_levels = self.toaPos.shape[3]
    
    def __len__(self):
        """Total number of samples"""
        return self.num_iterations * self.num_points * self.num_noise_levels
    
    def __getitem__(self, idx):
        """
        Returns a single sample
        
        Returns:
        --------
        dict with keys: 'toaPos', 'realPos' and index info
        """
        # Convert linear index to 3D indices
        noise_idx = idx % self.num_noise_levels
        point_idx = (idx // self.num_noise_levels) % self.num_points
        iter_idx = idx // (self.num_noise_levels * self.num_points)
        
        return {
            'toaPos': self.toaPos[:, iter_idx, point_idx, noise_idx],
            'realPos': self.realPos[:, iter_idx, point_idx, noise_idx],
            'iter_idx': iter_idx,
            'point_idx': point_idx,
            'noise_idx': noise_idx
        }


class SimulationDatasets:
    """
    Container for all simulation datasets split by noise levels
    """
    def __init__(self, h5_filepath='../data/simulation_data.h5'):
        """
        Parameters:
        -----------
        h5_filepath : str
            Path to the HDF5 file
        """
        self.h5_filepath = h5_filepath
        
        if not os.path.exists(h5_filepath):
            raise FileNotFoundError(f"HDF5 file not found: {h5_filepath}")
        
        # Load only toaPos and realPos
        with h5py.File(h5_filepath, 'r') as f:
            self.toaPos = f['toaPos'][:]
            self.realPos = f['realPos'][:]

        # Determine sizes from toaPos
        self.num_noise_levels = self.toaPos.shape[3]
        self.num_iterations = self.toaPos.shape[1]
        self.num_points = self.toaPos.shape[2]
    
    def get_dataset_for_noise_level(self, noise_idx):
        """
        Get data for a specific noise level
        
        Parameters:
        -----------
        noise_idx : int
            Index of noise level (0-4)
            
        Returns:
        --------
        dict with all data for the specified noise level
        """
        return {
            'toaPos': torch.from_numpy(self.toaPos[:, :, :, noise_idx]).float(),
            'realPos': torch.from_numpy(self.realPos[:, :, :, noise_idx]).float(),
        }
    
    def get_all_data(self):
        """
        Get all data as tensors
        
        Returns:
        --------
        dict with all datasets converted to PyTorch tensors
        """
        return {
            'toaPos': torch.from_numpy(self.toaPos).float(),
            'realPos': torch.from_numpy(self.realPos).float(),
        }
    
    def printinfo(self):
        """Print dataset information"""
        print("Simulation Dataset Information:")
        print("=" * 60)
        print(f"Noise levels: {self.num_noise_levels}")
        print(f"Iterations: {self.num_iterations}")
        print(f"Points: {self.num_points}")
        print(f"Total samples: {self.num_noise_levels * self.num_iterations * self.num_points}")
        print("\nDataset shapes:")
        print(f"  toaPos: {self.toaPos.shape}")
        print(f"  realPos: {self.realPos.shape}")


# Example usage
if __name__ == "__main__":
    print("PyTorch Dataset Example")
    print("=" * 60)
    
    try:
        # Method 1: Using the simple Dataset class with DataLoader
        print("\n[Method 1] Using SimulationDataset with DataLoader")
        print("-" * 60)
        dataset = SimulationDataset()
        print(f"Dataset size: {len(dataset)}")
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        batch = next(iter(dataloader))
        print(f"Batch keys: {batch.keys()}")
        print(f"Batch z shape: {batch['z'].shape}")
        print(f"Batch toaPos shape: {batch['toaPos'].shape}")
        
        # Method 2: Using SimulationDatasets for better control
        print("\n[Method 2] Using SimulationDatasets")
        print("-" * 60)
        datasets = SimulationDatasets()
        datasets.printinfo()
        
        # Get data for specific noise level
        print("\nGetting data for noise level 0:")
        data_noise_0 = datasets.get_dataset_for_noise_level(0)
        print(f"z shape: {data_noise_0['z'].shape}")
        print(f"toaPos shape: {data_noise_0['toaPos'].shape}")
        
        # Get all data
        print("\nGetting all data:")
        all_data = datasets.get_all_data()
        for key, value in all_data.items():
            print(f"  {key}: {value.shape}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure to run preSimulateH5() in MATLAB first to generate the HDF5 file.")
