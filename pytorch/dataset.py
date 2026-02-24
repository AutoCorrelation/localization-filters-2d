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
        
        # Load data once
        with h5py.File(h5_filepath, 'r') as f:
            self.z = torch.from_numpy(f['z'][:]).float()
            self.toaPos = torch.from_numpy(f['toaPos'][:]).float()
            self.R = torch.from_numpy(f['R'][:]).float()
            self.Q = torch.from_numpy(f['Q'][:]).float()
            self.P0 = torch.from_numpy(f['P0'][:]).float()
            self.processNoise = torch.from_numpy(f['processNoise'][:]).float()
            self.toaNoise = torch.from_numpy(f['toaNoise'][:]).float()
            self.processbias = torch.from_numpy(f['processbias'][:]).float()
        
        # numIterations x numPoints x noiseVariances
        self.num_iterations, self.num_points, self.num_noise_levels = \
            self.z.shape[1], self.z.shape[2], self.z.shape[3]
    
    def __len__(self):
        """Total number of samples"""
        return self.num_iterations * self.num_points * self.num_noise_levels
    
    def __getitem__(self, idx):
        """
        Returns a single sample
        
        Returns:
        --------
        dict with keys: 'z', 'toaPos', 'R', 'Q', 'P0', 'processNoise', 'toaNoise', 'processbias'
        """
        # Convert linear index to 3D indices
        noise_idx = idx % self.num_noise_levels
        point_idx = (idx // self.num_noise_levels) % self.num_points
        iter_idx = idx // (self.num_noise_levels * self.num_points)
        
        return {
            'z': self.z[:, iter_idx, point_idx, noise_idx],
            'toaPos': self.toaPos[:, iter_idx, point_idx, noise_idx],
            'R': self.R[:, :, iter_idx, point_idx, noise_idx],
            'Q': self.Q[:, :, noise_idx],
            'P0': self.P0[:, :, noise_idx],
            'processNoise': self.processNoise[:, iter_idx, noise_idx],
            'toaNoise': self.toaNoise[:, iter_idx, noise_idx],
            'processbias': self.processbias[:, noise_idx],
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
        
        # Load all data
        with h5py.File(h5_filepath, 'r') as f:
            self.z = f['z'][:]
            self.toaPos = f['toaPos'][:]
            self.R = f['R'][:]
            self.Q = f['Q'][:]
            self.P0 = f['P0'][:]
            self.processNoise = f['processNoise'][:]
            self.toaNoise = f['toaNoise'][:]
            self.processbias = f['processbias'][:]
        
        self.num_noise_levels = self.z.shape[3]
        self.num_iterations = self.z.shape[1]
        self.num_points = self.z.shape[2]
    
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
            'z': torch.from_numpy(self.z[:, :, :, noise_idx]).float(),
            'toaPos': torch.from_numpy(self.toaPos[:, :, :, noise_idx]).float(),
            'R': torch.from_numpy(self.R[:, :, :, :, noise_idx]).float(),
            'Q': torch.from_numpy(self.Q[:, :, noise_idx]).float(),
            'P0': torch.from_numpy(self.P0[:, :, noise_idx]).float(),
            'processNoise': torch.from_numpy(self.processNoise[:, :, noise_idx]).float(),
            'toaNoise': torch.from_numpy(self.toaNoise[:, :, noise_idx]).float(),
            'processbias': torch.from_numpy(self.processbias[:, noise_idx]).float(),
        }
    
    def get_all_data(self):
        """
        Get all data as tensors
        
        Returns:
        --------
        dict with all datasets converted to PyTorch tensors
        """
        return {
            'z': torch.from_numpy(self.z).float(),
            'toaPos': torch.from_numpy(self.toaPos).float(),
            'R': torch.from_numpy(self.R).float(),
            'Q': torch.from_numpy(self.Q).float(),
            'P0': torch.from_numpy(self.P0).float(),
            'processNoise': torch.from_numpy(self.processNoise).float(),
            'toaNoise': torch.from_numpy(self.toaNoise).float(),
            'processbias': torch.from_numpy(self.processbias).float(),
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
        print(f"  z: {self.z.shape}")
        print(f"  toaPos: {self.toaPos.shape}")
        print(f"  R: {self.R.shape}")
        print(f"  Q: {self.Q.shape}")
        print(f"  P0: {self.P0.shape}")
        print(f"  processNoise: {self.processNoise.shape}")
        print(f"  toaNoise: {self.toaNoise.shape}")
        print(f"  processbias: {self.processbias.shape}")


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
