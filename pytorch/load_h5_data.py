import h5py
import numpy as np
import os

def load_h5_simulation_data(h5_filepath='../data/simulation_data.h5'):
    """
    Load HDF5 simulation data
    
    Parameters:
    -----------
    h5_filepath : str
        Path to the HDF5 file (default: '../data/simulation_data.h5')
        
    Returns:
    --------
    data_dict : dict
        Dictionary containing all datasets from the HDF5 file
    """
    if not os.path.exists(h5_filepath):
        raise FileNotFoundError(f"HDF5 file not found: {h5_filepath}")
    
    data_dict = {}
    
    with h5py.File(h5_filepath, 'r') as f:
        # Print file structure
        print("HDF5 File Structure:")
        print("-" * 50)
        
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name}, Shape: {obj.shape}, dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  Group: {name}")
        
        f.visititems(print_structure)
        print("-" * 50)
        
        # Load all datasets
        for key in f.keys():
            data_dict[key] = f[key][:]
            print(f"\nLoaded '{key}': shape {data_dict[key].shape}")
    
    return data_dict


def get_dataset_info(h5_filepath='../data/simulation_data.h5'):
    """
    Print detailed information about all datasets in the HDF5 file
    
    Parameters:
    -----------
    h5_filepath : str
        Path to the HDF5 file
    """
    if not os.path.exists(h5_filepath):
        raise FileNotFoundError(f"HDF5 file not found: {h5_filepath}")
    
    with h5py.File(h5_filepath, 'r') as f:
        print("Detailed Dataset Information:")
        print("=" * 60)
        
        for key in f.keys():
            dataset = f[key]
            print(f"\nDataset: {key}")
            print(f"  Shape: {dataset.shape}")
            print(f"  dtype: {dataset.dtype}")
            print(f"  Size: {dataset.size} elements")
            print(f"  Memory usage: {dataset.nbytes / (1024*1024):.2f} MB")
            if dataset.size <= 10:
                print(f"  Data: {dataset[:]}")
            else:
                print(f"  Sample (first 3 elements): {np.array(dataset).flat[:3]}")


# Example usage
if __name__ == "__main__":
    try:
        # Load data
        print("Loading HDF5 data...")
        data = load_h5_simulation_data()
        
        # Get detailed info
        print("\n")
        get_dataset_info()
        
        # Access specific datasets
        print("\n" + "=" * 60)
        print("Accessing specific datasets:")
        print(f"\nz shape: {data['z'].shape}")
        print(f"toaPos shape: {data['toaPos'].shape}")
        print(f"R shape: {data['R'].shape}")
        print(f"Q shape: {data['Q'].shape}")
        print(f"P0 shape: {data['P0'].shape}")
        print(f"processNoise shape: {data['processNoise'].shape}")
        print(f"toaNoise shape: {data['toaNoise'].shape}")
        print(f"processbias shape: {data['processbias'].shape}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure to run preSimulateH5() in MATLAB first to generate the HDF5 file.")
