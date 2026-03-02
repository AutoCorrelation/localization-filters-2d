import h5py
import numpy as np
import os

def load_h5_simulation_data(h5_filepath='data/simulation_data.h5'):
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
        # Print file structure (brief)
        print("HDF5 File Structure (showing datasets):")
        print("-" * 50)
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {key}, Shape: {obj.shape}, dtype: {obj.dtype}")
            else:
                print(f"  Group: {key}")
        print("-" * 50)

        # Load only toaPos and realPos (if present)
        for needed in ('toaPos', 'realPos'):
            if needed in f:
                data_dict[needed] = f[needed][:]
                print(f"Loaded '{needed}': shape {data_dict[needed].shape}")
            else:
                print(f"Warning: '{needed}' not found in HDF5 file")

    return data_dict


def get_dataset_info(h5_filepath='data/simulation_data.h5'):
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
        
        # Access specific datasets (only toaPos and realPos)
        print("\n" + "=" * 60)
        print("Accessing specific datasets (loaded):")
        if 'toaPos' in data:
            print(f"toaPos shape: {data['toaPos'].shape}")
        else:
            print("toaPos: not loaded")
        if 'realPos' in data:
            print(f"realPos shape: {data['realPos'].shape}")
        else:
            print("realPos: not loaded")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure to run preSimulateH5() in MATLAB first to generate the HDF5 file.")
