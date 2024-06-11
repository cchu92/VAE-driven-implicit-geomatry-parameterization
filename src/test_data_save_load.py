import h5py
import numpy as np

# Example data generation: replace these with your actual SDFs and isovalues
sdfs = np.random.rand(5, 1, 20, 20, 20)  # 5 SDFs of shape 20x20x20
isovalues = np.random.rand(5)  # 5 corresponding isovalues

# # Save data
# with h5py.File('sdf_dataset.h5', 'w') as f:
#     f.create_dataset('sdfs', data=sdfs)
#     f.create_dataset('isovalues', data=isovalues)


# Load data
with h5py.File('sdf_dataset.h5', 'r') as f:
    sdfs_loaded = f['sdfs'][:]
    isovalues_loaded = f['isovalues'][:]

# Now sdfs_loaded and isovalues_loaded contain your SDFs and isovalues, respectively
