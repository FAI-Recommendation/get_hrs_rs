import scipy.sparse as sp
import numpy as np

path = r"E:\DoCode\CD2\source\Source\get_hrs_rs\hr\Data\vcr\s_social_adj_mat.npz"

mat = sp.load_npz(path)

print(f"Shape      : {mat.shape}")
print(f"dtype      : {mat.dtype}")
print(f"nnz (non-zero elements): {mat.nnz}")
print(f"Density    : {mat.nnz / (mat.shape[0] * mat.shape[1]):.8f}")

dense = mat.toarray()
print(f"\nMin value  : {dense.min()}")
print(f"Max value  : {dense.max()}")
print(f"Sum        : {dense.sum()}")

print(f"\nFirst 5x5 block:\n{dense[:5, :5]}")
