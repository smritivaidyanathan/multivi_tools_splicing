import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import anndata as ad # Import AnnData
from typing import Dict, List, Optional

class AnnDataDataset(Dataset):
    """
    PyTorch Dataset for loading data from specified layers of an AnnData object,
    handling NaNs in the input layer by creating a mask.

    Args:
        adata (ad.AnnData): The AnnData object containing the data.
        x_layer (str): The key for the layer in `adata.layers` to be used as input features (`x`).
                       Expected to contain NaNs for missing values.
        junction_counts_layer (str): The key for the layer containing junction counts
                                     (successes for loss function).
        cluster_counts_layer (str): The key for the layer containing cluster counts
                                    (total trials for loss function).
        obs_indices (Optional[List[int]]): Optional list of observation indices to use
                                           (e.g., for train/validation split). If None, uses all observations.
    """
    def __init__(self,
                 adata: ad.AnnData,
                 x_layer: str = 'junc_ratio',
                 junction_counts_layer: str = 'cell_by_junction_matrix',
                 cluster_counts_layer: str = 'cell_by_cluster_matrix',
                 obs_indices: Optional[List[int]] = None):

        super().__init__()
        self.adata = adata
        self.x_layer = x_layer
        self.junction_counts_layer = junction_counts_layer
        self.cluster_counts_layer = cluster_counts_layer

        # --- Handle subsetting ---
        if obs_indices is None:
            self.indices = np.arange(self.adata.n_obs)
        else:
            # Ensure indices are valid
            if not all(0 <= i < self.adata.n_obs for i in obs_indices):
                 raise IndexError("Provided obs_indices are out of bounds for AnnData object.")
            self.indices = np.array(obs_indices)

        self.n_obs = len(self.indices)

    def __len__(self) -> int:
        """Returns the number of observations in the dataset (or subset)."""
        return self.n_obs

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Fetches data for a single observation (cell).

        Args:
            idx (int): The index within the subset of observations defined during initialization.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                'x': Input features tensor (from x_layer, NaNs replaced with 0.0). Shape (n_vars,).
                'mask': Binary mask tensor (1 where ATSE counts exist, 0 where they don't). Shape (n_vars,).
                'junction_counts': Junction counts tensor. Shape (n_vars,).
                'cluster_counts': Cluster counts tensor. Shape (n_vars,).
        """
        # Map the dataset index `idx` to the original AnnData index
        adata_idx = self.indices[idx]

        # --- Extract data from AnnData layers ---
        # Note: Accessing layers by index might return NumPy arrays or sparse matrices.
        # We convert to dense NumPy arrays before creating tensors.
        # Use .copy() to avoid potential issues with views if manipulating data later.

        # Input features (potentially with NaNs)
        x_data = self.adata.layers[self.x_layer][adata_idx]
        if hasattr(x_data, "toarray"): # Handle sparse matrix case
            x_data = x_data.toarray().flatten().copy()
        else:
            x_data = np.array(x_data).flatten().copy() # Ensure it's a NumPy array

        # Counts for loss function
        junction_counts = self.adata.layers[self.junction_counts_layer][adata_idx]
        if hasattr(junction_counts, "toarray"):
            junction_counts = junction_counts.toarray().flatten().copy()
        else:
             junction_counts = np.array(junction_counts).flatten().copy()

        cluster_counts = self.adata.layers[self.cluster_counts_layer][adata_idx]
        if hasattr(cluster_counts, "toarray"):
            cluster_counts = cluster_counts.toarray().flatten().copy()
        else:
             cluster_counts = np.array(cluster_counts).flatten().copy()

        # --- Create the mask based on ATSE counts existence ---
        # Mask is True where ATSE counts exist (not zero or missing)
        # This ensures we only include junctions where ATSE counts are available
        mask_np = (cluster_counts > 0)  # ATSE counts must be positive to be considered valid

        # --- Handle NaNs in x_data for model input ---
        x_data_clean = np.nan_to_num(x_data, nan=0.0)

        # --- Convert to PyTorch Tensors ---
        x_tensor = torch.from_numpy(x_data_clean).float()
        # Create a BOOLEAN tensor for the mask
        mask_tensor = torch.from_numpy(mask_np).bool()
        junction_counts_tensor = torch.from_numpy(junction_counts).float()
        cluster_counts_tensor = torch.from_numpy(cluster_counts).float()

        return {
            'x': x_tensor,
            'mask': mask_tensor, # Now returns a boolean tensor based on ATSE count existence
            'junction_counts': junction_counts_tensor,
            'cluster_counts': cluster_counts_tensor,
        }