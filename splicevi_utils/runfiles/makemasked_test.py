#!/usr/bin/env python
"""
Pre-mask splicing modality in a test MuData by masking out a fraction of ATSEs.
Stores original junc_ratio values for masked ATSEs in a new layer `junc_ratio_masked_original`.
Prints final MuData summary to confirm, including counts of masked ATSEs and junctions,
and a representative example showing that an ATSE was originally observed, is now masked,
and its original junction values are stored.
"""
import numpy as np
import mudata as mu
from scipy import sparse

# Configuration
INPUT_PATH = (
    "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/"
    "MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/072025/"
    "test_30_70_ge_splice_combined_20250730_164104.h5mu"
)
MASK_FRACTION = 0.2  # fraction of observed ATSEs to mask per cell
OUTPUT_PATH = (
    f"/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/"
    "MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/072025/"
    f"MASKED_{MASK_FRACTION}_test_30_70_ge_splice_combined_20250730_164104.h5mu"
)
SEED = 0

def main():
    # 1) Load MuData and extract the splicing AnnData
    print(f"Loading MuData from {INPUT_PATH}")
    mdata = mu.read_h5mu(INPUT_PATH)
    ad = mdata["splicing"]
    n_obs, n_junc = ad.n_obs, ad.n_vars
    print(f"Splicing AnnData: {n_obs} cells x {n_junc} junctions\n")

    # 2) Pull out the four layers we will modify
    jr_layer  = ad.layers["junc_ratio"]
    cj_layer  = ad.layers["cell_by_junction_matrix"]
    cc_layer  = ad.layers["cell_by_cluster_matrix"]
    psi_layer = ad.layers["psi_mask"]

    # Convert to arrays for easy indexing
    jr_arr       = jr_layer.toarray()  if sparse.issparse(jr_layer)  else jr_layer.copy()
    cj_arr       = cj_layer.toarray()  if sparse.issparse(cj_layer)  else cj_layer.copy()
    cc_arr       = cc_layer.toarray()  if sparse.issparse(cc_layer)  else cc_layer.copy()
    psi_arr      = psi_layer.toarray() if sparse.issparse(psi_layer) else psi_layer.copy()

    # Keep a copy of the original junc_ratio before any masking
    jr_original  = jr_arr.copy()

    # Prepare an array to store the original junc_ratio for masked entries
    orig_jr      = np.zeros_like(jr_arr)

    # Event IDs to map junctions → ATSEs
    event_ids    = ad.var["event_id"].values  # length = n_junc

    rng = np.random.RandomState(SEED)

    # Track unique ATSEs and junctions masked, and record one example
    masked_atse_set   = set()
    masked_junc_set   = set()
    example_record    = None  # (cell_idx, atse_id, junc_idx)

    # 3) Loop over each cell and mask a fraction of its observed ATSEs
    for i in range(n_obs):
        # junctions observed in this cell
        obs_juncs = np.nonzero(psi_arr[i])[0]
        if obs_juncs.size == 0:
            continue

        # ATSE event IDs present
        obs_events = np.unique(event_ids[obs_juncs])
        k = max(1, int(len(obs_events) * MASK_FRACTION))
        mask_events = rng.choice(obs_events, size=k, replace=False)

        # record into global set
        masked_atse_set.update(mask_events)

        # per-junction mask for this cell
        junc_mask = np.isin(event_ids, mask_events)
        # record unique junction indices
        masked_junc_set.update(np.where(junc_mask)[0].tolist())

        # record one example if not yet set
        if example_record is None and mask_events.size > 0:
            atse_ex = mask_events[0]
            junc_indices = np.where(event_ids == atse_ex)[0]
            if junc_indices.size > 0:
                junc_ex = junc_indices[0]
                example_record = (i, atse_ex, junc_ex)

        # store original and zero out layers
        orig_jr[i, junc_mask]   = jr_arr[i, junc_mask]
        jr_arr [i, junc_mask]   = 0
        cj_arr [i, junc_mask]   = 0
        cc_arr [i, junc_mask]   = 0
        psi_arr[i, junc_mask]   = False

    # 4) Write masked data back into the AnnData layers
    ad.layers["junc_ratio_masked_original"]    = sparse.csr_matrix(orig_jr)
    ad.layers["junc_ratio"]                    = sparse.csr_matrix(jr_arr)
    ad.layers["cell_by_junction_matrix"]       = sparse.csr_matrix(cj_arr)
    ad.layers["cell_by_cluster_matrix"]        = sparse.csr_matrix(cc_arr)
    ad.layers["psi_mask"]                      = sparse.csr_matrix(psi_arr)

    # 5) Print summary of masking
    print(f"Total unique ATSEs masked: {len(masked_atse_set)}")
    print(f"Total unique junctions masked: {len(masked_junc_set)}")

    if example_record:
        cell_i, atse_id, junc_i = example_record
        print("\nRepresentative example:")
        print(f"  Cell index:        {cell_i}")
        print(f"  ATSE event ID:     {atse_id}")
        print(f"  Junction index:    {junc_i}")
        print(f"  Original junc_ratio at that position: "
              f"{jr_original[cell_i, junc_i]}")
        print(f"  After masking, junc_ratio:              "
              f"{jr_arr[cell_i, junc_i]}")
        print(f"  Stored in 'junc_ratio_masked_original': "
              f"{orig_jr[cell_i, junc_i]}")

    # 6) Save the new MuData file
    print(f"\nWriting masked MuData to {OUTPUT_PATH}")
    mdata.write_h5mu(OUTPUT_PATH)
    print("Write complete.\n")

    # 7) Final MuData summary
    print("=== Final MuData Summary ===")
    print(mdata)
    print("Modalities:", list(mdata.mod.keys()))
    sp_mod = mdata["splicing"]
    print(f"Splicing layers after masking: {list(sp_mod.layers.keys())}")
    print(f"Splicing shape: {sp_mod.n_obs} cells x {sp_mod.n_vars} junctions")

if __name__ == "__main__":
    main()
