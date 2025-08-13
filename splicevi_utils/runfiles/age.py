#!/usr/bin/env python
"""
Simple script to load a MuData file and print the `age` column
from the splicing modality.
"""
import mudata as mu

# Path to the MuData file (update this path if needed)
INPUT_PATH = (
    "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/"
    "MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/072025/"
    "test_30_70_ge_splice_combined_20250730_164104.h5mu"
)

def main():
    # Load the MuData object
    mdata = mu.read_h5mu(INPUT_PATH)
    # Extract the splicing AnnData
    ad = mdata["splicing"]
    # Print the age column for each cell
    print(ad)

if __name__ == "__main__":
    main()
