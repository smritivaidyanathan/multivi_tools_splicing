#!/usr/bin/env python3
"""
consolidate_results.py

Merge results from parallel imputation jobs
"""

import os
import pandas as pd
import sys
import glob

def consolidate_results(batch_dir):
    """Consolidate all CSV results from parallel jobs"""
    
    print(f"→ Consolidating results from: {batch_dir}")
    
    # Find all result CSVs
    pattern = os.path.join(batch_dir, "*/imputation_results.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print("No CSV files found!")
        return
    
    print(f"→ Found {len(csv_files)} result files")
    
    # Read and combine all results
    all_results = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_results.append(df)
            print(f"  ✓ {csv_file}: {len(df)} rows")
        except Exception as e:
            print(f"  ✗ {csv_file}: {e}")
    
    if all_results:
        # Combine all dataframes
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Sort by conditions and models
        combined_df = combined_df.sort_values(['pct_rna', 'pct_splice', 'model'])
        
        # Save consolidated results
        output_file = os.path.join(batch_dir, "consolidated_imputation_results.csv")
        combined_df.to_csv(output_file, index=False)
        
        print(f"→ Consolidated results saved to: {output_file}")
        print(f"→ Total rows: {len(combined_df)}")
        print(f"→ Conditions: {combined_df[['pct_rna', 'pct_splice']].drop_duplicates().shape[0]}")
        print(f"→ Models: {combined_df['model'].nunique()}")
        
        # Print summary
        print("\n→ Summary by condition:")
        summary = combined_df.groupby(['pct_rna', 'pct_splice'])['model'].count()
        print(summary)
        
        return output_file
    else:
        print("No valid results found to consolidate")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python consolidate_results.py <batch_directory>")
        sys.exit(1)
    
    batch_dir = sys.argv[1]
    consolidate_results(batch_dir)