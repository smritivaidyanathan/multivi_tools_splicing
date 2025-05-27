#!/usr/bin/env python3
import os, re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Increase all font sizes significantly
plt.rcParams.update({
    "font.size":        18,    
    "axes.titlesize":   18,    
    "axes.labelsize":   18,    
    "xtick.labelsize":  18,    
    "ytick.labelsize":  18,    
    "legend.fontsize":  18,    
    "figure.titlesize": 18,    
})

def plot_imputation_dropoff(csv_path, models_to_plot, outdir):
    df = pd.read_csv(csv_path)
    df["model"] = df["model"].str.strip()

    # coerce metrics
    for c in ["rna_spearman","rna_median_l1","rna_mse",
              "spl_spearman","spl_median_l1","spl_mse"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # single missing fraction
    df["pct_missing"] = df[["pct_rna","pct_splice"]].max(axis=1)

    # filter
    wanted = [m.strip() for m in models_to_plot]
    df = df[df["model"].isin(wanted)]

    # parse Z and dist
    z_vals = sorted({int(re.search(r"Z\s*=\s*(\d+)", m).group(1))
                     for m in wanted if "Z=" in m})
    dists = ["Binomial","Beta-Binomial"]
    cmap = plt.get_cmap("tab10")
    color_map = {z: cmap(i) for i,z in enumerate(z_vals)}
    ls_map = {"Binomial":"-","Beta-Binomial":"--"}

    # NEW LAYOUT: 3 rows (metrics) x 3 columns (modalities)
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True, squeeze=False)

    # Define modalities (columns)
    modality_masks = [
        ("Gene Expression", (df["pct_rna"]>0)&(df["pct_splice"]==0)),
        ("Splicing", (df["pct_splice"]>0)&(df["pct_rna"]==0)),
        ("Joint", (df["pct_rna"]>0)&(df["pct_splice"]>0)),
    ]
    
    # Define metrics (rows)
    metrics = [
        ("Spearman ρ", "rna_spearman", "spl_spearman"),
        ("L₁ error", "rna_median_l1", "spl_median_l1"), 
        ("RMSE", "rna_mse", "spl_mse")
    ]

    for row, (metric_name, rna_col, spl_col) in enumerate(metrics):
        for col, (mod_name, mask) in enumerate(modality_masks):
            ax = axes[row, col]
            sub = df[mask]
            
            for m in wanted:
                mdf = sub[sub["model"]==m]
                if mdf.empty: continue
                
                z = int(re.search(r"Z\s*=\s*(\d+)", m).group(1))
                dist = "Beta-Binomial" if "Beta" in m else "Binomial"
                x = mdf["pct_missing"] * 100
                
                # Choose the appropriate metric based on modality
                if col == 0:  # Gene Expression
                    y = mdf[rna_col]
                elif col == 1:  # Splicing
                    y = mdf[spl_col]  
                else:  # Joint - average both modalities
                    y = 0.5 * (mdf[rna_col] + mdf[spl_col])
                
                ax.plot(x, y,
                        color=color_map[z],
                        linestyle=ls_map[dist],
                        marker="o",
                        markersize=6,
                        linewidth=2)
            
            # Labels
            if row == 0:  # Top row gets column titles
                ax.set_title(mod_name)
            if col == 0:  # Left column gets row labels
                ax.set_ylabel(metric_name)
            if row == 2:  # Bottom row gets x-axis labels
                ax.set_xlabel("Percent missing")

    # Make legend handles with larger markers and lines
    z_handles = [
        mlines.Line2D([],[], color=color_map[z], marker="o", linestyle="-", 
                      label=f"Z={z}", markersize=8, linewidth=2)
        for z in z_vals
    ]
    dist_handles = [
        mlines.Line2D([],[], color="black", linestyle=ls_map[d], 
                      linewidth=3, label=d)
        for d in dists
    ]

    # Reserve right 20% for legends
    plt.tight_layout(rect=[0, 0, 0.78, 1.0])

    # Two figure legends in that margin
    fig.legend(
        handles=z_handles, title="Latent dim (Z)",
        loc="upper right", bbox_to_anchor=(0.95, 0.90),
        borderaxespad=0
    )
    fig.legend(
        handles=dist_handles, title="Distribution",
        loc="center right", bbox_to_anchor=(0.95, 0.50),
        borderaxespad=0
    )

    # Save as PDF and PNG
    pdf_out = os.path.join(outdir, "imputation_dropoff.pdf")
    png_out = os.path.join(outdir, "imputation_dropoff.png")
    
    fig.savefig(pdf_out, dpi=300, bbox_inches="tight", format='pdf')
    fig.savefig(png_out, dpi=300, bbox_inches="tight", format='png')
    
    plt.close(fig)
    print("Saved PDF to", pdf_out)
    print("Saved PNG to", png_out)


if __name__ == "__main__":
    # Update these paths to match your file locations
    CSV  = "/gpfs/commons/home/kisaev/multivi_tools_splicing/results/imputation/batch_20250524_144011/consolidated_imputation_results.csv"
    OUTD = "/gpfs/commons/home/kisaev/multivi_tools_splicing/results/imputation/batch_20250524_144011/figures"  # Output directory for plots
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTD, exist_ok=True)
    
    # All models in your dataset
    models_to_plot = [
        "Splice-VI(Binomial Z=20)",
        "Splice-VI(Binomial Z=40)", 
        "Splice-VI(Beta-Binomial Z=20)",
        "Splice-VI(Beta-Binomial Z=40)",
    ]
    
    plot_imputation_dropoff(CSV, models_to_plot, OUTD)