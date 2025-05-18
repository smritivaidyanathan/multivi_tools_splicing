#!/usr/bin/env python3
import os, re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# bump all fonts
plt.rcParams.update({
    "font.size":        14,
    "axes.titlesize":   14,
    "axes.labelsize":   14,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  12,
    "figure.titlesize": 16,
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

    # figure + axes
    fig, axes = plt.subplots(3,3, figsize=(14,9), sharex=True, squeeze=False)

    modality_masks = [
      ("Gene expression only", (df["pct_rna"]>0)&(df["pct_splice"]==0)),
      ("Splicing only",        (df["pct_splice"]>0)&(df["pct_rna"]==0)),
      ("Both modalities",      (df["pct_rna"]>0)&(df["pct_splice"]>0)),
    ]
    rows = [
      [("rna_spearman","Spearman ρ"),("rna_median_l1","L₁ error"),("rna_mse","RMSE")],
      [("spl_spearman","Spearman ρ"),("spl_median_l1","L₁ error"),("spl_mse","RMSE")],
      [("rna_spearman","Spearman ρ"),("rna_median_l1","L₁ error"),("rna_mse","RMSE")],
    ]

    for i, (mod_name, mask) in enumerate(modality_masks):
        sub = df[mask]
        for j, (col, pretty) in enumerate(rows[i]):
            ax = axes[i][j]
            for m in wanted:
                mdf = sub[sub["model"]==m]
                if mdf.empty: continue
                z    = int(re.search(r"Z\s*=\s*(\d+)", m).group(1))
                dist = "Beta-Binomial" if "Beta" in m else "Binomial"
                x = mdf["pct_missing"]*100
                if i<2:
                    y = mdf[col]
                else:
                    spl_col = col.replace("rna_","spl_")
                    y = 0.5*(mdf[col]+mdf[spl_col])
                ax.plot(x, y,
                        color=color_map[z],
                        linestyle=ls_map[dist],
                        marker="o")
            if i==0:    ax.set_title(pretty)
            if j==0:    ax.set_ylabel(mod_name)
            if i==2:    ax.set_xlabel("Percent missing")

    # make legend handles
    z_handles = [
      mlines.Line2D([],[], color=color_map[z], marker="o", linestyle="-", label=f"Z={z}")
      for z in z_vals
    ]
    dist_handles = [
      mlines.Line2D([],[], color="black", linestyle=ls_map[d], linewidth=2, label=d)
      for d in dists
    ]

    # reserve right 15% for legends
    plt.tight_layout(rect=[0,0,0.75,1.0])

    # two figure legends in that margin
    fig.legend(
      handles=z_handles, title="Latent dim (Z)",
      loc="upper right", bbox_to_anchor=(0.90, 0.90),
      borderaxespad=0
    )
    fig.legend(
      handles=dist_handles, title="Distribution",
      loc="center right", bbox_to_anchor=(0.90, 0.50),
      borderaxespad=0
    )

    out = os.path.join(outdir, "imputation_dropoff.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved to", out)


if __name__ == "__main__":
    CSV  = "/gpfs/commons/home/svaidyanathan/imputation_eval_runs/run_20250518_121234/imputation_results_mock.csv"
    OUTD = "/gpfs/commons/home/svaidyanathan/imputation_eval_runs/run_20250518_121234/figures"
    os.makedirs(OUTD, exist_ok=True)
    plot_imputation_dropoff(CSV, [
        "Splice-VI(Binomial Z=20)",
        "Splice-VI(Binomial Z=30)",
        "Splice-VI(Binomial Z=40)",
        "Splice-VI(Beta-Binomial Z=20)",
        "Splice-VI(Beta-Binomial Z=30)",
        "Splice-VI(Beta-Binomial Z=40)",
    ], OUTD)