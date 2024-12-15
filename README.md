# README

## Setting Up Development Environment

This README will walk you through setting up a development environment to use the `multivi_tools_splicing` repository. It covers creating a project directory, cloning the necessary repositories, setting up the Python environment, and providing an overview of the included notebooks.

### 1. Create a Project Directory

Start by creating a directory to house the project. You can name it something like `multivi_splicing_project`:

```bash
mkdir multivi_splicing_project
cd multivi_splicing_project
```

### 2. Clone the Required Repositories

#### Clone my forked `scvi-tools` repository, called `scvi-tools-splicing`

Clone the `scvi-tools-splicing` repository:

```bash
git clone https://github.com/smritivaidyanathan/scvi-tools-splicing.git
```

Navigate into the cloned repository:

```bash
cd scvi-tools
```

Add the main repository as a remote to keep up with upstream changes:

```bash
git remote add upstream https://github.com/scverse/scvi-tools.git
```

#### Clone the `multivi_tools_splicing` Repository

Go back to your project directory and clone the `multivi_tools_splicing` repository:

```bash
cd ..
git clone https://github.com/smritivaidyanathan/multivi_tools_splicing.git
```

### 3. Set Up the Python Environment

Create a virtual environment using `conda` with Python 3.12 (you can use any Python version between 3.9 and 3.12):

```bash
conda create -n scvi-env python=3.12
```

Activate the environment:

```bash
conda activate scvi-env
```

Install the development dependencies and the package in editable mode:

```bash
cd scvi-tools
pip install -e ".[dev]"
```

### 4. Confirm the Installation

(Optional) To confirm that the installation was successful and see where the forked `scvi-tools` is located, run:

```bash
pip show scvi-tools
```

This command will display information about the `scvi-tools` package, including its location in your environment. Ensure that this location matches the directory where you cloned `scvi-tools-splicing`, which should align with the location of `multivi_tools_splicing`.

## Overview of the Notebooks

This repository contains several notebooks designed to process input AnnData objects, create and train a MultiVI-Splice model, and perform data analysis. Each notebook includes more detailed explanations about the purpose of cells(Note: the pre-processed data file is very large, so I couldn't upload it, but the original Tabula Muris Senis Dataset can be found here:https://tabula-muris-senis.sf.czbiohub.org/). Here's how to use the notebooks:

### 1. `ann_data_maker.ipynb`

- **Purpose:** Load AnnData objects for ATAC-seq and gene expression data.
- **Instructions:** You'll need to change the file paths in the notebook to point to your local data files.
- **Outcome:** Running this notebook will create and save a combined AnnData object that merges your processed ATSE and GE annDatas.

### 2. `multi_vi_splice_notebook.ipynb`

- **Purpose:** Load the combined AnnData object and train the MultiVI-Splice model.
- **Instructions:** Run the cells to train the model, making sure you change file paths accordingly.
- **Outcome:** The notebook saves the trained model and intermediate files useful for data analysis, including an updated AnnData object with additional information.

### 3. `data_analysis.ipynb`

- **Purpose:** Perform data analysis on the outputs of the MultiVI-Splice model.
- **Instructions:** Load the updated AnnData object from the previous step.
- **Outcome:** The notebook generates figures and visualizations to help interpret your data.

### 4. `splicing_VAE.ipynb`

- **Purpose:** Run the SplicingVAE model independently.
- **Instructions:** Load your ATSE AnnData object into the notebook.
- **Outcome:** Train and evaluate the SplicingVAE model on your data.
