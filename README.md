Baseline OCD Neuroimaging
=========================================
Structural and Functional Neuroimaging analysis of OCD baseline data in view of computational brain modeling.

<!-- dependencies: pybct, h5py, nibabel, nilearn, pandas, scipy, sklearn, statsmodel.
     insert badges instead -->

Table of contents
-----------------
* [Installation](#installation)
* [Usage](#usage)
  - [Workflow](#workflow)
    + [Functional analysis](#functional)
    + [Structural analysis](#structural)
    + [Effective connectivity](#effective)
  - [Code structure](#code)
* [Known issues and limitations](#known-issues-and-limitations)
* [Getting help](#getting-help)
* [Contributing](#contributing)
* [License](#license)
* [Authors and history](#authors-and-history)
* [Acknowledgments](#acknowledgments)

Installation
------------
> Tested on Ubuntu 20.04
> Linux-5.8.0

It is strongly advised to install the project into a new virtual environment using python 3.9:

    pyenv install 3.9.7 OCDenv
    pyenv activate OCDenv

Then from the root of this source repository (where the `setup.py` is located), type

    pip install -e .


Usage
-----

We provide an overall walkthrough of the analysis to reproduce the results of the study in the [Workflow](#workflow) section.

A more specific description of each module is presented in [Code structure](#code).

For details about each module, refer to each file separately.

#### Workflow

This project contains 3 "*streams*" of analysis: functional, structural, and effective connectivity analysis.
> _n.b. technically the effective connectivity analysis is also functional_

###### Functional analysis
> The functional analysis assumes that [fMRIPrep](https://github.com/nipreps/fmriprep) has already been run. Before running the following scripts, ensure that the path to the project directory `proj_dir` is correctly set in those scripts and that the output folder `derivatives` has been generated from fMRIPrep with its adequate content.

To perform several preprocessing steps (denoising, filtering, global signal regression, scrubbing, etc.), and the first-level SPM analysis; from the HPC cluster run the following PBS script

    prep_spm_dcm.pbs

This calls `preprocessing/post_fmriprep_denoising.py` with a set of default preprocessing parameters. See this file for more details about the preprocessing pipeline and the [fmripop](https://github.com/brain-modelling-group/fmripop) package.

The second-level SPM analysis is performed by running the following command:

    python seed_to_voxel_analysis.py --min_time_after_scrubbing 2 --plot_figs --run_second_level --brain_smoothing_fwhm 8 --fdr_threshold 0.05 --save_outputs

Here, the arguments indicate to discard subjects with less than 2 minutes of data after scrubbing was performed, use the 8mm spatially smoothed data (need to be preprocessed accordingly above) and to use a FDR corrected p-value threshold of 0.05.


###### Structural analysis

The structural connectivity analysis starts by running the [QSIPrep](https://github.com/PennLINC/qsiprep) pipeline to preprocess DWI and perform tractography.
This is performed in the HPC cluster through the following script:

    qsiprep_parallel_combined.pbs

The parameters used for DWI preprocessing and the tractography algorithm can be found in `preprocessing/qsiprep_recon_file_100M_seeds.json`.

For each subject, this creates 100 millions streamlines and a connectivity matrix following some established atlases. However, we now want to focus on the structural connectivity between the volumes of interests (cluster or seed VOIs) extracted from the functional analysis. This implies creating a new parcellation (or brain atlas) from those VOIs.
> _#TODO:_ explain how to create atlas from VOIs

and generating a connectivity matrix from this new atlas:

    create_connectome_from_atlas.pbs

Then from a local workstation, we can create the track density maps for each individuals

    create_track_density_maps.sh

and finally run the voxel-wise analysis that extract the GFA using specific pathway mask based on track density:

    voxelwise_diffusion_analysis.py --compute_tdi --plot_tdi_distrib --plot_figs


###### Effective connectivity analysis

The effective connectivity analysis uses DCM, which is part of SPM12 software that runs on MATLAB.



#### Code structure

The analysis contains structural, functional and structure-function analysis, with their relation to behavioral measures (mainly Y-BOCS score) when possible.

> Note: each *_analysis.py script can be run individually with a set of default parameters.

**qsiprep_analysis.py:** Analysis of structural connectivity matrices, outputed from QSIPrep. Includes several functions that are used in other modules.

**fc_analysis.py:** Analysis of functional outputs from FMRIPrep and postprocessing from Luke.

**scfc_analysis.py:** Analysis of structure-function coupling and relation to Y-BOCS score.

**GSP_analysis.py**: Graph Signal Processing analysis of SC-FC coupling, interesting but on hold.

**atlas_extraction.py:** Create FC matrices from preprocessed BOLD signals and brain atlases.

**atlaser.py:** Utility module for handling atlases, ROI indexing, create subatlases, etc.
