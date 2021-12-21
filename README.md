# OCDbaseline4modeling
Structural and Functional Connectome analysis of OCD baseline data in view of computational brain modeling.

dependencies: pybct, h5py, nibabel, nilearn, pandas, scipy, sklearn, statsmodel.

## Code structure
The analysis contains structural, functional and structure-function analysis, with their relation to behavioral measures (mainly Y-BOCS score) when possible.

**qsiprep_analysis.py:** Analysis of structural connectivity outputs from QSIPrep. Includes several functions that are used in other modules.

**fc_analysis.py:** Analysis of functional outputs from FMRIPrep and postprocessing from Luke.

**scfc_analysis.py:** Analysis of structure-function coupling and relation to Y-BOCS score.

**GSP_analysis.py**: Graph Signal Processing analysis of SC-FC coupling, interesting but on hold.

**atlas_extraction.py:** Create FC matrices from preprocessed BOLD signals and brain atlases.

**atlaser.py:** Utility module for handling atlases, ROI indexing, create subatlases, etc.

Each *_analysis.py script can be run individually with a set of default parameters.
