This is a modified version of a script written by Venkat [original repo](https://github.com/venkatesh-sivaraman/facs_analysis). I've made some minor changes and the code could still be made clearer in a lot of places but it does work


# FACS Analysis

The Jupyter notebook in this repo, `facs_analysis.ipynb`, walks you through the basic steps of analyzing FACS results. For high-throughput binding experiments, the `highThroughputScripts` pipeline is available on [GitHub](https://github.com/KeatingLab/highThroughputScripts/tree/vs_optimization).

## Suggested Usage

Download the repo as a ZIP file, and extract the archive in the directory where you plan to conduct your analysis. If you plan to conduct multiple analyses, duplicate the `.ipynb` file and name it according to the experiment. The files can be moved around as long as `facs_utils.py` is in the same directory as the notebook.

To complete the analysis, run all cells in order one-by-one unless otherwise specified, filling in or changing input values as needed. To export a plot at higher resolution without changing the font size, you may want to change the DPI in the `plt.figure` command, i.e. `plt.figure(..., dpi=160)`.

## Requirements

This notebook requires the following modules:

* `FlowCytometryTools` (install using `pip install flowcytometrytools`)
* `lmfit`
* `ipywidgets`, version 7.2 or later (upgrade if necessary)
* The script `facs_utils.py` must be in the same directory as this notebook.

After installing these modules, be sure to restart the notebook kernel to make sure the modules are available.

*Written by*: Venkatesh Sivaraman, February 2019, adapted from scripts by Dustin Whitney and Theresa Hwang
