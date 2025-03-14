# Notebooks

This directory contains Jupyter notebooks that can be used to train and analyze models. Those notebooks are also intended
as an introduction to the application of the  ML models to Lattice Field Theories (LFT).

The notebooks are stored as R Markdown documents (\*.Rmd) and require the `jupytext` package to be installed to convert
them to and from Jupyter notebooks (\*.ipynb).
This is done automatically when the notebooks are run in Jupyter Lab.
To use Jupyter Lab, you will need to install it as it is not the part of the `neumc` package.
You will also need  the `matplotlib` package to plot the results.

Additionally, the U1 notebooks will use some functions from the `live_plot.py` and `mask_plots.py` files in the notebooks directory.
The `live_plot.py` file contains the utilities that can be used to plot the results of the training in real-time taken https://arxiv.org/abs/2101.08176 by M.S. Albergo et all.
The `mask_plots.py` file contains the utilities that display the masking patterns used in the gauge equivariant normalizing flows.
