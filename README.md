# Neural Monte Carlo (NeuMC)

This repository provides the `neumc` Python package for building, training, and sampling neural network models for
two-dimensional lattice field theories (LFT), including the phi^4 scalar field, U(1) gauge, and Schwinger models. It
also includes auxiliary scripts and notebooks demonstrating its usage.

## Installation

To use Python modules from this repository, you should install them as the package `neumc`. You will need Python >= 3.11
and pip >= 21.3. As always, it is best to create a separate virtual Python environment for the project.

After cloning the repository, navigate to its root directory and run:

```shell
pip install -e neumc/
```

The `-e` option installs the package in
[_developer_ mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html), allowing you to make changes
to the code inside the package and see them immediately without reinstalling.

Then, run:

```shell
pip install -r neumc/requirements.txt
```

to install required dependencies (currently `numpy` and `scipy`).

#### PyTorch

You will also need to install PyTorch. Installation instructions depend on your operating system and hardware, so please
refer to the official [PyTorch website](https://pytorch.org/get-started/locally/).

### Notebooks

If you would like to use notebooks provided in the repository, you will also need to install `jupyterlab`, `jupytext`
and `matplotlib`.

You can also use other environments such as Visual Studio Code or PyCharm to view and run notebooks. However, since the
notebooks in this repository are stored as R Markdown (.Rmd) files, you will
need [`jupytext`](https://jupytext.readthedocs.io/en/latest/) to convert them. You can do this via the command line:

```shell
jupytext some_notebook.Rmd --to notebook 
```

This will create a `some_notebook.ipynb` file. In JupyterLab, this conversion happens automatically if `jupytext.toml`
is present in your directory (which is included in this repository). You can then simply click on the file and select
Open With > Notebook from the menu. For more information, refer to the [Jupytext documentation](https://jupytext.readthedocs.io/en/latest/).

### Scripts

The scripts directory contains scripts that demonstrate how to use the package to train and sample models. For example,
to train the phi^4 model, run:

```shell
cd scripts
python ./phi4.py 
```

This will start training the phi^4 scalar field model on a 8x8 lattice with m^2 == 1.25 and lambda == 0.0 (free field)
using the reparameterization trick (RT) gradient estimator. If a CUDA-compatible GPU is available, the script will use
it. The training typically takes less than a minute on a GPU and several minutes on a CPU.

For details on other scripts, refer to the [scripts/README.md file](scripts/README.md).

