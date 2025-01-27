# Neural Monte Carlo

The main part of this repository contains the python package `neumc`. It contains all the necessary
ingredients to build, train and sample neural network models for lattice field theories (LTF). 
It implements two-dimensional phi^4 scalar field, U(1) gauge and
Schwinger (U(1) gauge with Wilson fermions) LTF.

The rest of the repository contains the auxiliary files like scripts and notebooks that show typical usages of the
package. Once the installation, described in the next section, is complete, you can start with notebooks in
the `notebooks` directory or one of the
scripts in `scripts` directory.

For example you can run the script that trains the phi^4 model by running

```shell
python ./scripts/phi4.py 
```

This will start training the phi^4 scalar field model on a 8x8 lattice with m^2 == 1.25 and lambda == 0.0 (free field)
using the REINFORCE
estimator. The training will be on 'cuda' device if available. More information can be
found in the [scripts/README.md file](scripts/README.md). The whole
script should take less than a minute on a GPU and several minutes on a CPU.

The rest of the scripts is described in the [scripts/README.md file](scripts/README.md).

## Installation

To use Python modules from this repository, you should install them as the package `neumc`.
You will need `Python>=3.11` and `pip3>= 21.3`. So please upgrade your python installation beforehand.
As always it is best to create a separate virtual Python environment for the project. Probably the easiest way is to use
`venv`, but you may use `conda/mamba` if you are familiar with it. Let's assume that you
will call this environment `neumc-env`.

### Linux/Unix

In your terminal, go to the directory where you want to store this environment and type

```shell
python -m venv neumc-env
```

to create an environment and then activate it by running

```shell
. ./neumc-env/bin/activate
```

Now you can start installing required packages. Go to the root directory of this repository and type

```shell
pip install -e neumc/
```

followed by

```shell
pip install -r neumc/requirements.txt
```

to install required packages (at the moment only `numpy`, `scipy`). The `-e` option installs the package in so-called
[_developer_ mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html). This allows you to make changes to the code inside package that will be immediately visible
without reloading the package.

#### PyTorch

You will also need to install `PyTorch`. I am not giving any instructions for that, as it depends on your operating system and
hardware. You can get instruction directly from [their site](https://pytorch.org/get-started/locally/).

If you would like to use notebooks provided in the repository, you will also need to install `jupyterlab`, `jupytext`
and `matplotlib`. You can do that by running

```shell
pip install notebook jupyterlab jupytext matplotlib
```

Of course, you can use some other environment to view and run notebooks like _e.g._ `Visual Studio Code` or `PyCharm`,
but you will need `jupytext` to transform notebooks that are stored in this repository as R Markdown (.Rmd)
files. That can be done via the command line like this

```shell
jupytext some_notebook.Rmd --to notebook 
```

That will create a `some_notebook.ipynb` file. In jupyter lab this happens automatically when you have
the `jupytext.toml` in you directory (it is provided in this repository). Then it is enough to click on the file name
and choose `Open With>Notebook` option from the menu.

After finishing work you can deactivate the environment by running

```shell
deactivate
```

If something goes very wrong, you can always delete the whole environment by running

```shell
rm -rf  neumc-env
```

### Windows

On Windows you essentially follow up same steps in command prompt or PowerShell, but you activate the environment by
running the `activate` script

```shell
neumc-env\Scripts\activate
```
