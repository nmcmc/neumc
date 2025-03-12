# Scripts

This directory contains scripts that can be used to train and analyze models.

## Training scripts

These scripts can be used to train different models. They are all very similar and contain lots of duplicated code that
was intentionally left to allow them to be further developed separately. All the models are defined on a
two-dimensional lattice. By default, all scripts use the 8x8 square lattice.

They are 'plain vanilla' scripts without any command line parameters (except for the `schwinger.py` script). All the
parameters have to be changed inside the scripts.

Two parameters `batch_size` and `n_batches` are crucial for the efficiency.
First,
for the REINFORCE estimator the total number of configurations `n_batches x batch_size` cannot be too small because of
its high initial variance. We recommend using at least 1024.
Second, one should choose the biggest `batch_size` that does not cause the out of memory error on your GPU.

One of the most important parameters is the learning rate `lr`. For the Schwinger model we recommend `lr=0.00025`.

Another important parameter is the gradient estimator. Currently, three estimators are implemented: 'RT' or
reparameterization trick, 'REINFORCE' and 'PathGradient.'

Scripts periodically save the model to a file. The filename is generated based on the loss function and lattice size.
For example `schwinger_REINFORCE_4x4.zip` is the name of the file for the Schwinger model trained
with the REINFORCE estimator on the 4x4 lattice. Those files are stored in the "out_{model name}" directory. If the
directory does not exist it will be created in the current directory. Any existing file of this name will be
overwritten. To choose another
filename, you can rename the file after running the script or modify the script itself.

All scripts detect if a cuda GPU is available and use it if it is. If no GPU is available, the script will use the CPU.

More detailed information can be found in the scripts themselves.

There are currently three training scripts:

### `phi4.py`

Trains a phi^4 scalar field model. It has two parameters m^2 and lambda (misspelled as lamda to avoid conflict with the
Python keyword).
When lambda equals zero, the model is a free scalar field, and for m^2 >0 can be solved exactly.
In such a case, the script will compare the results with the exact solution.

### `u1.py`

This is a pure gauge abelian U(1) model. It has one parameter beta which is the inverse coupling constant. The model is
exactly solvable for any beta. The script will compare the results with the exact solution. Due to the gauge symmetry
implementation, the lattice size must be a multiple of four.

This model has a large number of possible implementations. For more details, see the script itself.

### `schwinger.py`

This model extends the U(1) model by adding fermions. The fermionic contribution to the action is represented by the
determinant of the Wilson-Dirac operator.. We calculate this determinant explicitly using the
built-in `torch.logdet` function. Due to this, the model is practically limited to small lattice sizes, such as 20x20.
Running the model without a modern GPU with at least 8GB of memory is not recommended.

## Profiling scripts

This is a collection of scripts that can be used to measure some properties of the models, notably timings and memory
usage.

### `timings.py`

This script performs detailed timing measurements of a single pass of the gradient estimators for the Schwinger model.

### `memory.py`

This script measures the memory allocated during a single call to gradient estimator for the Schwinger model.

### `dag.py`

This script measures various properties of the directed acyclic graph (DAG) created by the gradient estimator for the
Schwinger model.





