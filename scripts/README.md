# Scripts

This directory contains scripts that can be used to train and analyze models.

## Training scripts

These scripts can be used to train different models. They are all very similar and contain lots of intentionally
duplicated code to allow them to be further developed separately. By default, all scripts use the 8x8 square lattice.

They are basic scripts without any command line parameters (except for the `schwinger.py` script). All the parameters
have to be changed inside the scripts.

Two key parameters, batch_size and n_batches, significantly impact efficiency. First, for the REINFORCE estimator, the
total number of configurations, calculated as n_batches × batch_size, must not be too small due to its high initial
variance. We recommend using at least 1024. Second, it is advisable to choose the largest possible batch_size that does
not cause an out-of-memory error on your GPU.

One of the most sensitive parameters is the learning rate (lr). The values used in the scripts work reasonably well, but
may require tuning when other parameters are changed.

Another important parameter is the gradient estimator. Currently, three estimators are implemented:
RT (reparameterization trick), 'REINFORCE' and 'PathGradient.'

The scripts periodically save the trained model to a file. The filename is generated based on the gradient estimator and
lattice size. For example, schwinger_REINFORCE_4x4.zip is the filename for the Schwinger model trained with the
REINFORCE estimator on a 4×4 lattice. These files are stored in the out_{model name} directory. If the directory does
not exist, it will be created in the current directory. Any existing file with the same name will be overwritten. To use
a different filename, rename the file after running the script or modify the script itself.

All scripts automatically detect if a CUDA-compatible GPU is available and use it if possible. If no GPU is detected,
the script will default to using the CPU.

For more details, refer to the scripts themselves.

There are currently three training scripts:

### `phi4.py`

Trains a phi^4 scalar field model. It has two parameters m^2 and lambda (misspelled as lamda to avoid conflict with the
Python keyword). When lambda equals zero, the model is a free scalar field and can be solved exactly. In such a case,
the script will compare the results with the exact solution.

### `u1.py`

This is a pure gauge U(1) model. It has one parameter beta which is the inverse coupling constant. The model is exactly
solvable for any beta. The script will compare the results with the exact solution. Due to the gauge symmetry
implementation, the lattice size must be a multiple of four.

This model has a large number of possible implementations. For more details, see the script itself.

### `schwinger.py`

This model extends the U(1) model by adding fermions. The fermionic contribution to the action is represented by the
determinant of the Wilson-Dirac operator. We calculate this determinant explicitly using the built-in `torch.logdet`
function. Due to this, the model is practically limited to small lattice sizes, such as 20x20. Running the model without
a modern GPU with at least 8GB of memory is not recommended.

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





