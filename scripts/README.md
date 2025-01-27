# Scripts

This directory contains scripts that can be used to train and analyze models.

## Training scripts

Those scripts can be used to train different models. They are all very similar and contain lots of duplicated code that
was left on purpose as to allow them to be further developed separately. All the models are defined on a
two-dimensional lattice. The lattice size is a parameter of the model. By default, all scripts use the 8x8 square
lattice.

They are 'plain vanilla' scripts without any command line parameters. All the parameters have to be changed inside the
script.

Two parameters `batch_size` and `n_batches`
are crucial for the efficiency. First of all for the REINFORCE estimator the total number of
configurations `n_batches x batch_size` cannot be too small because of its high initial variance.
We recommend at least 1024. Second, one should choose the biggest `batch_size` that does not cause the out of memory
error on your GPU.

One of the most important parameters is the learning rate `lr`. For the Schwinger model we recommend `lr=0.00025`.

Another important parameter is the loss function used to get the gradient estimator. Currently three loss
functions are implemented: 'rt' or reparameterization trick, 'REINFORCE' and 'path_gradient.' All scripts use the
'REINFORCE' estimator by default.

Scripts periodically save the model to a file. The file name is constructed from the name of the loss function and
the lattice size. For example `schwinger_REINFORCE_4x4.zip` is the name of the file for the Schwinger model trained
with the REINFORCE estimator on the 4x4 lattice. Those files are stored in the "out_{model name}" directory. If the
directory does not exist it will be created in the current directory. Any existing file of this name will be
overwritten. To choose another
file name you can rename the file after running the script or modify the script itself.

All scripts detect if a cuda GPU is available and use it if it is. If no GPU is available, the script will use the CPU.

More detailed information can be found in the scripts themselves.

There currently three training scripts:

### phi4.py

Trains a phi^4 scalar field model it has two parameters m^2 and lamda (misspelled lambda, because that is a python
keyword) when lambda equals zero, the model is a free scalar field, and for m^2 >0 can be solved exactly.
In such a case the script will compare the results with the exact solution.

### u1.py

This is a pure gauge abelian U(1) model. It has one parameter beta which is the inverse coupling constant. The model is
exactly solvable for any beta. The script will compare the results with the exact solution. Because of the gauge
symmetry implementation, the lattice size is limited to multiples of four.

This model has a large number of possible implementations. For more details, see the script itself.

### schwinger.py

This model extends the U(1) model by adding fermions. The fermions are implemented as the Wilson fermions. The fermionic
part of the action is the determinant of the Wilson-Dirac operator. We calculate this determinant explicitly using the
build in `torch.logdet` function. Because of that, in practice, the model is limited to the small lattice sizes like
20x20. It is not recommended to run the model without a modern GPU with at least 8GB of memory.

## Analysis scripts

### nmcmc

This script takes as an
argument the name of the file with the model trained by the `schwinger.py` script and loads the model. Next, it uses it
to generate new configurations which are then used in the Neural Markov Chain Monte-Carlo (NMCMC) algorithm.

```shell
python ./scripts/nmcmc.py  out_schwinger/scwinger_REINFORCE_4x4.zip
```

## Profiling scripts

This is a collection of scripts that can be used to measure some properties of the models, notably timings and memory
usage.

### timings

This scripts performs detailed timing measurements on the loss functions for the Schwinger model.

### memory

This script measures the memory allocated during a single call to loss function for the Schwinger model.

### dag

This script measures various properties of the DAG created by the loss function for the Schwinger model.





