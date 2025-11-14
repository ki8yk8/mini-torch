# MiniTorch
Minitorch is a way of me learning about autograd and creating a pytorch like library. 

## Features
1. Imports like Pytorch,
2. Implements basic functions like backpropagation, gradient descent, neurons, linear layers
3. Extensible with suport of custom modules like Python

## Remaining Task
1. Vectorization
2. Implementing CNNs, Transformers

I am working on theory of Jacobian Matrix and it's use for computing derivatives in a vector. So, will do this in V2.

## Usage
### Installation
1. Install all the necessary requirements using;
```bash
pip install -r requirements.txt
```

2. Build the minitorch package for local installation.
```bash
python -m build
```

3. Install the minitorch package locally
```bash
pip install .
```

### Running Demos
1. Demos can be found in `/demos`, you can run individual of them by;
```bash
python -m ./demos/<example-file-name>.py
```

2. If you want to create your own model, please look at documentation below.

**Note that** since, the minitorch library doesn't use vectorization things are not efficient hence, you cannot train the model on big data even like on MNIST. It becomes too slow. Try implementing some simpler stuffs.

## Documentation
### Value
- This is similar to Pytorch Tensor.
```python
from minitorch import Value

a = Value(0.0)
a.grad    # gradient of a
a.backward()    # backward propagate the value
```
You can create a complex arithmetic operation and do backward on top to see demo of backprop. See `demos/00-autograd.py`.

### Creating a custom model
You can import
- Linear
- Neuron
- Module
- RNN
- CrossEntropy
- MSE
- NLL

from `minitorch.nn` and create your own model.

Examples are at `demos/01-or-gate.py`, `demos/02-custom-model.py`.

### Using activations and optimizers
Following activations;
- Sigmoid
- Softmax
- ReLU
- LogSoftmax
- TanH

can be imported from `minitorch.activations`. 

And you can also load optimizer, for now gradient descent only. 

See example at; `05-simple-polynomial.py`

## Demo Video
Upload from GitHub

## Screenshots
![Activation functions](/pics/activation-functions.png)
![Loss curve for simple polynomial function](/pics/loss-curve-simple-polynomial.png)
![Loss curve for complex polynomial function](/pics/loss-curve-complex-polynomial.png)