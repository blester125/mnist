# mnist

[![PyPi Version](https://img.shields.io/pypi/v/get-mnist)](https://pypi.org/project/get-mnist/) [![Actions Status](https://github.com/blester125/mnist/workflows/Unit%20Test/badge.svg)](https://github.com/blester125/mnist/actions) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Download MNIST and Fashion MNIST datasets without needing to install tensorflow.

# Installation

```
pip install get-mnist
```

# CLI Download

```
mnist --dataset [mnist, fashion] --cache [CACHE]
```

Use the `--dataset` flag to decide if you want to download the original MNIST dataset or the Fashion MNIST dataset. Use the `--cache` flag to decide where to save the dataset. If omitted it defaults to `$XDG_DATA_HOME/MNIST` or `$XDG_DATA_HOME/FASHION_MNIST`.

# Programatic Download


```python
from mnist import get_mnist
x, y, x_test, y_test = mnist('MNIST')
x, y, x_test, y_test = fashion_mnist('FASHION_MNIST')
```

The function argument is the name of the directory to cache the dataset in. These functions can also take `train_url`, `train_label_url`, `test_url`, and `test_label_url` to download data from different sources.
