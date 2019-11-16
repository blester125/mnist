# mnist

[![Actions Status](https://github.com/blester125/get-mnist/workflows/Unit%20Test/badge.svg)](https://github.com/blester125/get-mnist/actions)

Download MNIST and Fashion MNIST datasets without needing to install tensorflow.

Install with `pip install get-mnist`

Download data with `from mnist import get_mnist; x, y, x_test, y_test = get_mnist('MNIST')` or use `get_fashion_mnist`. Data is downloaded and cached (in this case into the folder called `'MNIST'`).

If you don't set a cache directory it will default to `$XDG_DATA_HOME/MNIST`
