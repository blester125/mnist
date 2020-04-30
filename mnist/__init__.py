from __future__ import absolute_import

__version__ = "1.0.1"

# This lets us reference the mnist module (containing helper functions) even after we
# import the mnist function of the same name. This is used for mocking during tests
import mnist.mnist as mnist_module
from mnist.mnist import mnist, fashion_mnist
