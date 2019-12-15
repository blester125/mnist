from __future__ import absolute_import

import string
import pytest
import numpy as np
from mock import patch
from mnist.mnist import get_data


def rstr(l=None, mi=5, ma=21):
    if l is None:
        l = np.random.randint(mi, ma)
    choices = list(string.ascii_letters + string.digits)
    return "".join([np.random.choice(choices) for _ in range(l)])


def test_in_cache():
    path = rstr()
    with patch("mnist.mnist_module.urlretrieve") as r_mock:
        with patch("mnist.mnist_module.get_cache_path") as p_mock:
            p_mock.return_value = path
            with patch("mnist.mnist_module.check_cache") as c_mock:
                c_mock.return_value = True
                with patch("mnist.mnist_module.GzipFile") as g_mock:
                    get_data(None, None)
                g_mock.assert_called_once_with(path)
                r_mock.assert_not_called()


def test_not_in_cache():
    path = rstr()
    url = rstr()
    with patch("mnist.mnist_module.urlretrieve") as r_mock:
        r_mock.return_value = path, None
        with patch("mnist.mnist_module.get_cache_path") as p_mock:
            p_mock.return_value = path
            with patch("mnist.mnist_module.check_cache") as c_mock:
                c_mock.return_value = False
                with patch("mnist.mnist_module.GzipFile") as g_mock:
                    get_data(url, None)
                r_mock.assert_called_once_with(url, path)
                g_mock.assert_called_once_with(path)
