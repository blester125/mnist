from __future__ import absolute_import
import six

import os
import string
import hashlib
import pytest
from mock import patch
import numpy as np
from mnist.mnist import get_cache_path, check_cache, clear_cache, get_cache, XDG_DATA_DIR


def rstr(l=None, mi=5, ma=21):
    if l is None:
        l = np.random.randint(mi, ma)
    choices = list(string.ascii_letters + string.digits)
    return ''.join([np.random.choice(choices) for _ in range(l)])


@pytest.fixture
def xdg_data_home():
    loc = rstr()
    os.environ[XDG_DATA_DIR] = loc
    yield loc
    del os.environ[XDG_DATA_DIR]


def test_get_cache():
    gold = rstr()
    cache = get_cache(gold)
    assert cache == gold


def test_get_cache_xdg_data_set(xdg_data_home):
    gold = xdg_data_home
    cache = get_cache(None)
    assert cache == gold


def test_get_cache_xdg_default():
    with patch('mnist.mnist.os.path.expanduser') as home_patch:
        home = rstr()
        home_patch.return_value = home
        gold = os.path.join(home, '.local', 'share', 'MNIST')
        cache = get_cache(None)
    assert cache == gold


def test_get_cache_xdg_default_override_name():
    with patch('mnist.mnist.os.path.expanduser') as home_patch:
        home = rstr()
        home_patch.return_value = home
        name = rstr()
        gold = os.path.join(home, '.local', 'share', name)
        cache = get_cache(None, name=name)
    assert cache == gold


def test_get_cache_path():
    cache = rstr()
    url = rstr()
    if six.PY2:
        h = hashlib.sha1(url).hexdigest()
    else:
        h = hashlib.sha1(url.encode('utf-8')).hexdigest()
    with patch('mnist.mnist.os.path.exists') as e_mock:
        e_mock.return_value = True
        path = get_cache_path(url, cache)
    gold = os.path.join(cache, h)
    assert path == gold


def test_get_cache_path_creates():
    cache = rstr()
    with patch('mnist.mnist.os.path.exists') as e_mock:
        e_mock.return_value = False
        with patch('mnist.mnist.os.makedirs') as m_mock:
            get_cache_path("url", cache)
        m_mock.assert_called_once_with(cache)


def test_get_cache_path_none():
    cache = None
    url = rstr()
    path = get_cache_path(url, cache)
    assert path is None


def test_check_cache_none():
    path = None
    there = check_cache(path)
    assert there is False


def test_clear_cache():
    path = rstr()
    with patch('mnist.mnist.get_cache_path') as p_mock:
        p_mock.return_value = path
        with patch('mnist.mnist.os.path.exists') as e_mock:
            e_mock.return_value = True
            with patch('mnist.mnist.os.remove') as r_mock:
                clear_cache(None, None)
            r_mock.assert_called_once_with(path)


def test_clear_cache_none():
    with patch('mnist.mnist.get_cache_path') as p_mock:
        p_mock.return_value = None
        with patch('mnist.mnist.os.remove') as r_mock:
            clear_cache(None, None)
            r_mock.assert_not_called
