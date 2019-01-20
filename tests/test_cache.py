from __future__ import absolute_import
import six

import os
import hashlib
import pytest
from mock import patch
from mnist.mnist import get_cache_path, check_cache, clear_cache


def test_get_cache_path():
    cache = "cache"
    url = "url"
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
    cache = "cache"
    with patch('mnist.mnist.os.path.exists') as e_mock:
        e_mock.return_value = False
        with patch('mnist.mnist.os.makedirs') as m_mock:
            get_cache_path("url", cache)
        m_mock.assert_called_once_with(cache)


def test_get_cache_path_none():
    cache = None
    url = 'url'
    path = get_cache_path(url, cache)
    assert path is None

def test_check_cache():
    pass

def test_check_cache_none():
    pass

def test_clear_cache():
    pass

def test_clear_cache_none():
    pass
