from __future__ import absolute_import

import pytest

import os
import numpy as np
from mnist.mnist import read_labels, read_images

DATA_LOC = os.path.join(
    os.path.realpath(os.path.dirname(__file__)),
    'test_data'
)


def test_read_labels():
    gold_labels = np.load(os.path.join(DATA_LOC, 'gold_labels.npy'))
    with open(os.path.join(DATA_LOC, 'good_labels'), 'rb') as f:
        data = f.read()
    labels = read_labels(data)
    np.testing.assert_equal(labels, gold_labels)


def test_corrput_labels():
    with open(os.path.join(DATA_LOC, 'bad_labels'), 'rb') as f:
        data = f.read()
    with pytest.raises(AssertionError):
        read_labels(data)


def test_read_images():
    gold_images = np.load(os.path.join(DATA_LOC, 'gold_images.npy'))
    with open(os.path.join(DATA_LOC, 'good_images'), 'rb') as f:
        data = f.read()
    images = read_images(data)
    np.testing.assert_equal(images, gold_images)


def test_corrput_images():
    with open(os.path.join(DATA_LOC, 'bad_images'), 'rb') as f:
        data = f.read()
    with pytest.raises(AssertionError):
        read_images(data)
