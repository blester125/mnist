from __future__ import absolute_import

import os
import struct
import numpy as np
from mnist.mnist import LABEL_MAGIC, IMAGE_MAGIC


LOC = os.path.realpath(os.path.dirname(__file__))

B = np.random.randint(10, 15)
R = np.random.randint(20, 40)
C = np.random.randint(20, 40)

gold_labels = np.random.randint(0, 9, size=(B,)).astype(np.int32)
gold_images = np.random.randint(0, 255, size=(B, R, C)).astype(np.int32)

DATA_LOC = os.path.join(LOC, 'test_data')
if not os.path.exists(DATA_LOC):
    os.makedirs(DATA_LOC)

np.save(os.path.join(DATA_LOC, 'gold_labels'), gold_labels)
np.save(os.path.join(DATA_LOC, 'gold_images'), gold_images)

gold_labels = gold_labels.astype(np.uint8)
gold_images = gold_images.astype(np.uint8)

good_labels = struct.pack(">2i{}B".format(B), LABEL_MAGIC, B, *gold_labels)
bad_labels = struct.pack(">2i{}B".format(B), IMAGE_MAGIC, B, *gold_labels)

with open(os.path.join(DATA_LOC, 'good_labels'), 'wb') as f:
    f.write(good_labels)

with open(os.path.join(DATA_LOC, 'bad_labels'), 'wb') as f:
    f.write(bad_labels)

good_images = struct.pack(">4i{}B".format(B*R*C), IMAGE_MAGIC, B, R, C, *gold_images.ravel())
bad_images = struct.pack(">4i{}B".format(B*R*C), LABEL_MAGIC, B, R, C, *gold_images.ravel())

with open(os.path.join(DATA_LOC, 'good_images'), 'wb') as f:
    f.write(good_images)

with open(os.path.join(DATA_LOC, 'bad_images'), 'wb') as f:
    f.write(bad_images)
