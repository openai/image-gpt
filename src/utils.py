import os
import json
import time
import pickle
import subprocess
import math
from tqdm import tqdm

import numpy as np
import tensorflow as tf

def iter_data(*datas, n_batch=128, truncate=False, verbose=False, max_batches=float("inf")):
    n = len(datas[0])
    if truncate:
        n = (n//n_batch)*n_batch
    n = min(n, max_batches*n_batch)
    n_batches = 0
    for i in tqdm(range(0, n, n_batch), total=n//n_batch, disable=not verbose, ncols=80, leave=False):
        if n_batches >= max_batches: raise StopIteration
        if len(datas) == 1:
            yield datas[0][i:i+n_batch]
        else:
            yield (d[i:i+n_batch] for d in datas)
        n_batches += 1

def squared_euclidean_distance(a, b):
    b = tf.transpose(b)
    a2 = tf.reduce_sum(tf.square(a), axis=1, keepdims=True)
    b2 = tf.reduce_sum(tf.square(b), axis=0, keepdims=True)
    ab = tf.matmul(a, b)
    d = a2 - 2*ab + b2
    return d

def color_quantize(x, np_clusters):
    clusters = tf.Variable(np_clusters, dtype=tf.float32, trainable=False)
    x = tf.reshape(x, [-1, 3])
    d = squared_euclidean_distance(x, clusters)
    return tf.argmin(d, 1)

def count_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters
