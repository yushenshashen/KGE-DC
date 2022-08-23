import tensorflow.compat.v1 as tf
import numpy as np
import scipy.sparse as sp


def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = tf.random_uniform(
        [input_dim, output_dim],
        minval=-init_range,
        maxval=init_range,
        dtype=tf.float32
    )
    return tf.Variable(initial, name=name)

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out*(1./keep_prob)
    

class GraphConvolution():
    """Basic graph convolution layer for undirected graph without edge labels."""

    def __init__(self, input_dim, output_dim, adj, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = tf.nn.dropout(x, 1-self.dropout)
            x = tf.matmul(x, self.vars['weights'])
            x = tf.sparse_tensor_dense_matmul(self.adj, x)
            outputs = self.act(x)
        return outputs


class GraphConvolutionSparse():
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, num_features_nonzero, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.num_features_nonzero = num_features_nonzero
    #
    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = dropout_sparse(x, 1-self.dropout, self.num_features_nonzero)
            x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
            x = tf.sparse_tensor_dense_matmul(self.adj, x)
            outputs = self.act(x)
        return outputs


class InnerProductDecoder():
    """Decoder model layer for link prediction."""
    def __init__(self, name, dropout=0., act=tf.nn.sigmoid):
        self.name = name
        self.dropout = dropout
        self.act = act

    def __call__(self, inputs):
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs

class BilinearDecoder():
    """Decoder model layer for link prediction."""
    def __init__(self, weight, name, dropout=0., act=tf.nn.sigmoid):
        self.name = name
        self.dropout = dropout
        self.act = act
        self.vars = {}
        self.weight = weight

    def __call__(self, inputs):
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        x = inputs
        x = tf.matmul(x, self.weight )
        x = tf.matmul(x, tf.transpose(inputs))
        # x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs


class DEDICOMDecoder():
    """Decoder model layer for link prediction."""
    def __init__(self, weights_global, weights_local, name, dropout=0., act=tf.nn.sigmoid):
        self.name = name
        self.dropout = dropout
        self.act = act
        self.weights_global = weights_global
        self.weights_local = weights_local

    def __call__(self, inputs):
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        relation = tf.diag( self.weights_local )
        x = inputs
        x = tf.matmul(x, relation )
        x = tf.matmul(x, self.weights_global )
        x = tf.matmul(x, relation )
        x = tf.matmul(x, tf.transpose(inputs))
        # x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs

