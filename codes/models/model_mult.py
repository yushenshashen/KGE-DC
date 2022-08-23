import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from .layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder, BilinearDecoder, DEDICOMDecoder, weight_variable_glorot

def weight_variable_glorot2(input_dim, output_dim, name=""):
    initial = tf.random_uniform(
        [input_dim, output_dim],
        minval=0,
        maxval=1,
        dtype=tf.float32
    )
    return tf.Variable(initial, name=name)

class sdcnet():
    def __init__(self, placeholders, num_features, emb_dim, num_features_nonzero, name, use_cellweights=True, use_layerweights=True, act=tf.nn.relu, fncellscount=None):
        self.name = name
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.emb_dim = emb_dim
        self.num_features_nonzero = num_features_nonzero
        self.cellscount = fncellscount
        self.adjs1 = {cellidx : placeholders['net1_adj_norm_' + str(cellidx)] for cellidx in range(self.cellscount)}
        self.dropout = placeholders['dropout']
        self.act = act
        self.use_cellweights = use_cellweights
        self.use_layerweights = use_layerweights
        # self.att = tf.Variable(tf.constant([0.5, 0.33, 0.25]))
        with tf.variable_scope(self.name + '_cellweights'):
            if use_cellweights == True:
                self.cellweights = weight_variable_glorot2(self.cellscount, 1, name='cellweights')
            else:
                # self.cellweights = [ float(1 / self.adjs1[cellidx][0].shape[0]) for cellidx in range(self.cellscount) ] 
                self.cellweights = [1] * self.cellscount

        if use_layerweights == True:
            with tf.variable_scope(self.name + '_layerweights'):
                self.layerweights = weight_variable_glorot2(3, 1, name='layerweights')

        with tf.variable_scope(self.name + '_loop_weights'):
            self.loop_weights = {}
            self.loop_weights['weights_0'] = weight_variable_glorot(self.input_dim,self.emb_dim, name='weights_0')
            self.loop_weights['weights_1'] = weight_variable_glorot(self.emb_dim, self.emb_dim, name='weights_1') 
            self.loop_weights['weights_2'] = weight_variable_glorot(self.emb_dim, self.emb_dim, name='weights_2' )

        # with tf.variable_scope(self.name + '_bilinear_weights'):
        #     self.bilinear_weights = {}
        #     for cellidx in range(self.cellscount):
        #         self.bilinear_weights['weights_'+str(cellidx)] = weight_variable_glorot(self.emb_dim, self.emb_dim, name='weights_'+str(cellidx))

        with tf.variable_scope(self.name + '_weights_global'):
            self.weights_global = weight_variable_glorot(self.emb_dim, self.emb_dim, name='weights_global')

        with tf.variable_scope(self.name + '_weights_local'):
            self.weights_local = {}
            for cellidx in range(self.cellscount):
                self.weights_local['weights_local_'+str(cellidx)] =  tf.reshape( weight_variable_glorot(self.emb_dim, 1,name='weights_local_'+str(cellidx)), [-1])

        with tf.variable_scope(self.name):
            self.build()

    def build(self):

        self.hidden1_net = []
        for cellidx in range(self.cellscount):
            self.hidden1_net.append( self.cellweights[cellidx] * GraphConvolutionSparse(name='net1_layer1_'+str(cellidx),input_dim=self.input_dim,output_dim=self.emb_dim,adj=self.adjs1[cellidx],num_features_nonzero=self.num_features_nonzero,dropout=self.dropout,act=lambda x:x)(self.inputs) )
            self.hidden1_net.append( tf.matmul(tf.sparse.to_dense(self.inputs), self.loop_weights['weights_0'])) 
        self.hidden1_net = tf.add_n(self.hidden1_net)
        self.hidden1_net = tf.nn.relu( tf.nn.l2_normalize(self.hidden1_net, dim=0) )

        self.hidden2_net = []
        for cellidx in range(self.cellscount):
            self.hidden2_net.append( self.cellweights[cellidx] * GraphConvolution(name='net1_layer2_'+str(cellidx),input_dim=self.emb_dim,output_dim=self.emb_dim,adj=self.adjs1[cellidx],dropout=self.dropout,act=lambda x:x)(self.hidden1_net) )
            self.hidden2_net.append( tf.matmul(self.hidden1_net, self.loop_weights['weights_1']))
        self.hidden2_net = tf.add_n(self.hidden2_net)
        self.hidden2_net = tf.nn.relu( tf.nn.l2_normalize(self.hidden2_net, dim=0) )

        self.hidden3_net = []
        for cellidx in range(self.cellscount):
            self.hidden3_net.append( self.cellweights[cellidx] * GraphConvolution(name='net1_layer3_'+str(cellidx),input_dim=self.emb_dim,output_dim=self.emb_dim,adj=self.adjs1[cellidx],dropout=self.dropout,act=lambda x:x)(self.hidden2_net) )
            self.hidden3_net.append( tf.matmul(self.hidden2_net, self.loop_weights['weights_2']))
        self.hidden3_net = tf.add_n(self.hidden3_net)
        self.hidden3_net = tf.nn.relu( tf.nn.l2_normalize(self.hidden3_net, dim=0) )

        if self.use_layerweights == True:
            self.embeddings = self.layerweights[0] * self.hidden1_net + self.layerweights[1] * self.hidden2_net + self.layerweights[2] * self.hidden3_net
        else:
            self.embeddings = 1/3 * self.hidden1_net + 1/ 3*self.hidden2_net + 1/3*self.hidden3_net

        self.embeddings = tf.nn.l2_normalize(self.embeddings, dim=0)

        #decoder bilinear
        # self.reconstructions = []
        # for cellidx in range(self.cellscount):
        #     self.reconstructions.append( BilinearDecoder(name='bilinear_decoder_'+str(cellidx), weight=self.bilinear_weights['weights_'+str(cellidx)], act=tf.nn.sigmoid)(self.embeddings) )

        #dedicom
        self.reconstructions = []
        for cellidx in range(self.cellscount):
            self.reconstructions.append( DEDICOMDecoder(name='dedicom_decoder_'+str(cellidx), weights_global=self.weights_global, weights_local=self.weights_local['weights_local_'+str(cellidx)], act=tf.nn.sigmoid)(self.embeddings) )

