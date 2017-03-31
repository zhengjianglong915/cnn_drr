# -*- coding: utf-8 -*-
__author__ = 'Jeff_chen'
import theano
from keras import backend as K
import theano.tensor as T
import numpy as np
from keras import activations, initializations
from keras.layers.core import Layer
#from keras.layers.core import Layer, MaskedLayer

class Scores(Layer):

    def __init__(self, max_len=50,init='glorot_uniform',slice_number=None,batch_size=None,embedding_dim=None):

        super(Scores, self).__init__()
        self.max_len = max_len
        self.init = initializations.get(init)
        self.slice_number = slice_number
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim

    def build(self):
        self.W = self.init((self.slice_number, self.embedding_dim, self.embedding_dim))
        self.V = self.init((self.slice_number, 2*self.embedding_dim))
        self.b = self.init((self.slice_number,))
        self.u = self.init((1,self.slice_number))
        self.W_g = self.init((self.slice_number, 2*self.embedding_dim))
        self.b_g = self.init((self.slice_number,))
        self.trainable_weights = [self.W, self.V, self.b, self.W_g, self.b_g, self.u]
        #self.trainable_weights = [self.W, self.V, self.b, self.u]

    def cos_scores(self,lstm1,lstm2):
        scores = lstm1 * lstm2
        scores = T.sum(scores,axis=3)
        t1 = T.sqrt(T.sum(lstm1 ** 2,axis=3))
        t2 = T.sqrt(T.sum(lstm2 ** 2,axis=3))
        scores = scores / (t1*t2)
        scores = scores.dimshuffle(0,'x',1,2)
        return scores

    def tensor_scores(self,lstm1,lstm2):
        tw1 = T.tensordot(lstm1, self.W, axes=([3],[1]))
        tw2 = lstm2.dimshuffle(0,1,2,'x',3)   # add a slice number dimension
        tw2 = T.tile(tw2,(1,1,1,self.slice_number,1))
        result_w = T.sum((tw1*tw2),axis=4)

        tv1 = lstm1
        tv2 = T.tile(lstm2,(1,self.max_len,1,1))
        tv12 = T.concatenate((tv1,tv2),axis=3)
        result_v = T.tensordot(tv12,self.V,axes=([3],[1]))

        #result_linear = result_w + result_v + self.b
        #result_non_linear = T.nnet.sigmoid(result_linear)
        #result_final = result_non_linear

        gate = T.nnet.sigmoid(T.tensordot(tv12,self.W_g,axes=([3],[1])) + self.b_g)
        result_final = gate*result_w + (1-gate)*T.nnet.sigmoid(result_v) + self.b

        result_final = T.tensordot(result_final,self.u,axes=([3],[1]))
        result_final = result_final.reshape((lstm1.shape[0],self.max_len,self.max_len))

        #return result_final.dimshuffle(0,3,1,2)
        return result_final.dimshuffle(0,'x',1,2)

     def call(self, inputs, mask=None):
        lstm1,lstm2 = inputs

        lstm1 = lstm1.dimshuffle(0,'x',1,2)
        lstm2 = lstm2.dimshuffle(0,'x',1,2)
        lstm1 = T.tile(lstm1,(1,self.max_len,1,1))
        lstm1 = lstm1.dimshuffle(0,2,1,3)

        #computation of score matrix
        #cos_scores = self.cos_scores(lstm1,lstm2)
        tensor_scores = self.tensor_scores(lstm1,lstm2)

        #scores = self.g * tensor_scores + (1-self.g) * cos_scores
        return tensor_scores

    def get_output_shape_for(self, input_shape):
        return (None , 1 , self.max_len, self.max_len)

