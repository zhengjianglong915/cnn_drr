# coding =utf-8
import keras.backend as K
import theano.tensor as T
import numpy as np
from keras.utils.np_utils import convert_kernel
from keras import backend as K
from keras import activations, initializations, regularizers, constraints
from keras.engine import InputSpec, Layer, Merge
import theano

class GRN(Layer):
    def __init__(self, layers=None, max_len=50,init='glorot_uniform', slice_number=None,batch_size=None,embedding_dim=None,
        arguments=None, node_indices=None, tensor_indices=None, name=None):
        self.layers = layers
        self.max_len = max_len
        self.init = initializations.get(init)
        self.slice_number = slice_number
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim        

        self.node_indices = node_indices
        self.init = initializations.get(init)
        self.arguments = arguments if arguments else {}
        self.mode = "concat"
        # Layer parameters.
        self.inbound_nodes = []
        self.outbound_nodes = []
        self.constraints = {}   
        self._trainable_weights = []
        self._non_trainable_weights = []
        self.regularizers = []
        self.supports_masking = False
        self.uses_learning_phase = False
        self.input_spec = None  # Compatible with anything.
        if not name:
            prefix = self.__class__.__name__.lower()
            name = prefix + '_' + str(K.get_uid(prefix))
        self.name = name

        if layers:
            if not node_indices:
                node_indices = [0 for _ in range(len(layers))]
            
            input_shape = self._arguments_validation(layers,
                                       node_indices, tensor_indices)
            self.build(input_shape)
            self.add_inbound_node(layers, node_indices, tensor_indices)
        else:
            self.built = False

    def _arguments_validation(self, layers, node_indices, tensor_indices):
        '''Validates user-passed arguments and raises exceptions
        as appropriate.
        '''
        if type(layers) not in {list, tuple} or len(layers) < 2:
            raise Exception('A Merge should only be applied to a list of '
                            'layers with at least 2 elements. Found: ' + str(layers))
        if tensor_indices is None:
            tensor_indices = [None for _ in range(len(layers))]

        input_shapes = []
        for i, layer in enumerate(layers):
            layer_output_shape = layer.get_output_shape_at(node_indices[i])
            if type(layer_output_shape) is list:
                # Case: the layer has multiple output tensors
                # and we only need a specific one.
                layer_output_shape = layer_output_shape[tensor_indices[i]]
            input_shapes.append(layer_output_shape)
        return input_shapes


    def build(self, input_shape):
        self.W = self.init((self.slice_number, self.embedding_dim, self.embedding_dim))
        self.V = self.init((self.slice_number, 2*self.embedding_dim))
        self.b = self.init((self.slice_number,))
        self.u = self.init((1,self.slice_number))
        self.W_g = self.init((self.slice_number, 2*self.embedding_dim))
        self.b_g = self.init((self.slice_number,))
        self.trainable_weights = [self.W, self.V, self.b, self.W_g, self.b_g, self.u]
        #self.trainable_weights = [self.W, self.V, self.b, self.u]        
        self.built = True

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
   
    def __call__(self, inputs, mask=None):
        if not isinstance(inputs, list):
            raise TypeError('Merge can only be called on a list of tensors, '
                            'not a single tensor. Received: ' + str(inputs))
        if self.built:
            raise RuntimeError('A Merge layer cannot be used more than once, '
                               'please use '
                               'the "merge" function instead: '
                               '`merged_tensor = merge([tensor_1, tensor2])`.')

        all_keras_tensors = True
        for x in inputs:
            if not hasattr(x, '_keras_history'):
                all_keras_tensors = False
                break

        if all_keras_tensors:
            layers = []
            node_indices = []
            tensor_indices = []
            for x in inputs:
                layer, node_index, tensor_index = x._keras_history
                layers.append(layer)
                node_indices.append(node_index)
                tensor_indices.append(tensor_index)
            self.built = True
            self.add_inbound_node(layers, node_indices, tensor_indices)

            outputs = self.inbound_nodes[-1].output_tensors
            return outputs[0]  # Merge only returns a single tensor.
        else:
            return self.call(inputs, mask)

    def get_output_shape_for(self, input_shape):
        # Must have multiple input shape tuples.
        assert isinstance(input_shape, list)
        # Case: callable self._output_shape.
        if callable(self.mode):
            if callable(self._output_shape):
                output_shape = self._output_shape(input_shape)
                return output_shape
            elif self._output_shape is not None:
                return (input_shape[0][0],) + tuple(self._output_shape)
            else:
                # TODO: consider shape auto-inference with TF.
                raise ValueError('The Merge layer ' + self.name +
                                 ' has a callable `mode` argument, '
                                 'and we cannot infer its output shape '
                                 'because no `output_shape` '
                                 'argument was provided. '
                                 'Make sure to pass a shape tuple '
                                 '(or callable) '
                                 '`output_shape` to Merge.')
        # Pre-defined merge modes.
        
        return (input_shape[0][0],  1,  input_shape[0][1], input_shape[0][1])



        

   
    
