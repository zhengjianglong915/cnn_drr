import keras.backend as K
import numpy as np
from keras.utils.np_utils import convert_kernel
from keras import backend as K
from keras import activations, initializations, regularizers, constraints
from keras.engine import InputSpec, Layer, Merge
import theano

class MyMerge(Layer):
    def __init__(self, layers=None, slic = 1, concat_axis=-1, init='glorot_uniform',
                 arguments=None, node_indices=None, tensor_indices=None,
                 name=None):
        self.layers = layers
        self.slic = slic
        self.batch_size = 32
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
        shape1 = input_shape[0]
        shape2 = input_shape[1]

        input_dim = shape1[2]
        self.Wg_shape = (2 * input_dim, self.slic)
        self.Wg = self.init(self.Wg_shape, name='{}_Wg'.format(self.name))
        self.Bg = K.zeros((self.slic,), name='{}_Bg'.format(self.name))
        self.Mr_shape = (self.slic, input_dim, input_dim)
        self.Mr = self.init(self.Mr_shape, name='{}_Mr'.format(self.name))
        self.V_shape = (2 * input_dim, self.slic)
        self.V = self.init(self.V_shape, name='{}_V'.format(self.name))
        self.U_shape = (self.slic,)
        self.U = self.init(self.U_shape, name='{}_U'.format(self.name))
        self.b =  K.zeros((self.slic,), name='{}_b'.format(self.name))

        self.trainable_weights = [self.Wg, self.Bg, self.Mr, self.V, self.b, self.U]
        self.built = True


    def call(self, inputs, mask=None):
        if not isinstance(inputs, list) or len(inputs) <= 1:
            raise TypeError('Merge must be called on a list of tensors '
                            '(at least 2). Got: ' + str(inputs))
        # Case: "mode" is a lambda or function.
        if callable(self.mode):
            arguments = self.arguments
            arg_spec = inspect.getargspec(self.mode)
            if 'mask' in arg_spec.args:
                arguments['mask'] = mask
            return self.mode(inputs, **arguments)

        
        Arg1, Arg2 = inputs
        result, _ = theano.scan(fn=self.grn,
                             outputs_info=None,
                             sequences=[Arg1, Arg2])
        #output = K.reshape(resut, (Arg1.shape[0], Arg1.shape[1] * Arg1.shape[1]))
        return result
    

    def grn(self, arg1, arg2):
        def recurrence(_word1, arg2):
            def _recurrence(_word1, _word2):
                return self._grn(_word1, _word2)
                #return K.dot(_word1, _word2)

            _result, _ = theano.scan(fn=_recurrence,
                             outputs_info=None,
                             sequences=arg2,
                             non_sequences=_word1)
            return _result

        score, _ = theano.scan(fn=recurrence,
                               outputs_info=None,
                               sequences=arg1,
                               non_sequences=arg2)

        #score = K.reshape(score, (score.shape[0],  score.shape[1]))[0]
        return score


    def _grn(self, arg1, arg2):
        inputs = [arg1, arg2]
        hConcat = K.concatenate(inputs)
        
        # gate
        g = K.dot(hConcat, self.Wg) + self.Bg
        g = K.sigmoid(g) 
       

        # Bilinear Model part
        bi_score = self.mul3d(arg1, arg2)        
        bi_score =  bi_score * g
 
        # Single Layer Network
        si_score = K.dot(hConcat, self.V) 
        si_score = K.sigmoid(si_score)    
        
        gr = 1 - g
        si_score = si_score * gr
        score = bi_score + si_score + self.b
       
        U = K.transpose(self.U)
        score = K.dot(score, U)
        return K.sigmoid(score)
    
    def mul3d(self, arg1, arg2):
        for k in range(0, self.slic):
            x_v = K.dot(arg1, self.Mr[k])
            x_v = K.dot(x_v, K.transpose(arg2))
            if k == 0:
                bi_score = [x_v]
            else:
                bi_score = [bi_score, [x_v]]
                bi_score = K.concatenate(bi_score)
        return bi_score 

   
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
        
        return (input_shape[0][0], input_shape[0][1], input_shape[0][1])



        

   
    
