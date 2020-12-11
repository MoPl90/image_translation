from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv3D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.layers import Add
import tensorflow.keras.backend as K
from tensorflow.python.ops import array_ops
import numpy as np

# generator a resnet block
def resnet_block(n_filters, input_layer, dim=3, init='he_normal', norm_layer=BatchNormalization):

	if dim == 3:
		conv_layer = Conv3D
	elif dim == 2:
		conv_layer = Conv2D
	else:
		raise ValueError("need 2D or 3D data")

	# first layer convolutional layer
	g = conv_layer(n_filters, kernel_size=3, padding='same', kernel_initializer=init)(input_layer)
	g = norm_layer()(g)
	g = LeakyReLU(0.2)(g)

	# second convolutional layer
	g = conv_layer(n_filters, kernel_size=3, padding='same', kernel_initializer=init)(g)
	g = norm_layer()(g)
	# concatenate merge channel-wise with input layer
	g = Concatenate()([g, input_layer])

	g = LeakyReLU(0.2)(g)

	return g



def conv_d(layer_input, filters, kernel_size=4, norm=True, dim=3, init='he_normal', norm_layer=BatchNormalization):
	"""Layers used during downsampling"""

	if dim == 3:
		d = Conv3D(filters, kernel_size=kernel_size, strides=2, padding='same', kernel_initializer=init)(layer_input)
	elif dim == 2:
		d = Conv2D(filters, kernel_size=kernel_size, strides=2, padding='same', kernel_initializer=init)(layer_input)
	else:
		raise ValueError("need 2D or 3D data")

	if norm:
		d = norm_layer()(d)

		d = LeakyReLU(alpha=0.2)(d)
	return d


def deconv_d(layer_input, skip_input, filters, kernel_size=4, dropout_rate=0, dim=3, init='he_normal', norm_layer=BatchNormalization):
	"""Layers used during upsampling"""

	if dim == 3:
		upsampling_layer = UpSampling3D
		conv_layer = Conv3D
	elif dim ==2:
		upsampling_layer = UpSampling2D
		conv_layer = Conv2D
	else:
		raise ValueError("need 2D or 3D data")


	u = upsampling_layer(size=2)(layer_input)
	u = conv_layer(filters, kernel_size=kernel_size, strides=1, padding='same', kernel_initializer=init)(u)
	u = LeakyReLU(0.2)(u)

	if dropout_rate>0:
		u = Dropout(dropout_rate)(u)
	u = norm_layer()(u)
	u = Concatenate()([u, skip_input])

	u = LeakyReLU(0.2)(u)
	return u


#see https://arxiv.org/pdf/1903.07291.pdf
def spade_block(latent_in, input_mask, dim=3, init='he_normal'):
	"""Singe SPADE block, including the transformation of latent vectors to scale and shift parameters."""

	if dim == 2:
		conv_layer = Conv2D
	elif dim == 3:
		conv_layer = Conv3D

	# latent_in = BatchNormalization()(latent_in)

	b = conv_layer(filters=latent_in.shape[-1], kernel_size=3, strides=1, padding='same', kernel_initializer=init)(input_mask)
	g = conv_layer(filters=latent_in.shape[-1], kernel_size=3, strides=1, padding='same', kernel_initializer=init)(input_mask)
	out = SPADE()([latent_in, b, g])

	return out

# generator a resnet block
def spade_res_block(fil, latent_in, input_mask, dim=3, init='he_normal', rectify=False):
	"""S{ADE residual block; two-fold application of SPADE block and a 3x3 convolution.

	Input:
	    fil: 		filters for convolutions
	    latent_in: 	latent vector input
	    input_mask: segmentation mask input
	    dim: 		dimensionality of data
	    init: 		kernel initializer
	    rectify: 	one (True) or two (False, default) SPADE blocks?
	     """

	if dim == 2:
		conv_layer = Conv2D
	elif dim == 3:
		conv_layer = Conv3D


	out = spade_block(latent_in, input_mask, dim, init)
	out = LeakyReLU(0.2)(out)
	out = conv_layer(filters=fil, kernel_size=3, padding='same', kernel_initializer=init)(out)

	if not rectify:
		out = spade_block(out, input_mask, dim, init)
		out = LeakyReLU(0.2)(out)
		out = conv_layer(filters=fil, kernel_size=3, padding='same', kernel_initializer=init)(out)

	return out



class InstanceNormalization(Layer):
	''' Thanks for github.com/jayanthkoushik/neural-style '''
	def __init__(self, epsilon=1E-3, **kwargs):
		super(InstanceNormalization, self).__init__(**kwargs)
		self.epsilon = epsilon

	def build(self, input_shape):
		self.scale = self.add_weight(name='scale', shape=(input_shape[-1],), initializer="one", trainable=True)
		self.shift = self.add_weight(name='shift', shape=(input_shape[-1],), initializer="zero", trainable=True)
		super(InstanceNormalization, self).build(input_shape)

	def compute_output_shape(self, input_shape):

		return input_shape

	def get_config(self):
		config = {
				'epsilon':self.epsilon,
		}

		base_config  = super(InstanceNormalization, self,).get_config()

		return dict(list(base_config.items()) + list(config.items()))


	def call(self, inputs, mask=None):
		def image_expand(tensor):
			return K.expand_dims(K.expand_dims(tensor, -2), -2)

		def batch_image_expand(tensor):
			return image_expand(K.expand_dims(tensor, 0))

		input_shape = K.int_shape(inputs)
		reduction_axes = list(range(1, len(input_shape)-1))#average only in pixel space -> channels must be last!

		mu = K.mean(inputs, reduction_axes, keepdims=True)
		sigma = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
		normed = (inputs - mu) / sigma

		scale = batch_image_expand(self.scale)
		shift = batch_image_expand(self.shift)

		return normed * scale + shift


class GaussianNoiseAnneal(Layer):
	def __init__(self, stddev, decay_rate, **kwargs):
		super(GaussianNoiseAnneal, self).__init__(**kwargs)
		self.supports_masking = True
		self.stddev = stddev
		self.decay_rate = decay_rate

	def call(self, inputs, training=None):
		def noised():
			return inputs + K.random_normal(
				shape=array_ops.shape(inputs),
				mean=0.,
				stddev=self.stddev,
				dtype=inputs.dtype)

		self.stddev = self.stddev * (1 - self.decay_rate)
		return K.in_train_phase(noised, inputs, training=training)

	def get_config(self):
		config = {'stddev': self.stddev,
				  'decay_rate':self.decay_rate}
		base_config = super(GaussianNoiseAnneal, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	def compute_output_shape(self, input_shape):
		return input_shape

#Input b and g should be HxWxC
class SPADE(Layer):
    def __init__(self,
             axis=-1,
             momentum=0.99,
             epsilon=1e-3,
             center=True,
             scale=True,
             **kwargs):
        super(SPADE, self).__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
    
    
    def build(self, input_shape):
    
        dim = input_shape[0][self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape[0]) + '.')
    
        super(SPADE, self).build(input_shape)
    
    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs[0])
        
        beta = inputs[1]
        gamma = inputs[2]

        reduction_axes = [0, 1, 2]
        mean = K.mean(inputs[0], reduction_axes, keepdims=True)
        stddev = K.std(inputs[0], reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs[0] - mean) / stddev

        return normed * gamma + beta
    
    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale
        }
        base_config = super(SPADE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
    
        return input_shape[0]


class RandomWeightedAverage(Add):
	"""Takes a randomly-weighted average of two tensors. In geometric terms, this
	outputs a random point on the line between each pair of input points."""

	def __init__(self, BATCH_SIZE, **kwargs):
		super(RandomWeightedAverage, self).__init__(**kwargs)
		self.batch_size = BATCH_SIZE

	def get_config(self):
		config = super().get_config().copy()
		config.update({
			'BATCH_SIZE': self.batch_size,
		})
		return config

	def compute_output_shape(self, input_shape):
		return input_shape

	def call(self, inputs):
		assert inputs[0].shape[1:] == inputs[1].shape[1:], "Inputs must have same shape!"

		if len(inputs[0].shape[1:]) == 4:
			weights = K.random_uniform((self.batch_size, 1, 1, 1, 1))
		elif len(inputs[0].shape[1:]) == 3:
			weights = K.random_uniform((self.batch_size, 1, 1, 1))
		else:
			raise ValueError("Wrong input shape.")

		return (weights * inputs[0]) + ((1 - weights) * inputs[1])