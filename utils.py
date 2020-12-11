import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
# from tensorflow.keras.layers import _Merge

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# choose random instances
	if len(dataset) == n_samples:
		ix = range(n_samples)
	else:
		ix = np.random.randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = np.ones((n_samples, *patch_shape))
	return X, y
	
# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, dataset, patch_shape):
	# generate fake instance
	if g_model.name == 'GAN_generator':
		X = g_model.predict(np.random.normal(0,1, (len(dataset), *g_model.inputs[0].shape[1:])))
	elif g_model.name == 'PIX2PIX_generator':
		X = g_model.predict(dataset)
	elif g_model.name == 'SPADE_generator':
		latent = np.random.normal(0, 1, (len(dataset), *g_model.inputs[0].shape[1:]))
		X = g_model.predict([latent, dataset])		
	
	# create 'fake' class labels (0)
	y = np.ones((len(X), *patch_shape))
	return X, y

# update image pool for fake images
def update_image_pool(pool, images, max_size=50):
	selected = list()
	for image in images:
		if len(pool) < max_size:
			# stock the pool
			pool.append(image)
			selected.append(image)
		elif np.random.random() < 0.5:
			# use image, but don't add it to the pool
			selected.append(image)
		else:
			# replace an existing image and use replaced image
			ix = np.random.randint(0, len(pool))
			selected.append(pool[ix])
			pool[ix] = image
	return np.asarray(selected)

#calculate the rceptive field of the PatchGAN discriminator
def receptive_field(num_strde_two_convs=4):

	def f(output_size, kernel_size, strides):
		return (output_size-1) *strides + kernel_size

	last_layer = f(1, 4, 1) #input size for each output pixel/voxel
	tmp = f(last_layer, 4, 1)

	for _ in range(num_strde_two_convs):
		tmp = f(tmp, 4, 2)

	return tmp

def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch."""

    return K.mean(y_true * y_pred)


# class RandomWeightedAverage(_Merge):
#     """Takes a randomly-weighted average of two tensors. In geometric terms, this
#     outputs a random point on the line between each pair of input points.
#     Inheriting from _Merge is a little messy but it was the quickest solution I could
#     think of. Improvements appreciated."""
#
#     def __init__(self, batch_size):
#         super().__init__()
#         self.batch_size = batch_size
#
#     def _merge_function(self, inputs):
#         weights = K.random_uniform((self.batch_size, 1, 1, 1, 1))
#         return (weights * inputs[0]) + ((1 - weights) * inputs[1])

def gradient_penalty_loss(y_true, y_pred, averaged_samples,gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the
    loss function that penalizes the network if the gradient norm moves away from 1.
    However, it is impossible to evaluate this function at all points in the input
    space. The compromise used in the paper is to choose random points on the lines
    between real and generated samples, and check the gradients at these points. Note
    that it is the gradient w.r.t. the input averaged samples, not the weights of the
    discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator
    and evaluate the loss. Then we get the gradients of the discriminator w.r.t. the
    input averaged samples. The l2 norm and penalty can then be calculated for this
    gradient.
    Note that this loss function requires the original averaged samples as input, but
    Keras only supports passing y_true and y_pred to loss functions. To get around this,
    we make a partial() of the function with the averaged_samples argument, and use that
    for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = tf.gradients(y_pred, averaged_samples,unconnected_gradients='zero')[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

def jens_loss(y_true, y_pred):
	return K.mean(K.square(y_true - y_pred) + K.square(y_true**2 - y_pred**2), axis=-1)

