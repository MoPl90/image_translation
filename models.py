from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape, Flatten
from tensorflow.keras.layers import Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose
from tensorflow.keras.layers import UpSampling2D, UpSampling3D
from tensorflow.keras.layers import MaxPooling2D, MaxPooling3D
from tensorflow.keras.layers import Cropping2D, Cropping3D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.initializers import RandomNormal
from functools import partial
import matplotlib.pylab as plt
import numpy as np
from tensorflow.keras.utils import plot_model
import seaborn as sns
from layers import *
from utils import *
from data_generator import *

plt.rc('font', **{'size':14, 'family':'serif'})


class PIX2PIX():

	def __init__(self,
				 image_shape=(128, 128, 128, 3),
				 n_classes = 1,
				 batchsize = 1,
				 batchsize_eval =5,
				 lr = 0.0002,
				 gf = 64,
				 n_res = -1,
				 filters_d = [32, 64, 128, 256, 256],
				 norm = 'instance',
				 out_activation="tanh",
				 leakiness=0.02,
				 loss = 'mse',
				 dis_weight =.5,
				 adv_weight = 1.,
				 recon_weight = 100.,
				 pool_size = 50,
				 init_weights=0.02,
				 noise=False,
				 noise_decay=0.01,
				 decay = 0,
				 dropoutrate=0,
				 out_dir="."
				 ):

		self.image_shape = np.array(image_shape)
		self.input_shape = np.array(image_shape)
		self.input_shape[-1] = n_classes

		self.BATCH_SIZE = batchsize
		self.BATCH_SIZE_EVAL = batchsize_eval
		self.lr = lr
		self.dropout = dropoutrate
		self.steps = 0
		self.gf = gf
		self.filters_d = filters_d
		self.out_activation="tanh"
		self.leakiness=0.02,
		self.out_dir = out_dir

		#Normalization Layer
		if 'instance' in norm or 'Instance' in norm:
			self.norm_layer = InstanceNormalization
		else:
			self.norm_layer = BatchNormalization

		#adversarial loss type
		if 'jens' in loss:
			self.loss = jens_loss
		elif 'wasserstein' in loss or 'Wasserstein' in loss:
			self.loss = wasserstein
		else:
			self.loss = 'mse'

		#loss weights
		self.dis_weight  = dis_weight
		self.adv_weight = adv_weight
		self.recon_weight = recon_weight

		#discriminator fake pool size
		self.pool_size = pool_size

		#initializers' standard deviation
		self.init_weights = init_weights

		#initialize the Gaussian noise
		self.noise = noise
		self.noise_decay = noise_decay


		#use residual blocks (n_res>0) or U-Net n_res<0
		self.n_res = n_res

		opt = Adam(lr=self.lr, beta_1=0.5, decay = decay)

		# generator
		self.g_model = self.define_generator()
		# discriminator: [real/fake]
		self.d_model = self.define_discriminator(opt)
		self.d_model.compile(loss='mse',
								   optimizer=opt,
								   metrics=['accuracy'])

		self.composite_model = self.define_composite_model(self.g_model, self.d_model, opt)
		self.d_loss = []
		self.g_loss = []

		self.d_loss_eval = [[0, 0]]
		self.g_loss_eval = [[0, 0, 0]]
		self.epochs = [0]

	# define the standalone generator model
	def define_generator(self):
		init = RandomNormal(stddev=self.init_weights)

		# Image input
		in_image = Input(shape=tuple(self.input_shape))

		#use U-Net
		if self.n_res < 0:

			d = []
			# Downsampling
			d1 = conv_d(in_image, self.gf, norm=False, dim=len(self.image_shape[:-1]), init=init, norm_layer=self.norm_layer, leakiness=self.leakiness)
			d.append(d1)
			d2 = conv_d(d1, self.gf * 2, dim=len(self.image_shape[:-1]), init=init, norm_layer=self.norm_layer, leakiness=self.leakiness)
			d.append(d2)
			d3 = conv_d(d2, self.gf * 4, dim=len(self.image_shape[:-1]), init=init, norm_layer=self.norm_layer, leakiness=self.leakiness)
			d.append(d3)
			d4 = conv_d(d3, self.gf * 8, dim=len(self.image_shape[:-1]), init=init, norm_layer=self.norm_layer, leakiness=self.leakiness)
			d.append(d4)
			if self.image_shape[-2] > 64:
				d5 = conv_d(d4, self.gf * 8, dim=len(self.image_shape[:-1]), init=init, norm_layer=self.norm_layer, leakiness=self.leakiness)
				d.append(d5)
			if self.image_shape[-2] > 96:
				d6 = conv_d(d5, self.gf * 8, dim=len(self.image_shape[:-1]), init=init, norm_layer=self.norm_layer, leakiness=self.leakiness)
				d.append(d6)
			if self.image_shape[-2] > 128:
				d7 = conv_d(d6, self.gf * 8, dim=len(self.image_shape[:-1]), init=init, norm_layer=self.norm_layer, leakiness=self.leakiness)
				d.append(d7)
				print('da')
			
			print([i.shape for i in  d], self.image_shape[-2])
			# Upsampling
			if self.image_shape[-2] > 128:
				u = deconv_d(d[-1], d[-2], self.gf * 8, dropout_rate=self.dropout, dim=len(self.image_shape[:-1]), init=init, norm_layer=self.norm_layer, leakiness=self.leakiness)
			else:
				u = d[-1]
				d.append(u)
			if self.image_shape[-2] > 96:
				u = deconv_d(u, d[-3], self.gf * 8, dropout_rate=self.dropout, dim=len(self.image_shape[:-1]), init=init, norm_layer=self.norm_layer, leakiness=self.leakiness)
			else:
				u = d[-2]
				d.append(u)
			if self.image_shape[-2] > 64:
				u = deconv_d(u, d[-4], self.gf * 8, dropout_rate=self.dropout, dim=len(self.image_shape[:-1]), init=init, norm_layer=self.norm_layer, leakiness=self.leakiness)
			else:
				u = d[-3]
				d.append(u)
			u = deconv_d(u, d3, self.gf * 4, dropout_rate=self.dropout, dim=len(self.image_shape[:-1]), init=init, norm_layer=self.norm_layer, leakiness=self.leakiness)
			u = deconv_d(u, d2, self.gf * 2, dropout_rate=self.dropout, dim=len(self.image_shape[:-1]), init=init, norm_layer=self.norm_layer, leakiness=self.leakiness)
			u = deconv_d(u, d1, self.gf, dropout_rate=self.dropout, dim=len(self.image_shape[:-1]), init=init, norm_layer=self.norm_layer, leakiness=self.leakiness)

			if len(self.image_shape[:-1]) == 3:
				u = UpSampling3D(size=2)(u)
				out_img = Conv3D(self.image_shape[-1], kernel_size=4, strides=1, padding='same', activation=self.out_activation)(u)
			elif len(self.image_shape[:-1]) == 2:
				u = UpSampling2D(size=2)(u)
				out_img = Conv2D(self.image_shape[-1], kernel_size=4, strides=1, padding='same', activation=self.out_activation)(u)
			else:
				raise ValueError("Data must be 2D or 3D")


		#use ResNet blocks
		else:
			#check dimensionality of image data
			if len(self.image_shape[:-1]) == 3:
				conv_layer = Conv3D
				upsampling_layer = UpSampling3D
			elif len(self.image_shape[:-1]) == 2:
				conv_layer = Conv2D
				upsampling_layer = UpSampling2D
			else:
				raise ValueError("Data must be 2D or 3D")

			filters_g = [2**i * self.gf for i in range(3)]

			g = conv_layer(filters_g[0], kernel_size=7, padding='same', kernel_initializer=init)(in_image)
			g = self.norm_layer()(g)
			g = LeakyReLU(0.2)(g)

			for fil in filters_g[1:]:
				g = conv_layer(fil, kernel_size=3, strides=2, padding='same', kernel_initializer=init)(g)
				g = self.norm_layer()(g)
				g = LeakyReLU(0.2)(g)

			res_blocks = [g]
			for i in range(self.n_res):
				in_res = res_blocks[-1]
				res_blocks.append(resnet_block(filters_g[-1], in_res, dim=len(self.image_shape[:-1]), init=init, norm_layer=self.norm_layer, leakiness=self.leakiness))

			g = res_blocks[-1]

			for fil in filters_g[::-1][1:]:
				# g = conv_transpose_layer(fil, kernel_size=4, strides=2, padding='same', kernel_initializer=init)(g)
				g = conv_layer(fil, kernel_size=3, strides=1, padding='same', kernel_initializer=init)(g)
				g = upsampling_layer()(g)
				g = self.norm_layer()(g)
				g = LeakyReLU(self.leakiness)(g)

			n_filters = self.image_shape[-1]
			g = conv_layer(n_filters, kernel_size=7, padding='same', kernel_initializer=init)(g)
			g = Dropout(self.dropout)(g)
			g = self.norm_layer()(g)
			out_img = Activation(self.out_activation)(g)

		model = Model(in_image, out_img, name="PIX2PIX_generator")

		plot_model(model, to_file=self. out_dir + "/generator.png", show_shapes=True, show_layer_names=True)
		# model.summary()
		return model

	# define the discriminator model
	def define_discriminator(self, opt):
		# weight initialization
		init = RandomNormal(stddev=self.init_weights)
		#keep the patch size < image size:
		max_index = int(np.argwhere(self.filters_d == np.array([self.image_shape[1]]))[0])
		max_index += 1
		self.filters_d = self.filters_d[:max_index]
		if self.image_shape[1] < 256:
			d_filters = self.filters_d
		else:
			d_filters = self.filters_d

		#check dimensionality of image data
		if len(self.image_shape[:-1]) == 3:
			conv_layer = Conv3D
			conv_transpose_layer = Conv3DTranspose
		elif len(self.image_shape[:-1]) == 2:
			conv_layer = Conv2D
		else:
			raise ValueError("Data must be 2D or 3D")

		# source image input
		in_map = Input(shape=tuple(self.input_shape))
		in_image = Input(shape=tuple(self.image_shape))
		d = Concatenate(axis=-1)([in_image,in_map])

		if self.noise:
			d = GaussianNoiseAnneal(0.2, self.noise_decay)(d)

		for fil in d_filters:
			d = conv_layer(fil, kernel_size=4, strides=2, padding='same', kernel_initializer=init)(d)
			d = self.norm_layer()(d)
			d = LeakyReLU(alpha=0.2)(d)

		d = conv_layer(self.filters_d[-1], kernel_size=4, padding='same', kernel_initializer=init)(d)
		d = self.norm_layer()(d)
		d = LeakyReLU(alpha=0.2)(d)
		# patch output
		patch_out = conv_layer(1, kernel_size=4, padding='same', kernel_initializer=init)(d)
		# define model
		model = Model([in_map, in_image], patch_out, name="discriminator")
		# compile model with weighting of least squares loss
		model.compile(loss=self.loss,  optimizer=opt, loss_weights=[self.dis_weight])

		# model.summary()
		plot_model(model, to_file=self. out_dir + "/discriminator.png", show_shapes=True, show_layer_names=True)
		return model


	# define a composite model for updating generators by adversarial and cycle loss
	def define_composite_model(self, g_model, d_model, opt):
		# ensure the model we're updating is trainable
		g_model.trainable = True
		# mark discriminator as not trainable
		d_model.trainable = False

		# Input images and their conditioning images
		img_A = Input(shape=tuple(self.input_shape))
		img_B = Input(shape=tuple(self.image_shape))

		# By conditioning on A generate a fake version of B
		fake_B = g_model(img_A)

		# Discriminators determines validity of translated images / condition pairs
		valid = d_model([img_A, fake_B])

		model = Model(inputs=[img_A, img_B], outputs=[valid, fake_B], name='composite_model')
		model.compile(loss=[self.loss, 'mae'],
							  loss_weights=[self.adv_weight, self.recon_weight],
							  optimizer=opt)

		# model.summary()
		plot_model(model, to_file=self. out_dir + "/composite_model.png", show_shapes=True, show_layer_names=True)
		return model

	def train_on_batch(self, trainGenerator, testGenerator):
		# determine the output square shape of the discriminator
		patch_shape = self.d_model.output_shape[1:]

		# unpack dataset
		trainA, trainB = next(trainGenerator)

		pool = list()
		# ---------------------
		#  Train Discriminator
		# ---------------------

		imgs_A, valid = generate_real_samples(trainA, self.BATCH_SIZE, patch_shape)
		imgs_B, _ = generate_real_samples(trainB, self.BATCH_SIZE, patch_shape)

		#check which channels to use
		if imgs_A.shape[-1] > self.input_shape[-1]:
			imgs_A = imgs_A[...,:self.input_shape[-1]]

		if imgs_B.shape[-1] > self.image_shape[-1]:
			imgs_B = imgs_B[..., :self.image_shape[-1]]

		#generate fake samples and patch labels
		fake_B, fake = generate_fake_samples(self.g_model, imgs_A, patch_shape)
		fake_B = update_image_pool(pool, fake_B, max_size=self.pool_size)

		# Train the discriminators (original images = real / generated = Fake)
		d_loss_real = self.d_model.train_on_batch([imgs_A, imgs_B], valid)
		d_loss_fake = self.d_model.train_on_batch([imgs_A, fake_B], fake)

		d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

		self.d_loss.append(d_loss)

		# -----------------
		#  Train Generator
		# -----------------
		# Train the generator
		g_loss = self.composite_model.train_on_batch([imgs_A, imgs_B], [valid, imgs_B])

		self.g_loss.append(g_loss)

		self.steps +=1

		# summarize performance every 50 batches
		if self.steps % 50 == 0:
			print("[Epoch %d] [Batch %d] [D loss: %f, acc: %3d%%] [G loss: %1.3f = %1.1f * %1.3f + %1.1f * %1.3f]" % (
																				   len(self.epochs),
																				   self.steps,
																				   d_loss[0],
																				   100 * d_loss[1],
																				   g_loss[0],
																				   self.adv_weight,
																				   g_loss[1],
																				   self.recon_weight,
																				   g_loss[2]))

		#evaluate and validate the performance after each epoch, the save
		if self.steps % len(trainGenerator) == 0:
			self.save_loss()
			self.epochs.append(self.steps)
			self.visualize(testGenerator)
			self.evaluate(testGenerator)
			self.save(len(self.epochs)-1)
			np.save(self.out_dir + '/res/epochs.npy', self.epochs)


	def save_loss(self, name='training'):

		"""Plot the loss functions and save plots."""

		d_loss, g_loss = np.asarray(self.d_loss), np.asarray(self.g_loss)

		f = plt.figure(figsize=(16, 8))
		ax = f.add_subplot(1, 2, 1)
		ax.plot(np.arange(1,self.steps+1), d_loss.T[0],  c='r', label='discriminator loss')
		ax.plot(np.arange(1,self.steps+1), d_loss.T[1],  c='b', label='discriminator accuracy')
		if len(self.d_loss_eval) > 1 and name == 'evaluation':
			ax.plot(np.asarray(self.epochs), np.asarray(self.d_loss_eval).T[0], 'r--')
			ax.plot(np.asarray(self.epochs), np.asarray(self.d_loss_eval).T[1], 'b--')
		ax.set_yscale('log')
		ax.legend()
		ax.set_title('Discriminator error')

		ax = f.add_subplot(1, 2, 2)
		ax.plot(np.arange(1,self.steps+1), g_loss.T[0],  c='r', label=r'total generator loss')
		ax.plot(np.arange(1,self.steps+1), g_loss.T[1], c='b', label=r'adversarial loss')
		ax.plot(np.arange(1,self.steps+1), g_loss.T[2], c='g', label=r'reconstruction loss')
		if len(self.g_loss_eval) > 1 and name == 'evaluation':
			ax.plot(np.asarray(self.epochs), np.asarray(self.g_loss_eval).T[0], 'r--')
			ax.plot(np.asarray(self.epochs), np.asarray(self.g_loss_eval).T[1], 'b--')
			ax.plot(np.asarray(self.epochs), np.asarray(self.g_loss_eval).T[2], 'g--')
		ax.set_yscale('log')
		ax.legend()
		ax.set_title('Generator loss')

		plt.savefig(self.out_dir + '/res/' + name + '.png')
		plt.close()

		np.save(self.out_dir + '/res/d_loss.npy', d_loss)
		np.save(self.out_dir + '/res/g_loss.npy', g_loss)

	def evaluate(self, testGenerator):
		"""Plot the loss and evaluation errors."""

		# determine the output square shape of the discriminator
		patch_shape = self.d_model.output_shape[1:]

		# unpack dataset
		i = 0
		d_loss = 0
		g_loss = 0
		while i < len(testGenerator):
			testA, testB = next(testGenerator)
			i += 1

			# select a batch of real samples
			imgs_A, valid = generate_real_samples(testA, self.BATCH_SIZE, patch_shape)
			imgs_B, _ = generate_real_samples(testB, self.BATCH_SIZE, patch_shape)
			#check which channels to use
			if imgs_A.shape[-1] > self.input_shape[-1]:
				imgs_A = imgs_A[...,:self.input_shape[-1]]

			if imgs_B.shape[-1] > self.image_shape[-1]:
				imgs_B = imgs_B[..., :self.image_shape[-1]]

			#generate fake samples and patch labels
			fake_B, fake = generate_fake_samples(self.g_model, imgs_A, patch_shape)

			#evaluate discriminator
			d_loss_real = self.d_model.evaluate([imgs_A, imgs_B], valid, verbose=0)
			d_loss_fake = self.d_model.evaluate([imgs_A, fake_B], fake, verbose=0)

			d_loss += 0.5 * np.add(d_loss_real, d_loss_fake)


			# evaluate generator
			g_loss += np.asarray(self.composite_model.evaluate([imgs_A, imgs_B], [valid, imgs_B], verbose=0))



		self.d_loss_eval.append(d_loss / len(testGenerator))
		self.g_loss_eval.append(g_loss / len(testGenerator))

		np.save(self.out_dir + '/res/d_loss_eval.npy', self.d_loss_eval)
		np.save(self.out_dir + '/res/g_loss_eval.npy', self.g_loss_eval)

		self.save_loss(name='evaluation')

	def visualize(self, testGen):
			"""Plot a volume, cycled to domain B and then back to domain A"""

			imgs_A, imgs_B = next(testGen)
			idx = np.random.randint(len(imgs_A))
			image_A = imgs_A[idx]
			image_B = imgs_B[idx]
			n_classes = self.input_shape[-1]
			cols = sns.color_palette("pastel", n_classes - 2)
			# background is transparent
			cols.insert(0, 'none')
			# Add stroke class
			cols.append((1,0,0))
			fake_B = self.g_model.predict(imgs_A)

			if len(self.image_shape) > 3:
				fake_B = fake_B[idx]
				image_A = np.transpose(image_A, (2,1,0,3))
				image_B = np.transpose(image_B, (2,1,0,3))
				fake_B = np.transpose(fake_B, (2,1,0,3))

				non_zero_slices = np.argwhere(np.any(image_B[..., 0] > 0, axis=(1, 2))).T[0]
			else:
				image_A = np.transpose(imgs_A, (0,2,1,3))
				image_B = np.transpose(imgs_B, (0,2,1,3))
				fake_B = np.transpose(fake_B, (0,2,1,3))
				non_zero_slices = np.argwhere(np.any(image_B[...,0] > 0, axis=(1, 2))).T[0]

			image_A = np.flip(image_A, axis=(1,2))
			image_B = np.flip(image_B, axis=(1,2))
			fake_B = np.flip(fake_B, axis=(1,2))

			num_ims = min(10, len(non_zero_slices))

			fig, ax_arr = plt.subplots(num_ims, 3, figsize=(9, 3*num_ims))

			ax_arr = ax_arr.reshape((num_ims, 3))

			for i in range(0,num_ims):
				(ax1, ax2, ax3) = ax_arr[i]

				idx = (len(non_zero_slices)) // num_ims
				ax1.contourf(np.argmax(image_A[non_zero_slices[i*idx],::-1,:],axis=-1),
							levels = np.arange(n_classes + 1) - 0.5,
							colors = cols)
				ax1.set_xticks([], [])
				ax1.set_yticks([], [])
				ax1.set_aspect(1)

				ax2.imshow(fake_B[non_zero_slices[i*idx], :, :, 0],
						cmap='bone',
						vmin=-1,
						vmax=1)
				if self.image_shape[-1] > 1:
					ax2.contour(fake_B[non_zero_slices[i*idx], :, :, 1],
								linewidths=0.5,
								levels=[0.],
								colors=['r'])
				else:
					ax2.contour(np.argmax(image_A[non_zero_slices[i*idx], :, :, :], axis=-1),
								levels = np.arange(n_classes + 1) - 0.5,
								colors=cols,
								linewidths=0.5)
				ax2.set_xticks([], [])
				ax2.set_yticks([], [])

				ax3.imshow(image_B[non_zero_slices[i*idx], :, :, 0],
						cmap='bone',
						vmin=-1,
						vmax=1)
				if self.image_shape[-1] > 1:
					ax3.contour(image_B[non_zero_slices[i*idx], :, :, 1],
								linewidths=0.5,
								levels=[0.],
								colors=['r'])
				else:
					ax3.contour(np.argmax(image_A[non_zero_slices[i*idx], :, :, :], axis=-1),
								levels = np.arange(n_classes + 1) - 0.5,
								colors=cols,
								linewidths=0.5)
				ax3.set_xticks([], [])
				ax3.set_yticks([], [])

				if i==0:
					ax1.set_title('segmentation map')
					ax2.set_title('generated image')
					ax3.set_title('ground truth')

			plt.savefig(self.out_dir + '/res/output_sample' + str(len(self.epochs) - 1) + '.png')
			plt.close()

	def saveModel(self, model, name):  # Save a Model

		"""Save a model."""

		model.save(name + ".h5")

	def save(self, num=-1):

		"""Save the GAN's submodels individually.

        Input:
            int; identifier for a given model.
        """

		if num < 0:
			num = str(self.steps)
		else:
			num = str(num)


		self.saveModel(self.g_model, self.out_dir + "/Models/gen_" + num)
		self.saveModel(self.d_model, self.out_dir + "/Models/dis_" + num)

	def load(self, location, num=-1):
		"""Load the GAN's submodels.

        Input:
            int; identifier for the desired model.
        """
		if num < 0:
			num =""
		else:
			num = str(num)
		opt = Adam(lr=self.lr, beta_1=0.5)

		self.g_model.load_weights(location + "/Models/gen_" + num + '.h5')
		self.d_model.load_weights(location + "/Models/dis_" + num + '.h5')

		self.composite_model = self.define_composite_model(self.g_model, self.d_model, opt)

		self.epochs = list(np.load(location + '/res/epochs.npy')[:int(num) + 1])
		self.steps = self.epochs[-1]
		self.d_loss = list(np.load(location + '/res/d_loss.npy')[:self.steps])
		self.d_loss_eval = list(np.load(location + '/res/d_loss_eval.npy')[:(int(num)+1)])
		self.g_loss = list(np.load(location + '/res/g_loss.npy')[:self.steps])
		self.g_loss_eval = list(np.load(location + '/res/g_loss_eval.npy')[:(int(num)+1)])


class SPADE():

	def __init__(self,
				 image_shape=(128, 128, 128, 3),
				 n_classes=1,
				 batchsize=1,
				 batchsize_eval=5,
				 lr=0.0002,
				 gf=64,
				 latent_dim=100,
				 filters_d=[32, 64, 128, 256, 256],
				 norm = 'instance',
				 out_activation="tanh",
				 leakiness=0.02,
				 loss = 'mse',
				 dis_weight =.5,
				 adv_weight = 1.,
				 recon_weight = 100.,
				 pool_size = 50,
				 init_weights=0.02,
				 noise=False,
				 noise_decay=0.01,
				 decay = 0,
				 dropoutrate=0,
				 out_dir=".",
				 ):

		self.image_shape = np.array(image_shape)
		self.input_shape = np.array(image_shape)
		self.input_shape[-1] = n_classes

		self.BATCH_SIZE = batchsize
		self.BATCH_SIZE_EVAL = batchsize_eval
		self.lr = lr
		self.dropout = dropoutrate
		self.steps = 0
		self.gf = gf
		self.latent_dim = latent_dim
		self.filters_d = filters_d
		self.out_dir = out_dir
		self.out_activation=out_activation
		self.leakiness=leakiness

		#Normalization Layer
		if 'instance' in norm or 'Instance' in norm:
			self.norm_layer = InstanceNormalization
		else:
			self.norm_layer = BatchNormalization

		#adversarial loss type
		if 'jens' in loss:
			self.loss = jens_loss
		elif 'wasserstein' in loss or 'Wasserstein' in loss:
			self.loss = wasserstein
		else:
			self.loss = 'mse'

		#loss weights
		self.dis_weight  = dis_weight
		self.adv_weight = adv_weight
		self.recon_weight = recon_weight

		#discriminator fake pool size
		self.pool_size = pool_size

		#initializers' standard deviation
		self.init_weights = init_weights

		#initialize the Gaussian noise
		self.noise = noise
		self.noise_decay = noise_decay

		opt = Adam(lr=self.lr, beta_1=0.5, decay = decay)

		# generator
		self.g_model = self.define_generator()
		# discriminator: [real/fake]
		self.d_model = self.define_discriminator(opt)
		self.d_model.compile(loss='mse',
								   optimizer=opt,
								   metrics=['accuracy'])

		self.composite_model = self.define_composite_model(self.g_model, self.d_model, opt)
		self.d_loss = []
		self.g_loss = []

		self.d_loss_eval = [[0, 0]]
		self.g_loss_eval = [[0, 0, 0]]
		self.epochs = [0]

	# define the standalone generator model
	def define_generator(self):
		init = RandomNormal(stddev=self.init_weights)

		# Image input
		latent_in = Input(shape=(self.latent_dim))
		mask_in = Input(shape=tuple(self.input_shape))

		g = Dense(1 * 4 * 4 * self.gf)(latent_in)

		dim = len(self.image_shape[:-1])
		if dim == 2:
			g = Reshape((4, 4, self.gf))(g)

			pooling_layer = MaxPooling2D
			upsampling_layer = UpSampling2D
			conv_layer = Conv2D
		elif dim == 3:
			g = Reshape((4, 4, 1, self.gf))(g)

			pooling_layer = MaxPooling3D
			upsampling_layer = UpSampling3D
			conv_layer = Conv3D

		res = mask_in.shape[1]
		masks = {str(res):mask_in}
		while res > 4:
			key = str(res // 2)
			masks[key] = pooling_layer()(masks[str(res)])
			res = res // 2

		fil = self.gf
		while g.shape[1] < self.image_shape[1]:

			if fil == g.shape[-1]:
				g = Add()([g, spade_res_block(fil, g, masks[str(g.shape[1])], dim=dim, init=init, leakiness=self.leakiness)])
			else:
				g = Add()([spade_res_block(fil, g, masks[str(g.shape[1])], dim=dim, init=init, rectify=True, leakiness=self.leakiness),
						   spade_res_block(fil, g, masks[str(g.shape[1])], dim=dim, init=init, leakiness=self.leakiness)])

			g = upsampling_layer()(g)
			fil = fil // 2

		g = conv_layer(1, kernel_size=7, strides=1, padding='same', kernel_initializer=init)(g)

		out_img = Activation(self.out_activation)(g)

		model = Model([latent_in, mask_in], out_img, name="SPADE_generator")

		# plot_model(model, to_file=self. out_dir + "/generator.png", show_shapes=True, show_layer_names=True)
		model.summary()
		return model

	# define the discriminator model
	def define_discriminator(self, opt):
		# weight initialization
		init = RandomNormal(stddev=self.init_weights)
		#keep the patch size < image size:
		max_index = int(np.argwhere(self.filters_d == np.array([self.image_shape[1]]))[0])
		max_index += 1
		self.filters_d = self.filters_d[:max_index]
		if self.image_shape[1] < 256:
			d_filters = self.filters_d
		else:
			d_filters = self.filters_d

		#check dimensionality of image data
		if len(self.image_shape[:-1]) == 3:
			conv_layer = Conv3D
			conv_transpose_layer = Conv3DTranspose
		elif len(self.image_shape[:-1]) == 2:
			conv_layer = Conv2D
		else:
			raise ValueError("Data must be 2D or 3D")

		# source image input
		in_map = Input(shape=tuple(self.input_shape))
		in_image = Input(shape=tuple(self.image_shape))
		d = Concatenate(axis=-1)([in_image,in_map])

		if self.noise:
			d = GaussianNoiseAnneal(0.2, self.noise_decay)(d)

		for fil in d_filters:
			d = conv_layer(fil, kernel_size=4, strides=2, padding='same', kernel_initializer=init)(d)
			d = self.norm_layer()(d)
			d = LeakyReLU(alpha=0.2)(d)

		d = conv_layer(self.filters_d[-1], kernel_size=4, padding='same', kernel_initializer=init)(d)
		d = self.norm_layer()(d)
		d = LeakyReLU(alpha=0.2)(d)
		# patch output
		patch_out = conv_layer(1, kernel_size=4, padding='same', kernel_initializer=init)(d)
		# define model
		model = Model([in_map, in_image], patch_out, name="discriminator")
		# compile model with weighting of least squares loss
		model.compile(loss=self.loss,  optimizer=opt, loss_weights=[self.dis_weight])

		model.summary()
		# plot_model(model, to_file=self. out_dir + "/discriminator.png", show_shapes=True, show_layer_names=True)
		return model


	# define a composite model for updating generators by adversarial and cycle loss
	def define_composite_model(self, g_model, d_model, opt):
		# ensure the model we're updating is trainable
		g_model.trainable = True
		# mark discriminator as not trainable
		d_model.trainable = False

		# Input images and their conditioning images
		latent_in = Input(shape=self.latent_dim)
		mask = Input(shape=tuple(self.input_shape))
		img = Input(shape=tuple(self.image_shape))

		# By conditioning on A generate a fake version of B
		fake = g_model([latent_in, mask])

		# Discriminators determines validity of translated images / condition pairs
		valid = d_model([mask, fake])

		model = Model(inputs=[latent_in, mask, img], outputs=[valid, fake], name='composite_model')
		model.compile(loss=[self.loss, 'mae'],
							  loss_weights=[self.adv_weight, self.recon_weight],
							  optimizer=opt)

                # model.summary()
		# plot_model(model, to_file=self. out_dir + "/composite_model.png", show_shapes=True, show_layer_names=True)
		return model

	def train_on_batch(self, trainGenerator, testGenerator):
		# determine the output square shape of the discriminator
		patch_shape = self.d_model.output_shape[1:]

		# unpack dataset
		trainA, trainB = next(trainGenerator)

		pool = list()
		# ---------------------
		#  Train Discriminator
		# ---------------------

		imgs_A, valid = generate_real_samples(trainA, self.BATCH_SIZE, patch_shape)
		imgs_B, _ = generate_real_samples(trainB, self.BATCH_SIZE, patch_shape)

		#check which channels to use
		if imgs_A.shape[-1] > self.input_shape[-1]:
			imgs_A = imgs_A[...,:self.input_shape[-1]]

		if imgs_B.shape[-1] > self.image_shape[-1]:
			imgs_B = imgs_B[..., :self.image_shape[-1]]

		#generate fake samples and patch labels
		fake_B, fake = generate_fake_samples(self.g_model, imgs_A, patch_shape)
		fake_B = update_image_pool(pool, fake_B, max_size=self.pool_size)

		# Train the discriminators (original images = real / generated = Fake)
		d_loss_real = self.d_model.train_on_batch([imgs_A, imgs_B], valid)
		d_loss_fake = self.d_model.train_on_batch([imgs_A, fake_B], fake)

		d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

		self.d_loss.append(d_loss)

		# -----------------
		#  Train Generator
		# -----------------
		# Train the generator
		latent = np.random.normal(0, 1, (self.BATCH_SIZE, self.latent_dim))
		g_loss = self.composite_model.train_on_batch([latent, imgs_A, imgs_B], [valid, imgs_B])

		self.g_loss.append(g_loss)

		self.steps +=1

		# summarize performance every 50 batches
		if self.steps % 50 == 0:
			print("[Epoch %d] [Batch %d] [D loss: %f, acc: %3d%%] [G loss: %1.3f = %1.1f * %1.3f + %1.1f * %1.3f]" % (
																				   len(self.epochs),
																				   self.steps,
																				   d_loss[0],
																				   100 * d_loss[1],
																				   g_loss[0],
																				   self.adv_weight,
																				   g_loss[1],
																				   self.recon_weight,
																				   g_loss[2]))

		#evaluate and validate the performance after each epoch, the save
		if self.steps % len(trainGenerator) == 0:
			self.save_loss()
			self.epochs.append(self.steps)
			self.visualize(testGenerator)
			self.evaluate(testGenerator)
			self.save(len(self.epochs)-1)
			np.save(self.out_dir + '/res/epochs.npy', self.epochs)

	def save_loss(self, name='training'):

		"""Plot the loss functions and save plots."""

		d_loss, g_loss = np.asarray(self.d_loss), np.asarray(self.g_loss)

		f = plt.figure(figsize=(16, 8))
		ax = f.add_subplot(1, 2, 1)
		ax.plot(np.arange(1,self.steps+1), d_loss.T[0],  c='r', label='discriminator loss')
		ax.plot(np.arange(1,self.steps+1), d_loss.T[1],  c='b', label='discriminator accuracy')
		if len(self.d_loss_eval) > 1 and name == 'evaluation':
			ax.plot(np.asarray(self.epochs), np.asarray(self.d_loss_eval).T[0], 'r--')
			ax.plot(np.asarray(self.epochs), np.asarray(self.d_loss_eval).T[1], 'b--')
		ax.set_yscale('log')
		ax.legend()
		ax.set_title('Discriminator error')

		ax = f.add_subplot(1, 2, 2)
		ax.plot(np.arange(1,self.steps+1), g_loss.T[0],  c='r', label=r'total generator loss')
		ax.plot(np.arange(1,self.steps+1), g_loss.T[1], c='b', label=r'adversarial loss')
		ax.plot(np.arange(1,self.steps+1), g_loss.T[2], c='g', label=r'reconstruction loss')
		if len(self.g_loss_eval) > 1 and name == 'evaluation':
			ax.plot(np.asarray(self.epochs), np.asarray(self.g_loss_eval).T[0], 'r--')
			ax.plot(np.asarray(self.epochs), np.asarray(self.g_loss_eval).T[1], 'b--')
			ax.plot(np.asarray(self.epochs), np.asarray(self.g_loss_eval).T[2], 'g--')
		ax.set_yscale('log')
		ax.legend()
		ax.set_title('Generator loss')

		plt.savefig(self.out_dir + '/res/' + name + '.png')
		plt.close()

		np.save(self.out_dir + '/res/d_loss.npy', d_loss)
		np.save(self.out_dir + '/res/g_loss.npy', g_loss)


	def evaluate(self, testGenerator):
		"""Plot the loss and evaluation errors."""

		# determine the output square shape of the discriminator
		patch_shape = self.d_model.output_shape[1:]

		# unpack dataset
		i = 0
		d_loss = 0
		g_loss = 0
		while i < len(testGenerator):
			testA, testB = next(testGenerator)
			i += 1

			# select a batch of real samples
			imgs_A, valid = generate_real_samples(testA, self.BATCH_SIZE, patch_shape)
			imgs_B, _ = generate_real_samples(testB, self.BATCH_SIZE, patch_shape)
			#check which channels to use
			if imgs_A.shape[-1] > self.input_shape[-1]:
				imgs_A = imgs_A[...,:self.input_shape[-1]]

			if imgs_B.shape[-1] > self.image_shape[-1]:
				imgs_B = imgs_B[..., :self.image_shape[-1]]

			#generate fake samples and patch labels
			fake_B, fake = generate_fake_samples(self.g_model, imgs_A, patch_shape)

			#evaluate discriminator
			d_loss_real = self.d_model.evaluate([imgs_A, imgs_B], valid, verbose=0)
			d_loss_fake = self.d_model.evaluate([imgs_A, fake_B], fake, verbose=0)

			d_loss += 0.5 * np.add(d_loss_real, d_loss_fake)


			# evaluate generator
			latent = np.random.normal(0, 1, (self.BATCH_SIZE, self.latent_dim))
			g_loss += np.asarray(self.composite_model.evaluate([latent, imgs_A, imgs_B], [valid, imgs_B], verbose=0))



		self.d_loss_eval.append(d_loss / len(testGenerator))
		self.g_loss_eval.append(g_loss / len(testGenerator))

		np.save(self.out_dir + '/res/d_loss_eval.npy', self.d_loss_eval)
		np.save(self.out_dir + '/res/g_loss_eval.npy', self.g_loss_eval)

		self.save_loss(name='evaluation')

	def visualize(self, testGen):
		"""Plot a volume, cycled to domain B and then back to domain A"""

		imgs_A, imgs_B = next(testGen)
		idx = np.random.randint(len(imgs_A))
		image_A = imgs_A[idx]
		image_B = imgs_B[idx]
		n_classes = self.input_shape[-1]
		cols = sns.color_palette("pastel", n_classes - 2)
		# background is transparent
		cols.insert(0, 'none')
		# Add stroke class
		cols.append('r')
		latent = np.random.normal(0, 1, (self.BATCH_SIZE_EVAL, self.latent_dim))
		fake_B = self.g_model.predict([latent, imgs_A])


		if len(self.image_shape) > 3:
			fake_B = fake_B[idx]
			image_A = np.transpose(image_A, (2,1,0,3))
			image_B = np.transpose(image_B, (2,1,0,3))
			fake_B = np.transpose(fake_B, (2,1,0,3))

			non_zero_slices = np.argwhere(np.any(image_B[..., 0] > 0, axis=(1, 2))).T[0]
		else:
			image_A = np.transpose(imgs_A, (0,2,1,3))
			image_B = np.transpose(imgs_B, (0,2,1,3))
			fake_B = np.transpose(fake_B, (0,2,1,3))
			non_zero_slices = np.argwhere(np.any(image_B[...,0] > 0, axis=(1, 2))).T[0]

		image_A = np.flip(image_A, axis=(1,2))
		image_B = np.flip(image_B, axis=(1,2))
		fake_B = np.flip(fake_B, axis=(1,2))

		num_ims = min(10, len(non_zero_slices))


		fig, ax_arr = plt.subplots(num_ims, 3, figsize=(9, 3*num_ims))

		ax_arr = ax_arr.reshape((num_ims, 3))

		for i in range(0,num_ims):
			(ax1, ax2, ax3) = ax_arr[i]

			idx = (len(non_zero_slices)) // num_ims
			ax1.contourf(np.argmax(image_A[non_zero_slices[i*idx]][::-1],axis=-1),
						 levels = np.arange(n_classes + 1) - 0.5,
						 colors = cols)
			ax1.set_xticks([], [])
			ax1.set_yticks([], [])
			ax1.set_aspect(1)

			ax2.imshow(fake_B[non_zero_slices[i*idx], :, :, 0],
					   cmap='bone',
					   vmin=-1,
					   vmax=1)
			if self.image_shape[-1] > 1:
				ax2.contour(fake_B[non_zero_slices[i*idx], :, :, 1],
							linewidths=0.5,
							levels=[0.],
							colors=['r'])
			else:
				ax2.contour(np.argmax(image_A[non_zero_slices[i*idx], :, :, :], axis=-1),
							levels = np.arange(n_classes + 1) - 0.5,
							colors=cols,
							linewidths=0.5)
			ax2.set_xticks([], [])
			ax2.set_yticks([], [])

			ax3.imshow(image_B[non_zero_slices[i*idx], :, :, 0],
					   cmap='bone',
					   vmin=-1,
					   vmax=1)
			if self.image_shape[-1] > 1:
				ax3.contour(image_B[non_zero_slices[i*idx], :, :, 1],
							linewidths=0.5,
							levels=[0.],
							colors=['r'])
			else:
				ax3.contour(np.argmax(image_A[non_zero_slices[i*idx], :, :, :], axis=-1),
							levels = np.arange(n_classes + 1) - 0.5,
							colors=cols,
							linewidths=0.5)
			ax3.set_xticks([], [])
			ax3.set_yticks([], [])

			if i==0:
				ax1.set_title('segmentation map')
				ax2.set_title('generated image')
				ax3.set_title('ground truth')

		plt.savefig(self.out_dir + '/res/output_sample' + str(len(self.epochs) - 1) + '.png')
		plt.close()

	def saveModel(self, model, name):  # Save a Model

		"""Save a model."""

		model.save(name + ".h5")

	def save(self, num=-1):

		"""Save the GAN's submodels individually.

        Input:
            int; identifier for a given model.
        """

		if num < 0:
			num = str(self.steps)
		else:
			num = str(num)

		self.saveModel(self.g_model, self.out_dir + "/Models/gen_" + num)
		self.saveModel(self.d_model, self.out_dir + "/Models/dis_" + num)

	def load(self, location, num=-1):
		"""Load the GAN's submodels.

        Input:
            int; identifier for the desired model.
        """
		if num < 0:
			num = ""
		else:
			num = str(num)
		opt = Adam(lr=self.lr, beta_1=0.5)

		self.g_model.load_weights(location + "/Models/gen_" + num + '.h5')
		self.d_model.load_weights(location + "/Models/dis_" + num + '.h5')

		self.composite_model = self.define_composite_model(self.g_model, self.d_model, opt)

		self.epochs = list(np.load(location + '/res/epochs.npy')[:int(num) + 1])
		self.steps = self.epochs[-1]
		self.d_loss = list(np.load(location + '/res/d_loss.npy')[:self.steps])
		self.d_loss_eval = list(np.load(location + '/res/d_loss_eval.npy')[:(int(num)+1)])
		self.g_loss = list(np.load(location + '/res/g_loss.npy')[:self.steps])
		self.g_loss_eval = list(np.load(location + '/res/g_loss_eval.npy')[:(int(num)+1)])




class GAN():

	def __init__(self,
				 image_shape=[32, 32, 32, 1],
				 n_classes=1,
				 batchsize=1,
				 batchsize_eval=5,
				 lr=0.0002,
				 filter_d=8,
				 latent_size=100,
				 init_weights=0.02,
				 norm = 'instance',
				 dropoutrate=0.3,
				 out_activation="tanh",
				 leakiness=0.02,
				 loss='mse',
				 loss_weights=[1,0,1],
				 mode=0,
				 pool_size = 50,
				 out_dir="."
				 ):

		self.image_shape = np.array(image_shape)
		self.n_classes = n_classes
		self.BATCH_SIZE = batchsize
		self.BATCH_SIZE_EVAL = batchsize_eval
		self.lr = lr
		self.dropout = dropoutrate
		self.leakiness = leakiness
		self.steps = 0
		self.filter_d = filter_d
		self.latent_size = latent_size
		self.mode=mode
		self.out_dir = out_dir

		#initializers' standard deviation
		self.init_weights = init_weights

		#output layer activation
		self.out_activation = out_activation

		#Normalization Layer
		if 'instance' in norm.lower():
			self.norm_layer = InstanceNormalization
		else:
			self.norm_layer = BatchNormalization

		#set up optimizer
		self.opt = Adam(lr=self.lr, beta_1=0.5)

		self.loss_weights = loss_weights

		self.g_loss = []
		self.d_loss = []
		self.d_loss_eval = [[1,1]]
		self.g_loss_eval = [1]
		self.epochs = [0]

		#set up loss function
		self.loss = loss.lower()
		if 'wasserstein' in self.loss:
			self.loss = wasserstein_loss
			self.d_loss_eval = [[1,1,1]]
		elif 'jens' in self.loss:
			self.loss = jens_loss

		#discriminator fake pool size
		self.pool_size = pool_size

		#Set up the models
		self.g_model = self.define_generator()
		self.d_model = self.define_discriminator()
		self.adversarial_model = self.define_composite_model(self.g_model, self.d_model)

		if self.loss is wasserstein_loss:
			self.average_model = self.define_average_model(self.d_model)


	# define the standalone generator model
	def define_generator(self):

		# weight initialization
		init = RandomNormal(stddev=self.init_weights)

		# image input
		inp = Input(shape=[self.latent_size])

		# check dimensionality of image data
		if len(self.image_shape[:-1]) == 3:
			conv_layer = Conv3D
			upsampling_layer = UpSampling3D

			# image input
			g = Dense(4*4*4*self.latent_size)(inp)
			g = Reshape((4, 4, 4, self.latent_size))(g)

		elif len(self.image_shape[:-1]) == 2:
			conv_layer = Conv2D
			upsampling_layer = UpSampling2D

			g = Dense(8*8*self.latent_size)(inp)
			g = Reshape((8, 8, self.latent_size))(g)
		else:
			raise ValueError("Data must be 2D or 3D")

		g = self.norm_layer()(g)
		g = LeakyReLU(self.leakiness)(g)
		fil = 128
		while g.shape[1] < self.image_shape[1]:
			g = conv_layer(fil, kernel_size=3, strides=1, padding='same', kernel_initializer=init, use_bias=False)(g)
			g = self.norm_layer()(g)
			g = upsampling_layer()(g)
			g = LeakyReLU(self.leakiness)(g)
			fil = int(fil / 2)

		g = conv_layer(self.image_shape[-1], kernel_size=7, strides=1, padding='same', kernel_initializer=init, use_bias=False)(g)

		out_image = Activation(self.out_activation)(g)

		# for non-cubic 3D volumes
		crop = (out_image.shape[1:-1] - self.image_shape[:-1]) // 2
		if not np.sum(crop) == 0:
			crop = np.array([crop, crop]).T
			if len(self.image_shape[:-1]) == 3:
				out_image = Cropping3D(crop)(out_image)
			elif len(self.image_shape[:-1]) == 2:
				out_image = Cropping2D(crop)(out_image)

		# define model
		model = Model(inp, out_image, name="GAN_generator")

		plot_model(model, to_file=self.out_dir + "/generator.png", show_shapes=True, show_layer_names=True)
		return model


	# define the discriminator model
	def define_discriminator(self):
		
		if len(self.image_shape[:-1]) == 3:
			conv_layer = Conv3D
		elif len(self.image_shape[:-1]) == 2:
			conv_layer = Conv2D
		else:
			raise ValueError("need 2D or 3D data")

		# weight initialization
		init = RandomNormal(stddev=self.init_weights)

		in_image = Input(shape=[*self.image_shape, ])
		d = in_image
		fil = self.filter_d
		while d.shape[1] > 1:
			d = conv_layer(fil, kernel_size=5, strides=2, padding='same', kernel_initializer=init, use_bias=False)(d)
			d = LeakyReLU(self.leakiness)(d)
			# d = Dropout(self.dropout)(d)
			fil = int(fil * 2)

		d = Flatten()(d)
		d = Dense(512, use_bias=False)(d)
		out = Dense(1, use_bias=False)(d)

		model = Model(in_image, out)

		model.compile(loss=[self.loss], optimizer=self.opt, loss_weights=[0.5])
		plot_model(model, to_file=self. out_dir + "/discriminator.png", show_shapes=True, show_layer_names=True)
		return model

	def define_composite_model(self, g_model, d_model):
		"""This is the composite generator model with adversarial loss"""
		# ensure the model we're updating is trainable
		g_model.trainable = True
		# mark discriminator as not trainable
		d_model.trainable = False

		input_gen = Input(shape=[self.latent_size])
		gen_out = g_model(input_gen)
		output_d = d_model(gen_out)

		model = Model(input_gen, output_d)

		model.compile(loss=[self.loss], optimizer=self.opt)
		plot_model(model, to_file=self. out_dir + "/composite_model.png", show_shapes=True, show_layer_names=True)
		return model

	def define_average_model(self, d_model):
		"""This is the discriminative model used with Wasserstein loss"""
		# mark discriminator as trainable
		d_model.trainable = True

		real_in = Input(shape=[*self.image_shape, ])
		fake_in =  Input(shape=[*self.image_shape, ])

		average = RandomWeightedAverage(self.BATCH_SIZE)([real_in, fake_in])
		da = d_model(average)

		partial_gp_loss = partial(gradient_penalty_loss,
								  averaged_samples=average,
								  gradient_penalty_weight=10)
		partial_gp_loss.__name__ = 'gradient_penalty'

		model = Model([real_in, fake_in], [da])
		model.compile(loss=[partial_gp_loss], optimizer=self.opt)


		plot_model(model, to_file=self. out_dir + "/average_model.png", show_shapes=True, show_layer_names=True)
		return model

	def train_on_batch(self, trainGenerator, testGenerator):
		# determine the output square shape of the discriminator
		if len(self.d_model.outputs[0].shape[1:-1]) == 0:
			patch_shape = [1]
		else:
			patch_shape = self.d_model_A.outputs[0].shape[1:-1]

		# unpack dataset
		if isinstance(trainGenerator, (DataGenerator)):
			num_batches = len(trainGenerator)
			train = next(trainGenerator)
		elif isinstance(trainGenerator, (list, np.ndarray)):
			num_batches = len(trainGenerator) // self.BATCH_SIZE
			train = trainGenerator
		else:
			raise ValueError("Get your data in order!")

		#Training mode: 0 = segmentations, 1 = images, 2 = differences
		if self.mode > 1:
			mask, ims = train
			ims[mask[..., -1] < 1.] = 0.
		else:
			ims = train[self.mode]

		# use only desired channels
		if ims.shape[-1] > self.image_shape[-1]:
			ims = ims[..., :self.image_shape[-1]]

		# prepare image pool for fakes
		pool = list()

		# select a batch of real samples
		X_real, y_real = generate_real_samples(ims, self.BATCH_SIZE, patch_shape)
		# generate a batch of fake samples
		X_fake, y_fake = generate_fake_samples(self.g_model, ims, patch_shape)
			
		# update fakes from pool
		X_fake = update_image_pool(pool, X_fake, self.pool_size)

		# update generator
		g_loss = self.adversarial_model.train_on_batch(np.random.normal(0,1,(self.BATCH_SIZE, self.latent_size)), self.loss_weights[-1]*y_real)

		# update discriminator
		if self.loss is wasserstein_loss:
			# _, d_loss1, d_loss2, d_loss3 = self.average_model.train_on_batch([X_real, X_fake], [y_real, y_fake, np.zeros_like(y_real)])
			d_loss1 = self.d_model.train_on_batch(X_real, self.loss_weights[0] * y_real)
			d_loss2 = self.d_model.train_on_batch(X_fake, self.loss_weights[1] * y_fake)
			d_loss3 = self.average_model.train_on_batch([X_real, X_fake], [np.zeros_like(y_real)])
			self.d_loss.append([d_loss1, d_loss2, d_loss3])
		else:
			d_loss1 = self.d_model.train_on_batch(X_real, self.loss_weights[0] * y_real)
			d_loss2 = self.d_model.train_on_batch(X_fake, self.loss_weights[1] * y_fake)
			self.d_loss.append([d_loss1, d_loss2])
			d_loss3 = 0

		self.g_loss.append(g_loss)

		self.steps +=1
		# summarize performance every 50 batches
		if self.steps % 50 == 0:
			if self.loss is not wasserstein_loss:
				print("[Epoch %d] [Batch %d] [D loss real: %1.3f, fake: %1.3f] [G loss: %1.3f]" % (
							len(self.epochs),
							self.steps,
							d_loss1,
							d_loss2,
							g_loss))
			else:
				print("[Epoch %d] [Batch %d] [D loss real: %1.3f, fake: %1.3f, gradient penalty: %1.3f] [G loss: %1.3f]" % (
																					   len(self.epochs),
																					   self.steps,
																					   d_loss1,
																					   d_loss2,
																					   d_loss3,
																					   g_loss))

		#evaluate and validate the performance after each epoch, then save
		if self.steps % len(trainGenerator) == 0:
			self.save_loss()
			self.epochs.append(self.steps)
			self.visualize(X_real, X_fake)
			self.evaluate(testGenerator)
			self.save(len(self.epochs))
			np.save(self.out_dir + '/res/epochs.npy', self.epochs)

	def save_loss(self, name='training'):

		"""Plot the loss functions and save plots."""

		d_loss, g_loss = np.asarray(self.d_loss), np.asarray(self.g_loss)

		f = plt.figure(figsize=(16, 8))
		ax = f.add_subplot(1, 2, 1)
		ax.plot(np.arange(1,self.steps+1), d_loss.T[0],  c='r', label='real image loss')
		ax.plot(np.arange(1,self.steps+1), d_loss.T[1],  c='b', label='fake image loss')
		if len(self.d_loss_eval) > 1 and name == 'evaluation':
			ax.plot(np.asarray(self.epochs), np.asarray(self.d_loss_eval).T[0], 'r--')
			ax.plot(np.asarray(self.epochs), np.asarray(self.d_loss_eval).T[1], 'b--')
		if self.loss is not wasserstein_loss:
			ax.set_yscale('log')
		else:
			ax.plot(np.arange(1,self.steps+1), d_loss.T[2],  c='g', label='gradient penalty loss')
			if len(self.d_loss_eval) > 1 and name == 'evaluation':
				ax.plot(np.asarray(self.epochs), np.asarray(self.d_loss_eval).T[2], 'g--')
		ax.legend()
		ax.set_title('Discriminator loss')

		ax = f.add_subplot(1, 2, 2)
		ax.plot(np.arange(1, self.steps + 1), g_loss, 'g-')
		if len(self.g_loss_eval) > 1 and name == 'evaluation':
			ax.plot(np.asarray(self.epochs), np.asarray(self.g_loss_eval), 'g--', label='evaluation')
			ax.legend()
		if self.loss is not wasserstein_loss:
			ax.set_yscale('log')
		ax.set_title('Generator loss')

		plt.savefig(self.out_dir + '/res/' + name + '.png')
		plt.close()

		np.save(self.out_dir + '/res/d_loss.npy', d_loss)
		np.save(self.out_dir + '/res/g_loss.npy', g_loss)


	def evaluate(self, testGenerator):
		"""Plot the loss and evaluation errors."""

		if isinstance(testGenerator, (DataGenerator)):
			num_batches = len(testGenerator)
		else:
			num_batches = len(testGenerator) // self.BATCH_SIZE
			eval = testGenerator

		if len(self.d_model.outputs[0].shape[1:-1]) == 0:
			patch_shape = [1]
		else:
			patch_shape = self.d_model_A.outputs[0].shape[1:-1]


		i = 0
		d_loss1, d_loss2, d_loss3 = 0, 0, 0
		g_loss = 0
		while i < num_batches:
			i+=1
			if isinstance(testGenerator, (DataGenerator)):
				eval = next(testGenerator)

			#Training mode: 0 = segmentations, 1 = images, 2 = differences
			if self.mode > 1:
				mask, ims = eval
				ims[mask[..., -1] < 1.] = 0.
			else:
				ims = eval[self.mode]


			# use only desired channels
			if ims.shape[-1] > self.image_shape[-1]:
				ims = ims[..., :self.image_shape[-1]]

			# select a batch of real samples
			X_real, y_real = generate_real_samples(ims, self.BATCH_SIZE_EVAL, patch_shape)
			# generate a batch of fake samples
			X_fake, y_fake = generate_fake_samples(self.g_model, ims, patch_shape)

			# evaluate generator
			g_loss += self.adversarial_model.evaluate(np.random.normal(0, 1, (self.BATCH_SIZE_EVAL, self.latent_size)), self.loss_weights[-1]*y_real, verbose=0)

			# evaluate discriminator
			if self.loss is wasserstein_loss:
				d_loss1 = self.d_model.evaluate(X_real, self.loss_weights[0]*y_real, verbose=0)
				d_loss2 = self.d_model.evaluate(X_fake, self.loss_weights[1]*y_fake, verbose=0)
				d_loss3 = self.average_model.evaluate([X_real, X_fake], [np.zeros_like(y_real)], verbose=0)
			else:
				d_loss1 += self.d_model.evaluate(X_real, self.loss_weights[0]*y_real, verbose=0)
				d_loss2 += self.d_model.evaluate(X_fake, self.loss_weights[1]*y_fake, verbose=0)

		if self.loss is wasserstein_loss:
			self.d_loss_eval.append([d_loss1/num_batches, d_loss2/num_batches, d_loss3/num_batches])
		else:
			self.d_loss_eval.append([d_loss1/num_batches, d_loss2/num_batches])

		self.g_loss_eval.append(g_loss/num_batches)

		np.save(self.out_dir + '/res/d_loss_eval.npy', self.d_loss_eval)
		np.save(self.out_dir + '/res/g_loss_eval.npy', self.g_loss_eval)

		self.save_loss(name='evaluation')

	def visualize(self, image_A, image_B):
		"""Plot a volume, cycled to domain B and then back to domain A"""

		real_im = image_A
		fake_im = image_B
		if len(real_im.shape[1:-1]) > 2:
			real_im = real_im[0]
			fake_im = fake_im[0]

		num_ims = min(10, real_im.shape[0])
		fig, ax_arr = plt.subplots(num_ims, 2, figsize=(6, 3*num_ims))

		ax_arr = ax_arr.reshape((num_ims, 2))

		real_im = np.transpose(real_im, (2,1,0,3))
		fake_im = np.transpose(fake_im, (2,1,0,3))

		i = 0
		for i in range(num_ims):
			(ax1, ax2) = ax_arr[i]
			idx = real_im.shape[0] // num_ims

			ax1.imshow(real_im[i*idx, ..., 0], cmap='bone', vmin=-1, vmax=1)
			if self.image_shape[-1] > 1:
				ax1.contour(real_im[i*idx, :, :, 1], levels=[0.], colors=['r'])
			ax1.set_xticks([], [])
			ax1.set_yticks([], [])

			ax2.imshow(fake_im[i*idx, ..., 0], cmap='bone', vmin=-1, vmax=1)
			if self.image_shape[-1] > 1:
				ax2.contour(fake_im[i*idx, :, :, 1], levels=[0.], colors=['r'])
			ax2.set_xticks([], [])
			ax2.set_yticks([], [])

			if i==0:
				ax1.set_title('real image')
				ax2.set_title('fake image')


		plt.savefig(self.out_dir + '/res/output_sample' + str(len(self.epochs) - 1) + '.jpg')
		plt.close()

	def saveModel(self, model, name):  # Save a Model

		"""Save a model."""
		model.save(name + ".h5")

	def save(self, num=-1):

		"""Save the GAN's submodels individually.

        Input:
            int; identifier for a given model.
        """


		if num < 0:
			num = str(self.steps)
		else:
			num = str(num)


		self.saveModel(self.g_model, self.out_dir + "/Models/gen_" + num)
		self.saveModel(self.d_model, self.out_dir + "/Models/dis_" + num)

	def load(self, location, num=-1):
		"""Load the GAN's submodels.

		Input:
			int; identifier for the desired model.
		"""
		if num < 0:
			num = ""
		else:
			num = str(num)

		self.g_model.load_weights(location + "/Models/gen_" + num + '.h5')
		self.d_model.load_weights(location + "/Models/dis_" + num + '.h5')

		self.adversarial_model = self.define_composite_model(self.g_model, self.d_model)

		self.epochs = list(np.load(location + '/res/epochs.npy')[:int(num) + 1])
		self.steps = self.epochs[-1]
		self.d_loss = list(np.load(location + '/res/d_loss.npy')[:self.steps])
		self.d_loss_eval = list(np.load(location + '/res/d_loss_eval.npy')[:self.steps])
		self.g_loss = list(np.load(location + '/res/g_loss.npy')[:self.steps])
		self.g_loss_eval = list(np.load(location + '/res/g_loss_eval.npy')[:self.steps])


if __name__ == "__main__":

	test = trainableModel(image_shape=(32,32,1), loss='wasserstein')

	test.g_model.summary()
	test.d_model.summary()
