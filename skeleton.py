	#import os
import matplotlib.pyplot as plt
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib 
import numpy as np
import pickle
import gzip
import scipy.ndimage
import skimage 
from skimage import io, color
from skimage.transform import resize
from tf_unet import unet, util, image_util
from tf_unet import unet, util, image_util

class Segment(object):

		path = ""
		net = None
		trainer = None
		count = 0
		def __init__(self, data_dir):
			"""
			data_directory : path like /home/rajat/nnproj/dataset/
			includes the dataset folder with '/'
			Initialize all your variables here
			"""
			self.path = data_dir
			self.net = unet.Unet(layers=3, features_root=64, channels=1, n_class=2)
			self.trainer = unet.Trainer(self.net)
			self.count = 0
			#pickle.dump(self, open("project.pickle", 'wb'))



		def train(self):
			"""
			Trains the model on data given in path/train.csv

			No return expected
			"""

			#data_provider = image_util.ImageDataProvider(search_path = path + "*", data_suffix="train.jpg", mask_suffix="mask.jpg")
			#net = unet.Unet(layers=3, features_root=64, channels=1, n_class=2)


		def get_mask(self, image):
			"""
			image : a variable resolution RGB image in the form of a numpy array

			return: A list of lists with the same 2d size as of the input image with either 0 or 1 as each entry

			"""
			sha = image.shape
			# img = color.rgb2gray(image)
			# img = np.asarray(img)
			# row_pad = 572 - (sha[0] % 572)
			# col_pad = 572 - (sha[1] % 572)
			# img1 = np.pad(img, ((0, row_pad), (0, col_pad)), 'constant', constant_values=(0))
			# sha1 = img1.shape
			# final_image = np.zeros((572,572), dtype=int)
			# rows = sha1[0] / 572
			# cols = sha1[1] / 572
			# fin_image = np.zeros((1, sha1[1]))
			# data_provider = image_util.ImageDataProvider(search_path = "test_images/*", data_suffix="train.jpg", mask_suffix="mask.jpg")
			# for i in range(0, int(rows)):
			# 	final_image = np.zeros((572, 572), dtype=int)
			# 	for j in range(0, int(cols)):
			# 		img2 = img1[i*572: (i+1)*572, j*572: (j+1)*572]
			# 		randomPath = "./test_images/" + "1_train.jpg" 
			# 		randomPath2 = "./test_images/" + "1_mask.jpg"
			# 		io.imsave(randomPath, img2)
			# 		io.imsave(randomPath2, img2)
			# 		self.count = self.count + 1
			# 		x_test, y_test = data_provider(1)
			# 		prediction = self.net.predict("./temp/model.cpkt", x_test)
			# 		mask = prediction[0,...,1] > 0.5
			# 		mask1 = resize(mask, (572, 572))
			# 		temp = np.asarray(mask1)
			# 		final_image = np.append(final_image, temp, axis=1)
			# 		#print(final_image.shape)
			# 		mask1 = scipy.ndimage.binary_dilation(mask1).astype(dtype=int)
			# 	fin_image = np.append(fin_image, final_image[:, 572:], axis=0)
			
			# #print(fin_image.shape)
			# mask1 = fin_image[1:sha[0]+1,:sha[1]]
			# mask1 = scipy.ndimage.binary_dilation(mask1).astype(dtype=int)
			# mask1 = scipy.ndimage.binary_dilation(mask1).astype(dtype=int)
			# return mask1
			img = resize(color.rgb2gray(image), (572, 572))
			randomPath = "./test_images/" + "1_train.jpg" 
			randomPath2 = "./test_images/" + "1_mask.jpg"
			io.imsave(randomPath, img)
			io.imsave(randomPath2, img)
			self.count = self.count + 1
			data_provider = image_util.ImageDataProvider(search_path = "test_images/*", data_suffix="train.jpg", mask_suffix="mask.jpg")
			x_test, y_test = data_provider(1)
			#os.remove(randomPath)
			prediction = self.net.predict("./temp/model.cpkt", x_test)
			print(prediction.shape)
			#prediction = np.resize(prediction, (sha[0], sha[1]))
			#prediction = scipy.ndimage.binary_dilation(prediction).astype(dtype=int)
			mask = prediction[0,...,1] > 0.5
			mask1 = resize(mask, (sha[0], sha[1]))
			#for i in range(5):
			mask1 = scipy.ndimage.binary_dilation(mask1).astype(dtype=int)
			#mask1 = scipy.ndimage.binary_dilation(mask1).astype(dtype=int)
			#mask1 = scipy.ndimage.binary_dilation(mask1).astype(dtype=int)
			#mask1 = np.ndarray.astype(mask1, dtype="int")
			#prediction = 1 * (prediction>0.2)
			#prediction = np.resize(prediction, (sha[0], sha[1]))
			return mask1


		def save_model(self, **params):

			# file_name = params['name']
			# pickle.dump(self, gzip.open(file_name, 'wb'))

			"""
			saves model on the disk

			no return expected
			"""

		@staticmethod
		def load_model(**params):

			# # file_name = params['name']
			# # return pickle.load(gzip.open(file_name, 'rb'))
			# file_name = 'model.cpkt'
			# file = open("model.cpkt",'r')			
			# return pickle.load(file)
			# """
			#     returns a pre-trained instance of Segment class
			# """
			#return pickle.load(open("project.pickle", 'rb'))
			return Segment("")
			if __name__ == "__main__":
			# obj = Segment('dataset/')
			# obj.train()
			# obj.save_model(name="segment.gz")
				pass
