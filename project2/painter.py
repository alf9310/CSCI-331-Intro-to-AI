import skimage.io as io
import numpy as np
from matplotlib import pyplot as plt
import random
import argparse


def recombine(
	im1: np.ndarray, im2: np.ndarray
) -> np.ndarray:
	"""Create a new image from two images. 

	Vars:
		im1: the first image
		im2: the second image

	Returns:
		A new image, chosen by first, randomly choosing
		between the horizontal or vertical orientation,
		and then slicing each image into two pieces along
		a randomly-chosen vertical or horizontal line.
	"""

def mutate(im: np.ndarray) -> np.ndarray:
	"""Mutate an image.

	Vars:
		im: the image to mutate.

	Returns:
		A new image, which is the same as the original,
		except that on of the colors is the image is
		globally (i.e., everywhere it occurs in the image)
		replace with a randomly chosen new color.
	"""


def evaluate(im: np.ndarray):
	"""Evaluate an image.

	Vars:
		im: the image to evaluate.

	Returns:
		The value of the evaluation function on im.
		Since art is subjective, you have complete
		freedom to implement this however you like.
	"""
def main():
	parser = argparse.ArgumentParser(
    	prog='painter',
    	description='creates paintings according to a genetic algorithm'
	)

	parser.add_argument('-g', '--generations', default=100, help="The number of generations to run", type=int)
	parser.add_argument('-p', '--pools', default=10, help="The size of the pool", type=int)
	parser.add_argument('-m', '--mutation', default=.2, help="The chance of a mutation", type=float)
	parser.add_argument('-r', '--recombine', default = 2, help="The number of pairs to recombine in each generation", type=int)
	args = parser.parse_args()

	red = np.zeros((400,800,3))
	red[:,:,0] = 255
	plt.imsave("red.tiff", red/255)

	blue = np.zeros((400,800,3))
	blue[:,:,2] = 255
	# uncomment the lines below to view the image
	#plt.imshow(blue)
	#plt.show() 

	
if __name__ == '__main__':
	main()

