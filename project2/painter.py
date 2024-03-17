import skimage.io as io
import numpy as np
from matplotlib import pyplot as plt
import random
import argparse


def recombine(
	im1: np.ndarray, im2: np.ndarray
) -> np.ndarray:
	"""Create new images from two images. 

	Vars:
		im1: the first image
		im2: the second image

	Returns:
		New images, choosen by first generatinga a random 
		rectangle & location smaller than the canvas size.
		Then the rectangle is slized from im1 and inserted 
		into im2, and vice-verce, and both are returned.
	"""

	# Generate slicing rectangle dimentions & location 
	width = random.randrange(1, im1.shape[1])
	height = random.randrange(1, im1.shape[0])
	top = random.randrange(1, im1.shape[1] - width)
	left = random.randrange(1, im1.shape[1] - height)

	# Create a new images
	new_im1 = np.copy(im1)
	new_im1[top:top+height, left:left+width] = im2[top:top+height, left:left+width]
	new_im2 = np.copy(im2)
	new_im2[top:top+height, left:left+width] = im1[top:top+height, left:left+width]

	# Return new images
	return [new_im1, new_im2]
	

def mutate(im: np.ndarray) -> np.ndarray:
	"""Mutate an image.

	Vars:
		im: the image to mutate.

	Returns:
		A new image, which is the same as the original,
		except that one of the colors is the image is
		globally (i.e., everywhere it occurs in the image)
		replace with a randomly chosen new color.
	"""
	# Find existing color in image and generate a new random one to replace it
	color = im[random.randrange(0, im.shape[0])][random.randrange(0, im.shape[1])]
	new_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

	# Replace the old color with the new one in the array and return it
	im[np.all(im == color, axis=2)] = new_color
	return im.astype(int)


def evaluate(im: np.ndarray):
	"""Evaluate an image.

	Vars:
		im: the image to evaluate.

	Returns:
		The value of the evaluation function on im.
		Since art is subjective, you have complete
		freedom to implement this however you like.
	"""
	# Find images with the most amount of pastel colors
	return len(np.unique(im))

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

	
	pool = [red, blue]

	for i in range(args.generations):
		gen_pool = []
		if i == 0:
			for j in range(args.pools):
				child = recombine(pool[0], pool[1])
				gen_pool += child
		else:
			#each loop gives us a new pair of parents
			for j in range(0, args.recombine*2, 2):
				#make as many children as needed
				for children in range(int(args.pools/args.recombine)):
					child = recombine(pool[j], pool[j+1])
					gen_pool.append(child)

			#make any leftover children
			for j in range(args.pools%args.recombine):
				child = recombine(pool[0], pool[1])
				gen_pool.append(child)
		
		#shuffle to reduce homogeneity in the case of equally fit children
		random.shuffle(gen_pool)
		#sort by most fit
		gen_pool.sort(reverse=True, key=evaluate)
		pool = gen_pool
		
	# View the image
	plt.imshow(pool[0])
	plt.show() 
	

	
if __name__ == '__main__':
	main()