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
	width = random.randrange(10, im1.shape[1])
	height = random.randrange(10, im1.shape[0])
	left = random.randrange(0, im1.shape[1] - width)
	top = random.randrange(0, im1.shape[0] - height)

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
	# Define the range of RGB values that represent purple
	purple_lower = np.array([100, 0, 100])  # Adjust lower bound as needed
	purple_upper = np.array([255, 100, 255])  # Adjust upper bound as needed

	# Filter pixels that fall within the purple range
	purple_mask = np.all((im >= purple_lower) & (im <= purple_upper), axis=-1)

	# Count the number of purple pixels
	num_purple_pixels = np.sum(purple_mask)

	return num_purple_pixels
	
	'''
	# Calculate the difference between each RGB component and the maximum component
	diff_max_r = np.abs(im[:, :, 0] - np.max(im, axis=2))
	diff_max_g = np.abs(im[:, :, 1] - np.max(im, axis=2))
	diff_max_b = np.abs(im[:, :, 2] - np.max(im, axis=2))

	# Calculate the maximum difference across RGB components for each pixel
	max_diff = np.max(np.stack((diff_max_r, diff_max_g, diff_max_b), axis=2), axis=2)

	# Determine pastel pixels where the maximum difference is below a threshold
	pastel_pixels = max_diff < 70

	# Count the number of pastel pixels
	return np.sum(pastel_pixels)
	'''
	
	# Count the number of green pixels
	green_pixels = np.sum(im[:, :, 1] > im[:, :, 0])
	green_pixels = np.sum(im[:, :, 1] > im[:, :, 2])

    # Count the number of non-green pixels
	non_green_pixels = np.sum(im[:, :, 1] <= im[:, :, 0])
	non_green_pixels = np.sum(im[:, :, 1] <= im[:, :, 2])

    # Calculate fitness based on the difference between green and non-green pixels
	fitness = green_pixels - non_green_pixels
	if fitness == None:
		fitness = 0
	return fitness
	
	'''
	# Flatten the image array to a 2D array of RGB values
	flattened_im = im.reshape(-1, im.shape[-1])

	# Filter colors where the red channel is greater than the other channels
	red_colors = flattened_im[(flattened_im[:, 0] > flattened_im[:, 1]) & (flattened_im[:, 0] > flattened_im[:, 2])]

	# Find unique red colors
	unique_red_colors = np.unique(red_colors, axis=0)

	# Count the number of unique red colors
	return len(unique_red_colors)
	'''
	'''
	# Count the number of red pixels
	red_pixels = np.sum(im[:, :, 0] > im[:, :, 1])
	red_pixels = np.sum(im[:, :, 0] > im[:, :, 2])

    # Count the number of non-red pixels
	non_red_pixels = np.sum(im[:, :, 0] <= im[:, :, 1])
	non_red_pixels = np.sum(im[:, :, 0] <= im[:, :, 2])

    # Calculate fitness based on the difference between red and non-red pixels
	fitness = red_pixels - non_red_pixels
	if fitness == None:
		fitness = 0
	return fitness
	'''
	'''
	# Find images with the most amount of red colors
	fitness = 0
	# Flatten the bitmap to a 2D array of RGB values
	flattened_im = im.reshape(-1, im.shape[-1])
	colors = np.unique(flattened_im, axis = 0)
	for color in colors:
		if color[0] > color [1] and color[0] > color [2]:
			fitness += 1
		else:
			fitness -= 1
	return fitness
	'''
	'''
		# Define threshold for pastel colors (lightness)
		lightness_threshold = 0.7
		
		# Calculate lightness
		lightness = (max(color) + min(color)) / 2
		print(lightness)
		
		# Check if the color is pastel based on lightness
		if lightness < lightness_threshold:
			fitness += 1
		'''



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

	# Generate the pool of red and blue images 
	red = np.zeros((400,800,3))
	red[:,:,0] = 255
	blue = np.zeros((400,800,3))
	blue[:,:,2] = 255
	pool = [red, blue]

	# Genetic algorithm loop
	for generation in range(args.generations):
		# Randomise pool order in case children have equal fitness
		random.shuffle(pool)
		# Sort the pool based on evaluations (more fit first)
		pool.sort(reverse=True, key=evaluate)

		# Generate new pool for recombined images
		new_pool = []

		# Recombine
		for i in range(args.recombine):
			# Randomly select two parents from the pool
			parent1, parent2 = random.sample(pool, 2)
			# Perform recombination
			children = recombine(parent1, parent2)
			new_pool.extend(children)

		# Mutate
		for i in range(len(new_pool)):
			if random.random() < args.mutation:
				new_pool[i] = mutate(new_pool[i])

		# Merge current pool and new pool
		pool.extend(new_pool)

		# Select top pools
		pool = sorted(pool, key=evaluate, reverse=True)[:args.pools]
		
	# Save the top 3 images
	plt.imsave("art1.tiff", pool[0]/255)
	plt.imsave("art2.tiff", pool[1]/255)
	plt.imsave("art3.tiff", pool[2]/255)

	# View the top 3 images
	plt.imshow(pool[0])
	plt.show() 
	plt.imshow(pool[1])
	plt.show() 
	plt.imshow(pool[2])
	plt.show() 

	
if __name__ == '__main__':
	main()