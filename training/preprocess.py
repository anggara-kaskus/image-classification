import os
import math

from glob import glob
from time import time

from PIL import Image
from tqdm import tqdm


script_directory=os.path.dirname(__file__)

data_directory=os.path.join(script_directory, "data")
raw_images_directory=os.path.join(data_directory, "raw")
processed_images_directory=os.path.join(data_directory, "processed")
invalid_images_directory=os.path.join(data_directory, "invalid")

crop=False

def get_class(directory):
	classes = [x for x in os.listdir(directory) if not x.startswith(".")]
	return classes

def preprocess(data_directory, raw_images_directory, save_directory):
	t0 = time()
	if not os.path.exists(data_directory):
		print("Directory not found!")
		return

	# Setup and Create save directory if not exists

	if not os.path.exists(save_directory):
		os.makedirs(save_directory)

	classes = get_class(raw_images_directory)
	print("{} classes found: {}".format(len(classes), classes))

	for cls in classes:
		class_image_directory = os.path.join(raw_images_directory, cls)
		list_image_directories = [
			x for x in os.listdir(
				class_image_directory
			) if not x.startswith(".")
		]

		if len(list_image_directories) == 0:
			print("There is no images directory on {} image classes!".format(cls))
			continue

		print("Found {} image sub directory on {} classes!".format(len(list_image_directories), cls))

		save_cls_directory = os.path.join(save_directory, cls)
		if not os.path.exists(save_cls_directory):
			os.mkdir(save_cls_directory)
			os.mkdir(os.path.join(invalid_images_directory, cls))

		idx = 1
		for sub_image_directory in list_image_directories:
			sub_image_directory = os.path.join(class_image_directory, sub_image_directory)

			for image_filename in tqdm(glob(pathname=sub_image_directory+"/*")):
				try:
					image = Image.open(image_filename)
				except IOError:
					print("Cannot open file {}".format(image_filename))
					os.rename(image_filename, data_directory + '/invalid/' + cls + '/' + os.path.basename(image_filename))
					continue

				try:
					image = image.convert("RGB")
				except OSError:
					print("Corrupted image: {}".format(image_filename))
					os.rename(image_filename, data_directory + '/invalid/' + cls + '/' + os.path.basename(image_filename))
					continue
					

				# get the image size
				w, h = image.size

				# save the image
				save_filename = os.path.join(
					save_cls_directory,
					"{}_{}.jpg".format(cls, idx)
				)

				# crop the image into a same size
				if crop:
					if w < h:
						n = math.floor((h - w) / 2)
						image = image.crop((0, n, w, n + w))

					elif h < w:
						n = math.floor((w - h) / 2)
						image = image.crop((n, 0, n + h, h))

				image.save(save_filename, "JPEG")
				idx += 1

	print("Preprocess images done in {:.3f}s".format(time() - t0))

preprocess(
	data_directory=data_directory,
	raw_images_directory=raw_images_directory,
	save_directory=processed_images_directory
)