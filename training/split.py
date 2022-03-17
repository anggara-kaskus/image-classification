import os
from glob import glob
from time import time

import numpy as np
from PIL import Image
from tqdm import tqdm

script_directory=os.path.dirname(__file__)

data_directory=os.path.join(script_directory, "data")
processed_images_directory=os.path.join(data_directory, "processed")
split_image_directory=os.path.join(data_directory, "split")

test_size=0.3

def get_class(directory):
    classes=[x for x in os.listdir(directory) if not x.startswith(".")]
    return classes

def progression_pattern(total_data_count, test_size):
    total_idx=[x for x in range(total_data_count)]
    test_data_count=int(np.multiply(total_data_count, test_size))
    training_data_count=total_data_count - test_data_count
    progression_range=int(np.around(test_data_count))
    test_idx=np.linspace(
        start=0,
        stop=total_data_count - 1,
        num=progression_range,
        dtype=np.int
    )

    training_idx=list(set(total_idx) - set(test_idx))
    return training_idx, test_idx

def get_data(image_directory, cls):
    if not os.path.exists(image_directory):
        print("Folder not found!")
        return

    list_image_filename=[]
    class_image_directory=os.path.join(image_directory, cls)
    for image_filename in tqdm(glob(pathname=class_image_directory + "/*")):
        list_image_filename.append(image_filename)

    return list_image_filename

def shuffle_data(data, test_size):
    total_data_count=len(data)
    training_idx, test_idx=progression_pattern(
        total_data_count, test_size
    )
    training_data=np.array(data)[training_idx]
    testing_data=np.array(data)[test_idx]
    return list(training_data), list(testing_data)

def save(images_filename, save_directory, cls):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    cls_save_directory=os.path.join(save_directory, cls)
    if not os.path.exists(cls_save_directory):
        os.makedirs(cls_save_directory)

    for image_filename in tqdm(images_filename):
        save_image_filename=os.path.basename(image_filename)
        path=os.path.join(cls_save_directory, save_image_filename)
        try:
            image=Image.open(image_filename)
        except IOError:
            print("Cannot open image: {}".format(image_filename))
            continue
        image.save(path, "JPEG")

def main():
    print("Starting...")
    classes=get_class(processed_images_directory)

    print("Found {} classes: {}".format(len(classes), classes))

    for class_name in classes:
        t0=time()
        image_filename=get_data(processed_images_directory, class_name)
        training_data, test_data=shuffle_data(
            image_filename,
            test_size
        )
        print("\nSplitting '{}' data".format(class_name))
        print("Get {} training data and {} testing data".format(len(training_data), len(test_data)))
        training_data_directory=os.path.join(split_image_directory, "training_data")
        save(training_data, training_data_directory, class_name)
        test_data_directory=os.path.join(split_image_directory, "test_data")
        save(test_data, test_data_directory, class_name)

        print("Done saving data into two folder for training and testing")
        print("Process done in {:.3f}s".format(time() - t0))

if __name__ == "__main__":
    main()
