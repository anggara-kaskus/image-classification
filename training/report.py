import itertools
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow_hub as hub
import glob
from keras.preprocessing import image
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from time import time
from tqdm import tqdm

script_directory=os.path.dirname(__file__)
test_data_directory=os.path.join(script_directory, 'data', 'split', 'test_data')
model_directory=os.path.join(os.path.dirname(script_directory), 'models')

class_names=[]
with open(model_directory + '/labels.txt') as file:
    class_names=file.readlines()
    class_names=[line.rstrip() for line in class_names]

model=load_model(model_directory + '/saved_model.h5', custom_objects={'KerasLayer':hub.KerasLayer}, compile=False)
image_size=224
x_test=[]
y_test=[]
t0=time()
print('Class names: {}'.format(class_names))

print('Loading images...')
image_files=glob.glob(test_data_directory + "/**/*.jpg")
for image_file in tqdm(image_files):
    # Load the current image file
    image_data=image.load_img(image_file, target_size=(image_size, image_size))

    # Convert the loaded image file to a numpy array
    image_array=image.img_to_array(image_data)
    image_array /= 255

    # Add to list of test images
    x_test.append(image_array)
    # Now add answer derived from folder
    path_name=os.path.dirname(image_file)
    folder_name=os.path.basename(path_name)
    y_test.append(class_names.index(folder_name))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.get_cmap('Blues')):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm=cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Generated normalized confusion matrix")
    else:
        print('Generated confusion matrix without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt='.2f' if normalize else 'd'
    thresh=cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

print('Running predictions...')
x_test=np.array(x_test)
predictions=model.predict(x_test)
y_pred=np.argmax(predictions, axis=1)

# Compute confusion matrix
cnf_matrix=confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
fig=plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
fig.savefig(model_directory + '/confusion_matrix.png')
cr=classification_report(y_test, y_pred, target_names=class_names)
with open(model_directory + '/classification_report.txt', 'w') as the_file:
    the_file.write(cr)
print('Done in {:.3f} seconds'.format(time() - t0))
