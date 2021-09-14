# Kaskus Image Classification
Image classification using Keras and Convolutional Neural Network (CNN). Training script and model architecture is based from [nsfw_model](https://github.com/GantMan/nsfw_model/) with some modification for setting default folder values.

## Current Status
The pre-compiled model in `models/` folder is trained with following classification:
- safe (various neutral things) : 38,608 images
- sensitive (nudity, violence) : 16,545 images

However, you can train any image classification with your own defined classes using this code. Just follow the instruction.

### Model architecture
<p align="center">
<img alt="Model architecture" width="500" src="https://github.com/anggara-kaskus/nsfw-model/blob/main/models/model.png?raw=true">
</p>

### Dataset Sources

This dataset is collected from various sources with total of ~10GB of image files:
- Some neutral and NSFW images fetched using [nsfw_data_scraper](https://github.com/alex000kim/nsfw_data_scraper/)
- Some NSFW images fetched using [nsfw_data_source_urls](https://github.com/EBazarov/nsfw_data_source_urls)
- Some violence images scraped from subforum at Kaskus

Unfortunately, the accuracy of 'sensitive' class is quite low and may be caused by following reasons:
- The number of images is not balanced between classes
- The dataset is very noisy (ie: some safe images included in sensitive folder, or vice versa)

### Classification Report
From this report, although overall accuracy reach 91%, we can see that classification score for 'sensitive' images needs improvement.

```
              precision    recall  f1-score   support

        safe       0.92      0.97      0.94     16545
   sensitive       0.82      0.64      0.72      3813

    accuracy                           0.91     20358
   macro avg       0.87      0.80      0.83     20358
weighted avg       0.90      0.91      0.90     20358
```
<img alt="Confusion Matrix" width="500" src="https://github.com/anggara-kaskus/nsfw-model/blob/main/models/confusion_matrix.png?raw=true">

We can do fine-tuning later with better dataset.

## Setup
### Prerequisites
- Python 3.8+
- pip
- Jupyter Lab or Jupyter Notebook (optional)

### Installation
- Clone this repo
```sh
git clone git@github.com:anggara-kaskus/nsfw-model.git
cd nsfw-model
```
- Install dependencies
```sh
pip install -r requirements.txt
```

## Training

### Prepare sample data
Put your raw sample data in `data/raw/` folder, with subdirectory with class name that contains corresponding images.
For example if you want to classify 'nonsensitive' and 'sensitive' images, 

```
data/
  └─ raw/
     ├─ nonsensitive/
     │  ├─ flowers/
     │  |  ├─ flower_1.jpg
     │  |  ├─ flower_2.jpg
     │  |  ├─ flower_3.jpg
     │  |  └─ ...
     │  | 
     │  └─ animals/
     │     ├─ cat_1.jpg
     │     ├─ dog_2.jpg
     │     ├─ bird_3.jpg
     │     └─ ...
     │
     └─ sensitive/
        ├─ porn/
        |  ├─ porn_1.jpg
        |  ├─ porn_2.jpg
        |  ├─ porn_3.jpg
        |  └─ ...
        | 
        └─ gore/
           ├─ gore_1.jpg
           ├─ gore_2.jpg
           ├─ gore_3.jpg
           └─ ...
```

> Note: If you want to get sample data for testing, you can download one from [Tensorflow Dataset Collection
](https://www.tensorflow.org/datasets/catalog/overview#image_classification)

### Preprocess image data
Run following command to flatten all subdirectory contents of each classes into one folder:

```sh
python training/preprocess.py
```
Image will be formatted as RGB JPEG. This command will also check if there is corrupted or invalid image, it will be moved into `invalid/` folder.
Optionally, you can also set `crop=True` in file `preprocess.py` to crop images to square during process.

The result of this process will looks like this:
```
data/
  ├─ invalid/
  │  ├─ nonsensitive/
  │  │  ├─ corrupted_1.jpg
  │  │  └─ ...
  │  │
  │  └─ sensitive/
  │     ├─ invalid_1.jpg
  │     └─ ...
  │
  └─ processed/
     ├─ nonsensitive/
     │  ├─ flower_1.jpg
     │  ├─ flower_2.jpg
     │  ├─ flower_3.jpg
     │  ├─ cat_1.jpg
     │  ├─ dog_2.jpg
     │  ├─ bird_3.jpg
     │  └─ ...
     │
     └─ sensitive/
        ├─ porn_1.jpg
        ├─ porn_2.jpg
        ├─ porn_3.jpg
        ├─ gore_1.jpg
        ├─ gore_2.jpg
        ├─ gore_3.jpg
        └─ ...
```
### Split training and test data

Split files for training and test data<sup>*</sup>:

```sh
python training/split.py
```

The default ratio for splitting is 70:30 for training and test data.
To change it, please set `test_size=0.3` to desired value in `split.py`.

> _<sup>*</sup>) Not to be confused with validation data.
> Validation set is generated internally during training session, while test data is used for assessment of your model_

### Train the model
Run the training. For more detailed parameters, please consult [nsfw_model](https://github.com/GantMan/nsfw_model) documentation.

```sh
python training/train.py
```

Generated model files will be saved to folder `models/`

### Training Report
To generate graphs and prediction report, run:

```sh
python training/report.py
```

This will generate `models/confusion_matrix.png` and `models/classification_report.txt`

## Running classification on image

### Run one time prediction
Classify single image or multiple images in a directory:

```sh
python prediction/predict.py --image_source /path/to/image.jpg
```

or multiple images in a directory:

```sh
python prediction/predict.py --image_source /path/to/directory/image/
```

Output sample:
```
{
  "/path/to/directory/image/14jp9.jpg": {
    "safe": 0.3952472507953644,
    "sensitive": 0.604752779006958
  },
  "/path/to/directory/image/sensitive_3578.jpg": {
    "safe": 0.7168499827384949,
    "sensitive": 0.28314998745918274
  }
} 
```

### Run as a service

To start TCP service, run:
```sh
python prediction/server.py
```

The server will listen to port *1235*

You can connect as telnet client and input full path of target image (in server's storage)
```
$ gtelnet localhost 1235
Trying 127.0.0.1...
Connected to localhost.
Escape character is '^]'.


Welcome! Enter image path to scan
# Path: /path/to/image/directory/
> Scanning: /path/to/image/directory/
> Result for : /path/to/image/directory/

{
  "/path/to/image/directory/14jp9.jpg": {
    "safe": 0.3952472507953644,
    "sensitive": 0.604752779006958
  },
  "/path/to/image/directory/sensitive_3578.jpg": {
    "safe": 0.7168499827384949,
    "sensitive": 0.28314998745918274
  },
  "__time__": 0.631289
}

# Path: /path/to/image/directory/14jp9.jpg
> Scanning: /path/to/image/directory/14jp9.jpg
> Result for : /path/to/image/directory/14jp9.jpg

{
  "/path/to/image/directory/14jp9.jpg": {
    "safe": 0.3952471911907196,
    "sensitive": 0.604752779006958
  },
  "__time__": 0.089046
}

```

## Debugging
If you want to debug or just playing around, you can run:

```sh
cd notebooks
jupyter-lab &> logs/log.txt & # or jupyter notebook
```
