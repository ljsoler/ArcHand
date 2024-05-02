#### This is a pytorch repository with the implementation of Angular Margin for hand recognition (ArcHand):

## Requierements ##
- Python 3.8+
- pytorch-lightning==2.1.0
- torch==2.1.0
- pyeer

<hr/>

## ArcHand training ##
1. Download the training and test set of [HaGrid database](https://github.com/hukenovs/hagrid) and crop the hand images similar to the ones in the example.
2. Organise the folder in the following format:

    .
    ├── train                           # Training set folder
    │   ├── SubjectID-1                 # Subject ID
    │       ├── call                    # Gestures
    │           ├── right               # right hand 
    │               ├── images1.jpg     # images
    │               ├── images1.jpg             
    │               └── ...             # etc.
    │           ├── left                # left hand
    │               └── ...             # images
    │           ├── ...                 # More gestures
    │   ├── SubjectID-2                 # More subjects
    │       └── ...                     # etc.
    ├── test                            # Test set folder
    │   └── ...                         # etc.       

    ├── ...
    ├── docs                    # Documentation files (alternatively `doc`)
    │   ├── TOC.md              # Table of contents
    │   ├── faq.md              # Frequently asked questions
    │   ├── misc.md             # Miscellaneous information
    │   ├── usage.md            # Getting started guide
    │   └── ...                 # etc.
    └── ...
2. run: bash run_training.sh hagrid-folder-path model-output-folder

<hr/>

## ArcHand testing ##
2. run: bash run_test.py hagrid-folder-path output-folder model-weights

## Examples ##

openvino_infer.ipynb contains a example of algorithm usability using OpenVino inference optimisation. 

<hr/>

## Pre-trained Models ##

Pre-trained model of EfficientNet for the right hand can be found in ./models 

<hr/>

## Citation ##
If you use any of the code provided in this repository or the models provided, please cite the following paper:
```
```