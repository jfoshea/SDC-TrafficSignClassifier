# Traffic Sign Classifier 

## Overview
Develop a Deep Leaning Network to classify traffic signs. To accomplish this a Convolutional Neural Network (CNN) will be developed to classify traffic signs. Specifically the CNN will focus on German traffic signs. [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
This project develops a Convolutional Neural Network and uses many aspects of OpenCV, python, numpy, and matplotlib. The CNN code is executed within in a jupyter notebook environment. The requirements are:
- Explain Convolutional Neural Network Architecure.
- Outline any preprocessing techniques used (normalization, equalization, rgb to grayscale, etc)
- Outline any balancing techniques used on the number of examples per label (some have more than others).
- Evaluate if the Neural Network is over or underfitting?)
- Generate fake data
- The Neural Network needs to have a validation set accuracy >= 0.93

## Installing and Running the Pipeline
The following steps are used to run the pipeline:
1. Install jupyter notebook environment and packages
    ```
    https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/doc/configure_via_anaconda.md
    ```
2. Clone the SDC-TrafficSignClassifier git repository
    ```  
    $  git clone https://github.com/jfoshea/SDC-TrafficSignClassifier.git
    ```

3. enable cardnd-term1 virtualenv
    ```
    $ source activate carnd-term1
    ```
4. Run the Pipeline 
    ```
    $ jupyter notebook TrafficSignClassifier.ipynb
    ```

The random sign images are located in `random_sign_images` directory.

## Writeup 
A detailed writeup of the classifier and challenges are located here [writeup] (https://github.com/jfoshea/SDC-TrafficSignClassifier/blob/master/writeup.md)

