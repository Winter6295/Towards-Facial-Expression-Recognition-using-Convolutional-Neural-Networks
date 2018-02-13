
%%%%%%%%%%%%%%%%%%%%%%%%%%%% INFORMATION ABOUT THE DATASET %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


The submission folder consists of the dataset FER 2013 obtained from Kaggle's Facial Expression Recognition challenge 2013
The dataset fer2013.csv consists of a total 32298 images, divided into 27809 images as training and 3589 images as testing, each labelled with one of the seven emotion categories: Anger, Disgust, Fear, Happy, Sad, Surprise and Neutral
We divide up the training set into two parts:
1) 27000 training images
2) 1709 validation images


%%%%%%%%%%%%%%%%%%%%%%%%%% PYTHON DEPENDENCIES REQUIRED %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

In order to run the codes the machine has to be equipped with Python 2.7.14 with the following dependencies:
pandas==0.21.0
numpy==1.13.3
sklearn==0.0
itertools (should be a built-in module installed along with the Python package)
matplotlib==2.1.0
seaborn==0.8.1
tensorflow==1.4

If not installed, do $ pip install <dependency_name>


%%%%%%%%%%%%%%%%%%%%%%% INSTRUCTIONS TO INSTALL TENSORFLOW IF NOT INSTALLED PREVIOUSLY %%%%%%%%%%%%%%%%%%%%


Ensure pip version 8.1 or above is installed in your machine. If not, please issue this command:
$ sudo easy_install --upgrade pip

You can install tensorflow by invoking the following command:
$ pip install tensorflow

If this step fails, install the latest version of tensorflow by issuing command in the following format:
$ sudo pip  install --upgrade tfBinaryURL 		:where tfBinaryURL identifies the URL of the TensorFlow Python package. The appropriate value of tfBinaryURL depends on the operating system and Python version. For Mac OS and Python 2.7 issue the following command:

$ sudo pip install --upgrade \
https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.4.0-py2-none-any.whl

For validating the installation:

Invoke python from your shell as follows:
$ python

Enter the following short program inside the python interactive shell:
# Python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
If the system outputs the following, then Tensorflow is working properly:

Hello, TensorFlow!



%%%%%%%%%%%%%%%%%%%%% INSTRUCTIONS TO RUN THE CODE %%%%%%%%%%%%%%%%%%%%%%%%%%


The folder consists of codes for training our baseline model, the best 2 layer CNN and the best 3 layer CNN model
We have provided the .meta files for each of our models, that contains all the saved graph structure, also the data files that contain saved values of all learnable variables

Run the code test_3_layer.py to evaluate the test accuracy and confusion matrix for our 3 layer CNN model
$ python test_3_layer.py

Run the code test_2_layer.py to evaluate the test accuracy and confusion matrix for our 2 layer CNN model
$ python test_2_layer.py

Run the code test_baseline.py to evaluate the test accuracy and confusion matrix for our baseline model
$ python test_baseline.py



Note: Once the confusion matrix is displayed, please save the figure and then close the window. Only then the codes will complete their execution. This happens because the plt.show() function is blocking in nature. The overall test accuracy will be displayed in the command prompt.




In order to evaluate the training and validation accuracy:

Before running the following codes, please make sure the graph is saved in right directory (same as the test codes)
Depending upon Windows or Mac, use "\" or "/" respectively


Run the code train_baseline.py for our baseline model----------(use "\" or "/" respectively in line 13 and 241)
$ python train_baseline.py

Run the code train_2_layer.py for our 2 layer CNN model--------(use "\" or "/" respectively in line 13 and 291)
$ python train_2_layer.py

Run the code train_3_layer.py for our 3 layer CNN model--------(use "\" or "/" respectively in line 13 and 296)
$ python train_3_layer

The above codes also provide the plots of training and validation accuracy across 3000 iterations.

Note: Once the above plot is displayed, please save the figure and then close the window. Only then the codes will complete their execution. This happens because the plt.show() function is blocking in nature. The overall validation accuracy will be displayed in the command prompt and the corresponding graph structure along with all the learned variables will save in the same folder.

Note: Since the weights are initialised randomly, for every run of the above training codes, the validation accuracy will be slightly different.





















