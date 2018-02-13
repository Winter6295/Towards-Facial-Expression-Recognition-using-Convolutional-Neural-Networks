import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools 

# --------------------------- Data Extraction -------------------------

data = pd.read_csv("fer2013/fer2013.csv")
train_data = data[data.Usage == "Training"]
pixel_values = train_data.pixels.str.split(" ").tolist()
pixel_values = pd.DataFrame(pixel_values, dtype = int)
images = pixel_values.values
images = images.astype(np.float)

# show sample images from the dataset

# def show(img):
# 	im = img.reshape(48,48)
# 	plt.imshow(im, cmap='gray')
# 	plt.pause(10)
	#plt.savefig('Sample Image.jpg')

# show(images[6])


# -------------------------  Data Preprocessing -------------------------

images = images - images.mean(axis=1).reshape(-1,1)
images = np.multiply(images,100.0/255.0)
each_pixel_mean = images.mean(axis=0)
each_pixel_std = np.std(images, axis=0)
images = np.divide(np.subtract(images, each_pixel_mean),each_pixel_std)

# print images.shape			(28709x2304)
image_pixels = images.shape[1]
# print "Flat pixel values is %d" %(image_pixels)

image_width = image_height = np.ceil(np.sqrt(image_pixels)).astype(np.uint8)    # 48
labels_flat = train_data["emotion"].values.ravel()
labels_count = np.unique(labels_flat).shape[0]
#print "Number of unique facial expressions is %d " %labels_count

# One hot encoding of labels
def one_hot(all_labels,num_classes):
	num_labels = all_labels.shape[0]
	index_offset = np.arange(num_labels) * num_classes
	labels_one_hot = np.zeros((num_labels, num_classes))			# 28709 x 7
	labels_one_hot.flat[index_offset + all_labels.ravel()] = 1
	return labels_one_hot

labels = one_hot(labels_flat,labels_count)
labels = labels.astype(np.uint8)

# split data into training and validation
Validation_size = 1709
validation_images = images[:Validation_size]
validation_labels = labels[:Validation_size]
train_images = images[Validation_size:]
train_labels = labels[Validation_size:]

print ("The number of final training and validation images are %d %d" %(len(train_images), len(validation_images)))



# ----------------------- Tensorflow CNN Model -------------------------

# weight initialisation
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=1e-4)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1,shape=shape)
	return tf.Variable(initial)

# define convolution
def conv2d(x,W,padd):
	return tf.nn.conv2d(x,W,strides = [1,1,1,1], padding=padd)

# define pooling
def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize = [1,2,2,1],strides=[1,2,2,1],padding='SAME')

# input and output of NN

# images
x = tf.placeholder('float', shape=[None, image_pixels], name="x")		# (:,2304)
# labels
y_ = tf.placeholder('float', shape=[None, labels_count], name="y")	# (:,7)

# first convolutional layer 64
W_conv1 = weight_variable([5,5,1,64])
b_conv1 = bias_variable([64])
image = tf.reshape(x, [-1,image_width,image_height,1])
h_conv1 = tf.nn.relu(conv2d(image,W_conv1,"SAME") + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# local layer weight initialization
def local_weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.04)
	return tf.Variable(initial)

def local_bias_variable(shape):
	initial = tf.constant(0.0, shape=shape)
	return tf.Variable(initial)
	

# densely (fully) connected local layer 3
W_fc1 = local_weight_variable([24 * 24 * 64, 4096])
b_fc1 = local_bias_variable([4096])
h_pool2_flat = tf.reshape(h_pool1, [-1, 24 * 24 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# readout layer (softmax)
W_fc3 = weight_variable([4096, labels_count])
b_fc3 = bias_variable([labels_count])
y = tf.nn.softmax(tf.matmul(h_fc1, W_fc3) + b_fc3)


# Learning Rate
learn_rate = 1e-4

# Cost Function
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# Optimization (Adam)
train = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)

# Evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))		# boolean output
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))		# cast to float values
predict = tf.argmax(y,1)
tf.add_to_collection("predict", predict)

train_iter = 3000
batch_size = 50

epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]	# 27000

# feed train data in batches
def next_batch(batch_size):

	global train_images
	global train_labels
	global index_in_epoch
	global epochs_completed

	start = index_in_epoch
	index_in_epoch = index_in_epoch + batch_size

	# when all training images are used once, reorder randomly
	if index_in_epoch > num_examples:

		# epochs finished
		epochs_completed = epochs_completed+1
		# shuffle the data
		perm = np.arange(num_examples)
		np.random.shuffle(perm)
		train_images = train_images[perm]
		train_labels = train_labels[perm]

		start = 0
		index_in_epoch = batch_size
		assert batch_size <= num_examples
	end = index_in_epoch
	return train_images[start:end], train_labels[start:end]


# Start Tensorflow session
# init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()


# Visualization variables
train_accuracies = []
validation_accuracies = []
x_range = []
step = 1


# ------------------------------ Training ---------------------------------

for i in range(train_iter):

	# get new batch
	batch_xs, batch_ys = next_batch(batch_size)

	if i%step == 0 or (i+1) == train_iter:

		train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_:batch_ys})

		if(Validation_size):

			validation_accuracy = accuracy.eval(feed_dict={x:validation_images[0:batch_size], y_:validation_labels[0:batch_size]})

			print ('train_accuracy / validation_accuracy => %.4f / %.4f for step %d' %(train_accuracy,validation_accuracy,i))

			validation_accuracies.append(validation_accuracy)

		else:
			print ('train_accuracy => %.4f for step %d' %(train_accuracy,i))
	
		train_accuracies.append(train_accuracy)
		x_range.append(i)

		# increase display step
		if i%(step*10) == 0 and i and step<100:
			step *= 10

	# train on batch
	sess.run(train, feed_dict={x:batch_xs, y_:batch_ys})

 
	# ------------------------ Results visualization ----------------------------

	# check final accuracy on validation set
if (Validation_size):

	validation_accuracy = accuracy.eval(feed_dict={x:validation_images, y_:validation_labels})
	print ('validation accuracy => %.4f' %validation_accuracy)
	plt.plot(x_range,train_accuracies,'-b',label = 'Training')
	plt.plot(x_range,validation_accuracies,'r', label = 'validation')
	plt.legend(loc='lower right', frameon=False)
	plt.ylim(ymax=1.0, ymin=0.0)
	plt.ylabel('Accuracy')
	plt.xlabel('Iterations')
	plt.show()


saver.save(sess, "./my_model_baseline")







