import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools

# --------------------------- Data Extraction -------------------------

data = pd.read_csv("fer2013/fer2013.csv")
test_data = data[data.Usage == "PublicTest"]
train_data = data[data.Usage == "Training"]

train_pixel_values = train_data.pixels.str.split(" ").tolist()
train_pixel_values = pd.DataFrame(train_pixel_values, dtype = int)
train_images = train_pixel_values.values
train_images = train_images.astype(np.float)


test_pixel_values = test_data.pixels.str.split(" ").tolist()
test_pixel_values = pd.DataFrame(test_pixel_values, dtype=int)
test_images = test_pixel_values.values
test_images = test_images.astype(np.float)
#test_images = test_images - train_images.mean()
test_images = test_images - test_images.mean(axis=1).reshape(-1,1)
test_images = np.multiply(test_images,100.0/255.0)
each_pixel_mean = test_images.mean(axis=0)
each_pixel_std = np.std(test_images, axis=0)
test_images = np.divide(np.subtract(test_images,each_pixel_mean), each_pixel_std)
predicted_labels = np.zeros(test_images.shape[0])


with tf.Session() as sess:
	saver = tf.train.import_meta_graph('my_model_baseline.meta')
	saver.restore(sess, 'my_model_baseline')
	predict = tf.get_collection("predict")[0]

	batch_size = 50

	for i in range(0,test_images.shape[0]//batch_size):

		predicted_labels[i*batch_size : (i+1)*batch_size] = sess.run(predict, feed_dict={"x:0":test_images[i*batch_size : (i+1)*batch_size]})

	print ('predicted_labels({0})'.format(len(predicted_labels)))
	print ("Test accuracy is ", accuracy_score(test_data.emotion.values, predicted_labels))

	y_true = pd.Series(test_data.emotion.values)
	y_pred = pd.Series(predicted_labels)

	k = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
	print (k)


	confusion_matrix(test_data.emotion.values, predicted_labels)
	cnf_matrix = confusion_matrix(test_data.emotion.values, predicted_labels)
	

	# Plot the confusion matrix to visualize the classification accuracies for each emotion
	def plot_confusion_matrix(cm, classes, normalize = True, title = 'Confusion Matrix', cmap=plt.cm.Blues):

		plt.imshow(cm,interpolation='nearest',cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks,classes,rotation=45)
		plt.yticks(tick_marks,classes)

		if normalize:
			cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
			print("Normalized Confusion Matrix")
		else:
			print('Confusion Matrix without Normalization')

		print(cm)

		thresh = cm.max() / 2.0
		for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):

			#plt.text(j,i, int(cm[i j]*100)/100.0, horizontalalignment = "center", color = "white" if (cm[i,j]) > thresh else "black")
			plt.text(j,i, int(cm[i, j]*100)/100.0, horizontalalignment = "center", color = "white" if (cm[i,j]) > thresh else "black")
			#plt.text(j,i, int(cm[i, j]*100)/100.0, horizontalalignment = "center", color = "black")

		plt.tight_layout()
		plt.ylabel('True Labels')
		plt.xlabel('Predicted Labels')

	class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

	plot_confusion_matrix(cnf_matrix, classes=class_names,normalize=True,title='Confusion Matrix for test dataset')
	plt.show()

