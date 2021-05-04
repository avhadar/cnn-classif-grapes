# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import  ImageDataGenerator

# tf.__version__


# Part 1 - Data Preprocessing

# Preprocessing the Training set (including scaling - mandatory for Neural Networks)
# apply some transformations on the training set (only!!) to avoid overfitting
# -> image augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# connect train_datagen object to the training data
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary') # binary (here) / categorical

# Preprocessing the Test set 
# -> only scaling !! (not all transformations of the training set)
test_datagen = ImageDataGenerator(rescale=1./255)

# connect test_datagen object to the test data
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution -> use input_shape only for the 1st convolutional layer !!
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size=3, activation = 'relu', input_shape = [64, 64, 3])) 
# filters -> number of feature detectors; kernel_size -> size of the feature detector; activation -> rectifier; input_shape -> 3-dim for RGB coding colors

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
# stride -> how to slide (in pixels)

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size=3, activation = 'relu')) 
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())


# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid')) # units = 1 -> for binary classification

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # adam optimizer -> gradient descent

# Training the CNN on the Training set and evaluating it on the Test set
# Computer Vision: train (x) AND test (validation_data) at the same time

cnn.fit(x = training_set, validation_data = test_set, epochs = 25)


# Part 4 - Making a single prediction

import numpy as np
from keras.preprocessing import image

 # load image: it needs to have the same size as the one used during training !!
test_image = image.load_img('dataset/single_prediction/apple_or_grape_16.jpg', target_size = (64, 64)) # -> PIL format

test_image = image.img_to_array(test_image) # numpy array of the image

# model works with batches -> add extra dimension to convert into a batch; where -> batch is first dim.
test_image = np.expand_dims(test_image, axis = 0) # axis = 0 -> adding a dimension as first dimension (the batch)

result = cnn.predict(test_image)

training_set.class_indices # -> which index corresponds to apples(0), which to grapes(1) (result)

if result[0][0] == 1:
    prediction = 'grapes'
else:
    prediction = 'apples'
    
print(prediction)


#---------------------------------- Saving and loading the model -------------------------------
#------------------------------------------------------------------
# How to save the model (example)
#------------------------------------------------------------------

# Save model
# serialize model to JSON
model_json = cnn.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
print("Saved model to JSON file")

# serialize model to YAML (alternative to JSON)
model_yaml = cnn.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)    
print("Saved model to YAML file")   
 
# serialize weights to HDF5
cnn.save_weights("model.h5")
print("Saved model to disk")


#------------------------------------------------------------------
# How to load the model
#------------------------------------------------------------------

# .... load saved model
# JSON example
from keras.models import model_from_json
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
 # ... then use the model