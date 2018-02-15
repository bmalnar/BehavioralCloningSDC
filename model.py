import cv2
import csv
import numpy as np
import os
import errno
import json


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import sklearn
import matplotlib.pyplot as plt

def get_lines_from_csv_file(directory, driving_log, skip_header=False):
   """
   Returns the lines from a CSV file
   """
   all_lines = []
   file_name = os.path.join(directory, driving_log)  
   with open(file_name) as csv_file:
      print("Reading CSV file:", file_name)
      csv_reader = csv.reader(csv_file)
      if skip_header:
         next(csv_reader, None)
      for curr_line in csv_reader:
         all_lines.append(curr_line)
   return all_lines


def get_images_and_measurements(dir_name, driving_log):
   """
   Finds all the training images in the given directory
   """
   # directories = [x[0] for x in os.walk(dir_name)]
   # data_directories = list(filter(lambda directory: os.path.isfile(directory + '/' + driving_log), directories))
   data_directories = [dir_name]
   all_center_images = []
   all_left_images = []
   all_right_images = []
   all_measurements = []
   print("Working on directories:", data_directories)
   for directory in data_directories:
      lines = get_lines_from_csv_file(directory, driving_log, True)
      center_image = []
      left_image  = []
      right_image  = []
      measurements = []
      for line in lines:
         # print("Processing line:", line)
         measurements.append(float(line[3]))
         center_image .append(directory + '/' + line[0].strip())
         left_image .append(directory + '/' + line[1].strip())
         right_image .append(directory + '/' + line[2].strip())
      all_center_images.extend(center_image )
      all_left_images.extend(left_image )
      all_right_images.extend(right_image )
      all_measurements.extend(measurements)

   return (all_center_images, all_left_images, all_right_images, all_measurements)

def create_list_of_images_and_measurements(images_c, images_l, images_r, measurements, correction_factor):
   """
   Comibines the images from the center, left and right cameras into a single
   list, and applies the correction factor the left and right measurement cases
   """
   combined_images = []
   combined_images.extend(images_c)
   # combined_images.extend(images_l)
   # combined_images.extend(images_r)

   combined_measurements = []
   combined_measurements.extend(measurements)
   # combined_measurements.extend([x + correction_factor for x in measurements])
   # combined_measurements.extend([x - correction_factor for x in measurements])

   return (combined_images, combined_measurements)

def combine_images_and_measurements(images_c, images_l, images_r, measurements, correction_factor):
   """
   Comibines the images from the center, left and right cameras into a single
   list, and applies the correction factor the left and right measurement cases
   """
   combined_images = []
   combined_images.extend(images_c)
   combined_images.extend(images_l)
   combined_images.extend(images_r)
   
   combined_measurements = []
   combined_measurements.extend(measurements)
   combined_measurements.extend([x + correction_factor for x in measurements])
   combined_measurements.extend([x - correction_factor for x in measurements])

   return (combined_images, combined_measurements)

def generator(samples, batch_size):
   """
   Generate the required images and measurments for training
   """
   num_samples = len(samples)
   while 1: # Loop forever so the generator never terminates
      samples = sklearn.utils.shuffle(samples)
      for offset in range(0, num_samples, batch_size):
         batch_samples = samples[offset:offset+batch_size]

         images = []
         angles = []
         for imagePath, measurement in batch_samples:
            originalImage = cv2.imread(imagePath)
            image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
            images.append(image)
            angles.append(measurement)
            # Flipping
            images.append(cv2.flip(image,1))
            angles.append(measurement*-1.0)

            # trim image to only see section with road
            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)

def create_model():

   model = Sequential()
   # model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
   model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))
   # model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(88, 320, 3)))
   # model.add(Cropping2D(cropping=((50,20), (0,0))))

   model.add(Conv2D(24, (5, 5), activation="relu", padding='same', strides=(2, 2)))
   model.add(Conv2D(36, (5, 5), activation="relu", padding='same', strides=(2, 2)))
   model.add(Conv2D(48, (5, 5), activation="relu", padding='same', strides=(2, 2)))
   model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
   model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
   model.add(Flatten())
   model.add(Dense(1164))
   model.add(Dense(100))
   model.add(Dense(50))
   model.add(Dense(10))
   model.add(Dense(1))

   return model

# Set parameters
test_split_size = 0.2
train_batch_size = 32
validation_batch_size = 32
num_epochs = 3

# Reading images locations.

# Use one or the other depending on weather using the original data only or augmented
image_c, image_l, image_r, measurements = get_images_and_measurements('augm_same_recipe', 'driving_log_augm.csv')
# image_c, image_l, image_r, measurements = get_images_and_measurements('augm', 'driving_log_augm.csv')
# image_c, image_l, image_r, measurements = get_images_and_measurements('data', 'driving_log.csv')

# Use one or the other depending on weather using the original data only or augmented
# In the augmented dataset, we use only one image, so we can assume center, while we ignore the rest
# images_paths, measurements = combine_images_and_measurements(image_c, image_l, image_r, measurements, 0.2)
images_paths, measurements = create_list_of_images_and_measurements(image_c, image_l, image_r, measurements, 0.2)
print("Number of images:", len(images_paths))

# Splitting samples and creating generators.
from sklearn.model_selection import train_test_split
samples = list(zip(images_paths, measurements))
train_samples, validation_samples = train_test_split(samples, test_size=test_split_size)

num_train_samples = len(train_samples)
num_validation_samples = len(validation_samples)

print("Number of train samples:", num_train_samples)
print("Number of validation samples:", num_validation_samples)

train_generator = generator(train_samples, train_batch_size)
validation_generator = generator(validation_samples, validation_batch_size)

# Create and compile the model
model = create_model()
model.compile(loss='mse', optimizer='adam')

# Train the model:
print("Number of training epochs:", num_epochs)
num_steps_per_epoch = int(num_train_samples / train_batch_size)
print("Number of steps per epoch:", num_steps_per_epoch)

history_object = model.fit_generator(train_generator, \
   steps_per_epoch = num_steps_per_epoch, \
   validation_data = validation_generator, \
   nb_val_samples = num_validation_samples, \
   nb_epoch = num_epochs, \
   verbose=1)

def save_model(model, model_name='model.json', weights_name='model.h5'):
    """
    Save the model into the hard disk

    :param model:
        Keras model to be saved

    :param model_name:
        The name of the model file

    :param weights_name:
        The name of the weight file

    :return:
        None
    """
    silent_delete(model_name)
    silent_delete(weights_name)

    json_string = model.to_json()
    with open(model_name, 'w') as outfile:
        json.dump(json_string, outfile)

    model.save_weights(weights_name)


def silent_delete(file):
    """
    This method delete the given file from the file system if it is available
    Source: http://stackoverflow.com/questions/10840533/most-pythonic-way-to-delete-a-file-which-may-not-exist

    :param file:
        File to be deleted

    :return:
        None
    """
    try:
        os.remove(file)

    except OSError as error:
        if error.errno != errno.ENOENT:
            raise

# Save the model and report the training outcome
# model.save('model.h5')

save_model(model)

'''
json_string = model.to_json()
with open('model.json', 'w') as outfile:
    json.dump(json_string, outfile)

model.save_weights(weights_name)
'''

print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
