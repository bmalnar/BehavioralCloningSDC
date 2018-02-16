import errno
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.misc
from scipy.ndimage import rotate
from scipy.stats import bernoulli
import shutil

def crop(image, top_percent, bottom_percent):
   '''
   Crops an image according to top and bottom percentage
   '''
   top = int(np.ceil(image.shape[0] * top_percent))
   bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))

   return image[top:bottom, :]


def resize(image, new_dim):
   '''
   Resize an image to the given dimensions
   '''
   return scipy.misc.imresize(image, new_dim)


def random_flip(image, steering_angle, flipping_prob=0.5):
   ''' 
   Flip the image, but only with the probability of 0.5
   If flipped, change the sign of the steering angle, otherwise
   just return the same image amd angle
   ''' 
   head = bernoulli.rvs(flipping_prob)
   if head:
      return np.fliplr(image), -1 * steering_angle
   else:
      return image, steering_angle

def random_gamma(image):
   '''
   Perform gamma correction, based on the random value of gamma
   http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
   '''

   gamma = np.random.uniform(0.4, 1.5)
   inv_gamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

def random_affine(image, steering_angle, shear_range=200):
   ''' 
   Apply random shear on the input image
   '''
   rows, cols, ch = image.shape
   dx = np.random.randint(-shear_range, shear_range + 1)
   random_point = [cols / 2 + dx, rows / 2]
   pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
   pts2 = np.float32([[0, rows], [cols, rows], random_point])
   dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
   M = cv2.getAffineTransform(pts1, pts2)
   image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
   steering_angle += dsteering

   return image, steering_angle

def process_image(image, steering_angle, top_crop_percent=0.35, bottom_crop_percent=0.1,
                       resize_dim=(64, 64), do_shear_prob=0.9, shear_range=200):
   '''
   Process image and create a new image with shear, crop, flip, random gamma and resize ops
   Accordingly, modify the steering value
   '''

   head = bernoulli.rvs(do_shear_prob)
   if head == 1:
      image, steering_angle = random_affine(image, steering_angle, shear_range=shear_range)

   image = crop(image, top_crop_percent, bottom_crop_percent)

   image, steering_angle = random_flip(image, steering_angle)

   image = random_gamma(image)

   if resize_dim[0] > 0:
      image = resize(image, resize_dim)

   return image, steering_angle

def generate_augmented_data(data_root_dir, driving_log_file, out_dataset_size=100000, measurement_adjustment=0.2):
   '''
   Generate the augmented dataset based on the input dataset
   The output dataset will have the same structure, i.e. the csv file with the 
   same fields as the original one, and the IMG directory with all the images
   '''

   # Read the input csv file
   driving_log = pd.read_csv(os.path.join(data_root_dir, driving_log_file))

   number_of_input_img = len(driving_log)

   print("Number of input images:", number_of_input_img)

   list_column_names = ('left', 'center', 'right')
   list_measurement_adjustments = (measurement_adjustment, 0, 0.0 - measurement_adjustment)

   # Set the output destinations, and delete them if already there
   out_dir = os.path.join(data_root_dir, "augm")
   out_img_dir = os.path.join(out_dir, "IMG")
   out_csv = os.path.join(out_dir, "driving_log_augm.csv")

   print("Out csv file:", out_csv)

   if os.path.exists(out_dir):
      print("Removing the existing directory:", out_dir)
      shutil.rmtree(out_dir)

   os.makedirs(out_dir)
   os.makedirs(out_img_dir)

   format_str = '0' + str(len(str(out_dataset_size))) + 'd'

   count = 0
   next_display = 0.1

   print("Processing images...")

   # Open the csv file and write the header to it
   f_out_csv = open(out_csv, 'w')
   f_out_csv.write("center,left,right,steering,throttle,brake,speed\n")

   # Loop until the desired number of images is created
   while count < out_dataset_size:
      count += 1
      if (float(count)/out_dataset_size) > next_display:
        print ("Done:", int(next_display * 100), "%")
        next_display += 0.1
      rand_index = np.random.randint(number_of_input_img)

      # Use this only if using all 3 images from all 3 frontal cameras
      # rand_sub_index = np.random.randint(3)
      # Otherwise, rand_sub_index is 1, meaning center image      
      rand_sub_index = 1
      col_name = list_column_names[rand_sub_index]
      img_name = driving_log.iloc[rand_index][col_name].strip()
      measurement = driving_log.iloc[rand_index]['steering'] + list_measurement_adjustments[rand_sub_index]

      # Create the new image file name, and the new location that 
      # will be entered in the csv file
      suffix = "__" + format(count, format_str) + ".jpg"
      new_image_file_name = os.path.basename(img_name)
      new_image_file_name = new_image_file_name.replace(".jpg", suffix)
      mew_image_path = os.path.join(out_img_dir, new_image_file_name)
      new_image_file_name_csv = os.path.join("IMG", new_image_file_name)

      # Read and process image
      raw_image = plt.imread(os.path.join(data_root_dir, img_name))
      new_image, new_measurement = process_image(raw_image, measurement)
      plt.imsave(mew_image_path, new_image)

      # Store the modified entry in the csv file
      f_out_csv.write(new_image_file_name_csv + ",")
      f_out_csv.write(new_image_file_name_csv + ",")
      f_out_csv.write(new_image_file_name_csv + ",")
      f_out_csv.write(str(new_measurement))
      f_out_csv.write(",")
      f_out_csv.write(str(driving_log.iloc[rand_index]['throttle']))
      f_out_csv.write(",")
      f_out_csv.write(str(driving_log.iloc[rand_index]['brake']))
      f_out_csv.write(",")
      f_out_csv.write(str(driving_log.iloc[rand_index]['speed']))
      f_out_csv.write("\n")      

   print("Done processing images.")

   f_out_csv.close()

generate_augmented_data("data", "driving_log.csv", out_dataset_size=500000)
