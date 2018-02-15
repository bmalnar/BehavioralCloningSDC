##### **Behavioral Cloning Project**
#### Writeup / README

This is a summary of the work done to train the neural network to drive a car on a track in the Udacity self-driving car simulator. The Github project is located [here] (https://github.com/bmalnar/BehavioralCloningSDC)

The steps described in the following text include:
1) Dataset exploration and augmentation
2) Training the neural network
3) Testing the neural network in the simulator
4) Generating video of the car driving in the autonomous mode

### Udacity dataset for the project

To get to a quick start, Udacity kindly provides the starting dataset that can be used for initial exploration, i.e. setting up the working environment, training the neural network, and using the trained model with the simulator to get familiar with the process. The dataset can be downloaded [here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)

The main file in the dataset is `driving_log.csv`. The header and the first data row of the file are shown below:

```
center,left,right,steering,throttle,brake,speed
IMG/center_2016_12_01_13_30_48_287.jpg, IMG/left_2016_12_01_13_30_48_287.jpg, IMG/right_2016_12_01_13_30_48_287.jpg, 0, 0, 0, 22.14829
```

There are in total 8036 data rows in the file (excluding the header). Each data row contains three images (taken by central, left and right camera mounted at the front end of the car), together with four floating point values for steering, throttle, brake and speed. The goal of the project is to train the neural network to produce the steering angle based on the images of the front facing cameras. The training images listed in the csv file are stored in the IMG directory.  

It is interesting to observe the distribution of the steering angle values over the entire dataset provided by Udacity. This is shown in the following picture. 

<img src="images/udacity_dataset.png" width="480" alt="Steering angle distribution in the Udacity dataset" />

We can see that the majority of the data points can be found around the zero steering angle. This is due to the fact that the track that was used to generate the data is mostly flat, so steering is typically close to zero. 

I did try to train the neural network **on this data as-is**, but it didn't work - the car in the autonomous mode would simply lose control in the first curve and go off the road. Apparently the car learned well to go straight, but not really how to make turns. The neural network architecture was probably ok, because I managed to get a good model later with the same network using more data.    

### Collecting more data by running in the simulator

I personally found the simulator very difficult to work with. I spent a lot of time trying to keep the car in the middle of the road, but without major success. I ended up driving very slowly, and steering in a way that would probably not be a good set of data for training. I read online that people experienced the same problem and ended up recommending a joystick, whereas I had only the keyboard. Before purchasing a joystick, I decided to give it a shot by using only augmentation approaches based on the Udacity provided dataset, and it worked, as described in the subsequent sections. 

### Data augmentation



#### Pre-processing the image data

All three datasets (training, validation and testing) were pre-processed in two steps:
- Convert from RGB to grayscale using the method `rgb2gray`
- Normalize the grayscale images to have zero mean and unit variance using the method `normalize`

Conversion from RGB to grayscale is typically done to make the models simpler and smaller. However, sometimes having the color information could help the network to learn to classify better, if the color is indeed a decisive factor between the classes. In the case of traffic signs, it is probably not necessary to have the color information, because the traffic signs can be distinguished well even if the images are converted to grayscale for the purpose of classification. This is after all confirmed with the training and testing processes, which produce good results on grayscale images. 

Normalization is typically done to ensure that the input data to the network has similar distribution, which helps the network to converge faster during training. If normalization isn't done, we could have a situation that very bright and very dark images affect the training process negatively because the brightness factor may be something that the network also tries to learn, and that is not what we want. 

Below, we can see the result of changing the images from RGB to grayscale. We first show 42 RGB images from 42 different classes, and then we show the same images converted to RGB. We show only 42 images even though there are 43 classes, just to have the nice 7x6 grid for displaying the images (this is dufficient for illustration purposes). 

- RGB images:

<img src="writeup_images/rgb.png" width="480" alt="RGB images" />

- Grayscale images:

<img src="writeup_images/grayscale.png" width="480" alt="Grayscale images" />

#### The model architecture

The neural network chosen for this work resembles the LeNet network architecture, and has the following layers:

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| On the flat input, output is 120        									|
| RELU					|												|
| Fully connected		| Output is 84        									|
| RELU					|												|
| Fully connected		| Output is 43        									|
| Softmax				|     									|
 
 
#### The training process

For training, several different experiments were performed to investigate the impact of different settings and hyperparameters for achieving the desired accuracy. The list of settings experimented with includes the following:
- **Input data format:** use RGB images vs. grayscale without normalization vs. grayscale with normalization. Between the three different options, the last one was a clear winner to get the training process and testing converge quickly. Therefore, the model uses grayscale and normalized pictures (both for training and testing). 
- **Learning rate:** use a constant learning rate vs. a decaying learning rate. It is often recommended that the learning rate is best lowered later in the training process, to achieve faster learning rate at the beginning and more precise convergence later. However, with the experiments performed here, it was noticed that the best convergence happens with the constant learning rate. 
- **Optimizers:** use different optimizers (AdamOptimizer, RMSPropOptimizer, GradientDescentOptimizer). The impact of selecting different optimizers did not appear to influence the results, so eventually AdamOptimizer was chosen for the reasonce mostly involving prior experiences with that optimizer. 
- **L2 loss:** use L2 loss for weights vs. do not use L2 loss. Using L2 loss was a clear winner here, as the accuracy for both validation and testing datasets was increased by several percentages over the case where L2 loss was not used. 
- **Dropout:** use dropout for conv layers and FC layers vs. use dropout only for conv layers vs. do not use dropout at all. Using dropout only for conv layers was the better choise vs. using dropout also for the FC layers or not using dropout at all, resulting in the improvement in the validaton adn testing accuracy of several percentage points. 
- **Number of training epochs:** increase the number of training epochs. The initial number of epochs was 10, but it was eventually increased to 60. The runtime penalty is not significant because the training procedure runs relatively fast anyway (about 2-3 minutes for 60 epochs). 

The architecture was not changed during these experiments. The approach was to pick a reasonable architecture from the beginning (which is relatively small and simple) and to try to train it to get the desired accuracy. The architecture can be made deeper, with more layers, but this was eventually not necessary. 

During training, at the end of every epoch, the accuracy of the model is evaluated using the validation dataset. In other words, at the end of each epoch we have the estimate of how accurate the model is on the data it has not seen during the training. The accuracy is around 0.4 after the first epoch, over 80% after 5 epochs, over 90% after 12 epochs, and then slowly goes up to about the peak of 94%. At the end of the training, we evaluate the accuracy once more using the test dataset. The accuracy in that case is **94.6%**, which exceeds 93% (which was set as a requirement at the beginning). 

### Testing the model on new images

For additional testing, several images were downloaded from the internet and placed in the directory `test_images`:

```test_images/11_rightofway.jpg```

<img src="test_images/11_rightofway.jpg" width="200" alt="RGB images" />

```test_images/14_stop.jpg```

<img src="test_images/14_stop.jpg" width="200" alt="RGB images" />

```test_images/17_noentry.jpg```

<img src="test_images/17_noentry.jpg" width="200" alt="RGB images" />

```test_images/17_noentry_crop.jpg```

<img src="test_images/17_noentry_crop.jpg" width="200" alt="RGB images" />

```test_images/23_slippery.jpg```

<img src="test_images/23_slippery.jpg" width="200" alt="RGB images" />

```test_images/23_slippery_crop.jpg```

<img src="test_images/23_slippery_crop.jpg" width="200" alt="RGB images" />

```test_images/25_roadwork.jpg```

<img src="test_images/25_roadwork.jpg" width="200" alt="RGB images" />

```test_images/25_roadwork_2.jpg```

<img src="test_images/25_roadwork_2.jpg" width="200" alt="RGB images" />


The images are named in the way that the names begin with the number designating the class ID, as defined in the `signnames.csv` file. In that way, we do not have to store the labels separately, but we simply extract them from the file names. 

As we test the model with these immages, we can see the following output:

```
Image correctly classified: 11_rightofway.jpg as class 11
Image correctly classified: 14_stop.jpg as class 14
Image incorrectly classified: 17_noentry.jpg as class 0
Image correctly classified: 17_noentry_crop.jpg as class 17
Image incorrectly classified: 23_slippery.jpg as class 38
Image correctly classified: 23_slippery_crop.jpg as class 23
Image correctly classified: 25_roadwork.jpg as class 25
Image correctly classified: 25_roadwork_2.jpg as class 25**
```

We can see that the model accuracy on these new images is **75%** (6 out of 8 classfied correctly). This number is lower than the accuracy of 94.6% achieved on the test dataset, but we only have 8 images here, which may be statistically not significant enough to compare the two numbers. 

The reason why the images `17_noentry.png` and `23_slippery.png` are not correctly classified is most likely due to the fact that these images do not just show the traffic sign in a zoomed-in kind of way, but they also have substantial area around the signs, which show branches and leaves. The training dataset, on the other hand, shows only traffic sign images where the signs occupy the entire area of the images, and nothing else is shown. So the model learns to recognize the signs only in the absence of other content. That is why I think `17_noentry.jpg` is classified incorrectly, but `17_noentry_crop.jpg` is classified correctly. The latter image shows the same sign as the former, but simply the content around the sign itself has been removed. The situation is the same with `23_slippery.jpg` and `23_slippery_crop.jpg`. 

To investigate how confident the model is when predicting these two images, we can look at the following output from notebook:

```
Image 11_rightofway.jpg predicted as 11
Top 5 classes [[11 30 21 12 27]]
Top 5 values [[ 0.92752033  0.06129503  0.00334978  0.00217773  0.00173397]]

Image 14_stop.jpg predicted as 14
Top 5 classes [[14 33 13 17 34]]
Top 5 values [[ 0.91339582  0.01729976  0.01514864  0.0130596   0.00775108]]

Image 17_noentry.jpg predicted as 0
Top 5 classes: 0  1  8 33 36
Top 5 values: 0.25731021  0.10710046  0.09363007  0.0816074   0.05592449

Image 17_noentry_crop.jpg predicted as 17
Top 5 classes: 17  9 33 16 35]]
Top 5 values: 0.48428649  0.17560029  0.10189056  0.09391738  0.05823837

Image 23_slippery.jpg predicted as 38
Top 5 classes: 38 25 31 11 23
Top 5 values: 0.29157507  0.14607479  0.12122528  0.0940006   0.04848812

Image 23_slippery_crop.jpg predicted as 23
Top 5 classes: 23 11 19 21 31
Top 5 values: 0.41349074  0.19613142  0.11798017  0.07410568  0.069729  

Image 25_roadwork.jpg predicted as 25
Top 5 classes: 25 31 29 21 30
Top 5 values: 0.62263769  0.08411928  0.05425425  0.04839035  0.04740327

Image 25_roadwork_2.jpg predicted as 25
Top 5 classes: 25 29 24 22 18
Top 5 values: 0.98748207  0.00331216  0.00220919  0.00131308  0.00117823
```

We can see that some correct predictions have a very high probability of over 90%, but some have less than 50%. Compared to the cases where the predictions are incorrect, we still see that the output probabilities are relatively higher, and the margins with the second highest probabilities is relatively wider, compared to mis-predictions. 
