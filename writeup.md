**Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/random_samples.png "Random Samples"
[image2]: ./examples/class_distribution.png "Distribution of traffic signs"
[image3]: ./examples/Architecture.png "Model Architecture"
[image4]: ./new_traffic_signs_original_size/Speed_Limit_60km.png "Traffic Sign 1"
[image5]: ./new_traffic_signs_original_size/Gerneral_Caution.png "Traffic Sign 2"
[image6]: ./new_traffic_signs_original_size/Road_Work.png  "Traffic Sign 3"
[image7]: ./new_traffic_signs_original_size/Wild_Animal_Accrossing.png "Traffic Sign 4"
[image8]: ./new_traffic_signs_original_size/Keep_right.png "Traffic Sign 5"
[image9]: ./examples/Accuracies.png "Accuracy vs classes"
[image10]: ./examples/Convolution_Layer1.png "Activatin of conv layer 1"
[image11]: ./examples/Convolution_Layer2.png "Activatin of conv layer 2"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/etendue/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Step 0: Load the Data
The data sets are downloaded and extracted. The data formats are in pickle formats. So it is easy just to load the train, valid, test data by keywords. The code for load the data in cell 1 is already there. TODO for this step is just feed the paths for extracted pickle data.

### Step 1:Data Set Summary & Exploration


#### 1. The code for this step is contained in the second code cell of the IPython notebook.  

I got the  following summary statistics of the traffic signs data set by analyzing the loaded variables:

* The size of training set is **34799**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32,32,3)**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset

The code for this step is contained in the third code cell of the IPython notebook.  

#### 1. visualization of randomly picked train data
The code randomly picked 5 traffic sign images from train data and plotted using matplotlib tool. The traffic sign names are read using **pandas** lib to decorate the title of each subplot figure.
Here shows the figure.

![alt text][image1]

#### 2. visualization of train data vs classes
Following is  a histogram chart showing how the train data is distributed over different traffic signs, i.e. classes or labels
It seems that, the train data are not evenly distributed, especially classes between **20 ~ 40** are lower than other classes. This is potentially entry point for improvement by either adding new **augmented** train data for these classes

![alt text][image2]

### Step 2: Design and Test a Model Architecture



![alt text][image3]

#### 1. preprocessing
The traffic images as input are normalized by simply dividing the pixel value with 255. For train data the inputs and labels are shuffled for training, as original train data are grouped by labels It is necessary for SGD other training technology 

I shifted the converting images to grayscale inside the Model as Tensorflow provides the function [**tf.image.rgb_to_grayscale()**](https://www.tensorflow.org/versions/r1.1/api_docs/python/tf/image/rgb_to_grayscale)


The code for this step is contained in the fourth code cell of the IPython notebook.


#### 2. Train, validation and testing data
The data sets for train,validatoin and testing are load from pickle files. Only for Train the data is shuffled, and it is also shuffled during training(see in code cell 12) I did not augument new data so no further processing for above data set is done. 



#### 3. Model Architecture analysis

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x108 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x108				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x108  |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x108				    |
| **Layer=Conv1_2ndPooling + Conv2** | output 5x5x206                   |
| Flatten               | output 5400                                   |
| Fully connected		| output 5400 x 120  							|
| RELU					|												|
| Fully connected		| output 120 x 84  							    |
| RELU					|												|
| Dropout        		| keep_prob = 0.5 output 120 x 84       	    |
| Fully connected		| output 84 x 43  							    |
| Softmax				| output 43    									|

The Layer= Conv1_2ndPooling + Conv2 is inspired from [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).In this paper it is called multi-scale structure, meaning the full-connected layer sees not only the activation of second convolution layer but also the activation from first convolution layer. The idea behinds is related to perceptive fields of eye which is out of current scope. In practice it seems by doing this improving the validation precision about 0.5 percent. 


#### 4. Train, Validate and Test the model

The code for training the model is located in the 8th,9th,10th,11th, 12th cell of the ipython notebook. 

To train the model, I used batch size 128, epochs 20 and learning rate 0.001 with AdamOptimizer and cost function of softmax cross_entropy with logits. I wrote the train code by taking reference from [CarND_LeNet_Lab](https://github.com/udacity/CarND-LeNet-Lab) and did some adaptation like introducing dropout parameter. 

In 11 code cell, I wrote 2 evaluate functions.
* evaluate : get a overall validation accuracy for whole testing data
* evaluate_more: get the validation accuracies vs classes

The second evaluate function is used to inspect the model, where are the performance deficiency. 


#### 5. Evaluate the model

The code for calculating the accuracy of the model is located in the 13th cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.993
* validation set accuracy of 0.961 
* test set accuracy of 0.945

As I described the different steps to evolve the model. I tried establishing a working model which can do the basics then by changing some parts to get improvement or not. I could have spent a lot of time by trying different parameters alone, but there are good reference models available online so I took them and combined a little bit and  finally realized one in restricted time line.

I took the model of LeNet from [github CarND-LeNet-Lab](https://github.com/udacity/CarND-LeNet-Lab) and adapted it step by step. Here is what I did.

| changes | validation accurary|
| --------|--------------------|
| orginal network with adaptation of input format and adjusted the number of convolution nodes. 6x16 to 22x38 | 0.918 |
| convert input image from rgb to grayscale | 0.891 |
| increase the number of convolution nodes 22x38 to 108x108 |0.936 |
| add dropout regularization after fully connected layers  |0.949 |
| increase the train epochs |0.963|
| apply the multi-scale model [see reference](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) | 0.968|


The final model architecture is chosen by accumulating all above steps and is illustrated here. Here I describe the final model in details.


I have some doubts/questions on my final model:
* it is probably overfitted.
* augment more train samples may likely improve the accuracy.(see my analysis on accuracy vs classes)

Since the decision of model and its parameters is highly empirical, the final justificaiton of model selection is only the test/validation accuracy of tried models. 

#### 6. Accuracy vs classes

In 14th cell code I ran the evaluation\_more function to get a diagram of accuracy vs classes. I saw that accuracies differ dramatically over classes. Especially it is low between classes 20 ~ 30, where in this range the number of train data are also low.
See picture below. This strongly recommend to add more train data for classes 20 ~ 30 or which class has low amount of train  data.


![alt text][image9] 
![alt text][image2]


### Test a Model on New Images

#### 1. Five German traffic signs found on the web.

Here are five German traffic signs that I found on the web: https://en.wikipedia.org/wiki/Road_signs_in_Germany

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

These images have high resolutions and are in format png with RGBA 4 pixel channels. I did some manual work separately to convert them to wanted 32x32x3 formats. Fortunately all new images are classified correctly with quite big certainty. However I observed the 1 traffic sign was not recognized once with a trained model.




#### 2. Performance analysis



The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep Right   		    | Keep Right									| 
| Speed limit (60km/h)  | Speed limit (60km/h)							|
| Wild animals crossing	| Wild animals crossing							|
| General caution	    | General caution					 			|
| Road work		        | Road work      							    |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.9%

In 18th and 19th code cells depicts the top 5 candidate for recognized traffic signs and their probabilities receptively. The model is quite sure about the recognized traffic classes on new traffic sign images. By looking at the probabilities of top 5 recognized classes, the first class has probability > 99% 


### Step 4: Visualization of Neural Network's State with Test Images

It is interesting to look into the hidden layer to infer what the model has learned. To get the tensor flow node inside the model, especially these tensors defined inside a function, it is necessary to define the tensors(variables, nodes) with a name, i.e. providing the name parameter when creating the node variable or call [tf.identity](https://www.tensorflow.org/api_docs/python/tf/identity) function.  I plot the first and second convolution layer outputs (shown below).

* Layer 1 feature maps  assembles the input image, however layer 2 is much abstracter than layer 1.
* There are feature maps which are totally dark, which means they are not activated by the current image. 

For visualization I think it is good idea to keep the image size. 

![alt text][image10] ![alt text][image11]

