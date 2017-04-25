# **Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Network**

When building a network to accurately recognise and classify traffic signs, I followed these steps:
* Load the data set (I used the German Traffic Signs Dataset, **link below**!)
* Explore, summarize and visualize the dataset
* Pre-process, balance and augment the dataset
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images


[//]: # (Image References)

[image1]: ./images/hist_orig.png "Visualisation"
[image2]: ./examples/hist_new "New Visualisation"
[image3]: ./examples/processed_images.jpg "Processed Images"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"


Here's a link to my [Github repo](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration
Stats are as follows:
Number of training examples = 34799
Number of vaidation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

####  Exploratory Visualization of the Dataset

The German Traffic Signs dataset is visualised in the below histogram. It is split into 43 categories of signs, with the number of images in each category on the y-axis. Note the extremely unequal distribution of the dataset. This can and will introduce bias into the training model, and will be addressed below. Also, a low of ~200 examples for some classes is simply too low, and will need to be augmented.

![alt text][image1]

#### Data Preparation and Pre-processing


While the data can indeed simply be inserted into the model to begin training as-is, it would be to our benefit to process the data as follows:

##### 1. Grayscaling
Grayscaling reduces the image to a single layer instead of 3 RGB channels, drastically reducing the number of variables the network has to deal with. This results in vastly improved processing times. While loss of information due to loss of colour could be a concern, my tests with both RGB and BW images show no significant difference in performance.

##### 2. Equalisation
Equalisation of the image helps improve contrast and provides clearer, more well-defined edges. I initially used OpenCV's histogram equalisation, but found the results to be blurry and of poor contrast. skimage's adaptive CLAHE implementation took longer to process, but gave a far superior result.

##### 3. Normalisation
Normalisation involves scaling the image's intensity range from (0, 255) to (-1, 1). Smaller values with means about 0 prevent our gradients from going out of control and finding incorrect local minima.

##### 4. Augmentation (Transformation)
Due to the low number of examples from some classes, I've chosen to re-balance the dataset to prevent bias in the mode. There after, I tripled the size of the dataset over all classes, including the ones already heavily represented. I initially attempted penalised classification to make up for the dataset imbalance, but found good ol' data augmentation more effective.

##### 5. Shuffling
Rather self-explanatory - shuffles the dataset around so that the model doesn't train itself on the ORDER of the images instead of the features.


#### Grayscaling, Equalisation and Normalisation
My batch preprocess function is as follows. I began with conversion to grayscale, contrast limited adaptive histogram equalisation and normalisation from -1 to 1.

I initially used OpenCV's histogram equalisation, but found skimage's CLAHE implementation gave a much better result with more defined edges.
~~~
from skimage import exposure
from sklearn.utils import shuffle
from skimage import exposure
import cv2

def batchPreprocess(X):
    X_norm = []
    for image in X:
            bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            equ = exposure.equalize_adapthist(bw)
            equ = (equ * 2.0/equ.max())
            equ = np.reshape(equ,(32,32,1))-1
            X_norm.append(equ)
    return np.array(X_norm)
~~~

#### Augmentation (Transformation)
Here, I employed rotation (within a certain degree range) and warping through projective transforms (skimage). Projective transforms were chosen due to their similarity to changes in camera perspective.

Credits to navoshta for helping me figure out projective transforms!

NOTE: In earlier testing, doubling the dataset size without re-balancing it already yielded an accuracy of 95% on the validation data. While that seems nice, there is a high chance that the incorrect 5% stems from under-represented classes, rendering the prediction useless.
~~~
def randomTransform(image, intensity):
    
    # Rotate image within a set range, amplified by intensity of overall transform.
    rotation = 30 * intensity
    # print(image.shape)
    rotated = rotate(image, random.uniform(-rotation,rotation), mode = 'edge')
    
    # Projection transform on image, amplified by intensity.
    image_size = image.shape[0]
    magnitude = image_size * 0.3 * intensity
    tl_top = random.uniform(-magnitude, magnitude)     # Top left corner, top margin
    tl_left = random.uniform(-magnitude, magnitude)    # Top left corner, left margin
    bl_bottom = random.uniform(-magnitude, magnitude)  # Bottom left corner, bottom margin
    bl_left = random.uniform(-magnitude, magnitude)    # Bottom left corner, left margin
    tr_top = random.uniform(-magnitude, magnitude)     # Top right corner, top margin
    tr_right = random.uniform(-magnitude, magnitude)   # Top right corner, right margin
    br_bottom = random.uniform(-magnitude, magnitude)  # Bottom right corner, bottom margin
    br_right = random.uniform(-magnitude, magnitude)   # Bottom right corner, right margin
    
    transform = ProjectiveTransform()
    transform.estimate(np.array((
            (tl_left, tl_top),
            (bl_left, image_size - bl_bottom),
            (image_size - br_right, image_size - br_bottom),
            (image_size - tr_right, tr_top))),
            np.array((
            (0, 0),
            (0, image_size),
            (image_size, image_size),
            (image_size, 0)
            )))
    transformed = warp(rotated, transform, output_shape = (image_size, image_size), order = 1, mode = 'edge')
    return transformed
~~~

#### Altogether Now
Mini pipeline of the pre-processing phase.
~~~
# Processing both training and validation sets
X_train_proc = batchPreprocess(X_train)
X_valid_proc = batchPreprocess(X_valid)

print("Pre-processing complete!")
~~~
Rebalancing based on representation of classes.
~~~
unique, counts = np.unique(y_train, return_counts=True)
print("Original distribution of classes: ", counts)
multiplier = [int(round(max(counts)/i)) for i in counts] # Required multiplier for each class augmentation.
print("Multipliers for each class: ", multiplier)
multiplier = [i-2 for i in multiplier]

X_train_aug = X_train_proc
y_train_aug = y_train

for i in types:
    if multiplier[i] > 0: # Ignore classes which don't need oversampling
        X_train_add = []
        y_train_add = []
        index = np.where(y_train==i)
        for j in index:
            X_train_add.append(X_train_proc[j])
            y_train_add.append(y_train[j])
        X_train_add = np.array(X_train_add)
        X_train_add = np.reshape(X_train_add, (len(index[0]),32,32,1))
        y_train_add = np.array(y_train_add)
        y_train_add = np.reshape(y_train_add, (len(index[0])))
    
        print("Class: ", i+1)
        X_train_add, y_train_add = batchAugment(X_train_add, y_train_add, multiplier[i])
        X_train_aug = np.vstack((X_train_aug, X_train_add))
        print("New total dataset size: ",len(X_train_aug))
        y_train_aug = np.append(y_train_aug, y_train_add)
        print("")


unique, counts = np.unique(y_train_aug, return_counts=True)
print("New distribution of classes: ", counts)
~~~
Output:
~~~
Original distribution of classes:  [ 180 1980 2010 1260 1770 1650  360 1290 1260 1320 1800 1170 1890 1920 690 540  360  990 1080  180  300  270  330  450  240 1350  540  210  480  240 390  690  210  599  360 1080 330  180 1860  270  300  210  210]

New distribution of classes:  [1980 1980 2010 1260 1770 1650 2160 1290 1260 1320 1800 1170 1890 1920 2070 2160 2160  990 1080 1980 2100 1890 1980 1800 1920 1350 2160 2100 1920 1920 1950 2070 2100 1797 2160 1080 1980 1980 1860 1890 2100 2100 2100]
~~~

Here's a histogram of the new distribution. Not perfect, but vastly improved from the earlier spread.

![alt text][image2]


Lastly, I tripled the dataset size with rotation and transforms.
~~~
# Triple the dataset size by rotation and transformation
X_train_aug, y_train_aug = batchAugment(X_train_aug, y_train_aug, 1)    

print("Augmentation complete!")
~~~
Output:
~~~
New augmented size is:  234621
Augmentation complete!
~~~

#### Visualisation of the Pre-processing Steps
Here's what the images looked like before, during and after processing!


![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

