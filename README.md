# **Traffic Sign Recognition** 

The aim of this project was to accurately classify traffic signs from the [German Traffic Signs dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) by means of a convolutional neural network.

**Build a Traffic Sign Recognition Network**

Here are the steps I followed:
* Load the dataset
* Explore, summarize and visualize the dataset
* Pre-process, balance and augment the dataset
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images


[//]: # (Image References)

[image1]: ./images/hist_orig.png "Visualisation"
[image2]: ./images/hist_new.png "New Visualisation"
[image3]: ./images/processed_images.png "Processed Images"
[image4]: ./images/test_web.png "Web Test"
[image5]: ./images/results_web1.png "Web Results 1"
[image6]: ./images/results_web2.png "Web Results 2"
[image7]: ./images/results_web3.png "Web Results 3"
[image8]: ./images/results_web4.png "Web Results 4"
[image9]: ./images/results_web5.png "Web Results 5"
[image10]: ./images/results_web6.png "Web Results 6"


### Data Set Summary & Exploration
Stats are as follows:
*Number of training examples = 34799
*Number of vaidation examples = 4410
*Number of testing examples = 12630
*Image data shape = (32, 32, 3)
*Number of classes = 43

####  Exploratory Visualization of the Dataset

The German Traffic Signs dataset is visualised in the below histogram. It is split into 43 categories, with the number of images in each category on the y-axis. Note the extremely unequal distribution of the dataset. This can and will introduce bias into the training model, and will be addressed below. Also, a low of ~200 examples for some classes is simply too low, and will need to be augmented.

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
```python
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
```

#### Augmentation (Transformation)
Here, I employed rotation (within a certain degree range) and warping through projective transforms (skimage). Projective transforms were chosen due to their similarity to changes in camera perspective.

Credits to Alex for helping me figure out projective transforms! His code and write-up helped clarify the frankly rather confusing usage of skimage projective transforms. His blog here.

NOTE: In earlier testing, doubling the dataset size without re-balancing it already yielded an accuracy of 95% on the validation data. While that seems nice, there is a high chance that the incorrect 5% stems from under-represented classes, rendering the prediction useless.
```python
from skimage.transform import ProjectiveTransform
from skimage.transform import rotate
from skimage.transform import warp

def randomTransform(image, intensity):
    
    # Rotate image within a set range, amplified by intensity of overall transform.
    rotation = 20 * intensity
    # print(image.shape)
    rotated = rotate(image, np.random.uniform(-rotation,rotation), mode = 'edge')
    
    # Projection transform on image, amplified by intensity.
    image_size = image.shape[0]
    magnitude = image_size * 0.3 * intensity
    tl_top = np.random.uniform(-magnitude, magnitude)     # Top left corner, top margin
    tl_left = np.random.uniform(-magnitude, magnitude)    # Top left corner, left margin
    bl_bottom = np.random.uniform(-magnitude, magnitude)  # Bottom left corner, bottom margin
    bl_left = np.random.uniform(-magnitude, magnitude)    # Bottom left corner, left margin
    tr_top = np.random.uniform(-magnitude, magnitude)     # Top right corner, top margin
    tr_right = np.random.uniform(-magnitude, magnitude)   # Top right corner, right margin
    br_bottom = np.random.uniform(-magnitude, magnitude)  # Bottom right corner, bottom margin
    br_right = np.random.uniform(-magnitude, magnitude)   # Bottom right corner, right margin
    
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

def batchAugment(X, y, multiplier = 2):
    X_train_aug = []
    y_train_aug = []
    for i in range(len(X)):
        for j in range(multiplier):
            augmented = randomTransform(X[i], 0.5)
            X_train_aug.append(augmented)
            y_train_aug.append(y[i])
        X_train_aug.append(X[i])
        y_train_aug.append(y[i])
        
    X_train_aug, y_train_aug = shuffle(X_train_aug, y_train_aug)
    print("New augmented size is: ", len(X_train_aug))
    return X_train_aug, y_train_aug
```

#### Altogether Now
Mini pipeline of the pre-processing phase.
```python
# Processing both training and validation sets
X_train_proc = batchPreprocess(X_train)
X_valid_proc = batchPreprocess(X_valid)

print("Pre-processing complete!")
```
Rebalancing based on representation of classes.
```python
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
```
Output:
```python
Original distribution of classes:  [ 180 1980 2010 1260 1770 1650  360 1290 1260 1320 1800 1170 1890 1920 690 540  360  990 1080  180  300  270  330  450  240 1350  540  210  480  240 390  690  210  599  360 1080 330  180 1860  270  300  210  210]

New distribution of classes:  [1980 1980 2010 1260 1770 1650 2160 1290 1260 1320 1800 1170 1890 1920 2070 2160 2160  990 1080 1980 2100 1890 1980 1800 1920 1350 2160 2100 1920 1920 1950 2070 2100 1797 2160 1080 1980 1980 1860 1890 2100 2100 2100]
```

Here's a histogram of the new distribution. Not perfect, but vastly improved from the earlier spread.

![alt text][image2]


Lastly, I tripled the dataset size with rotation and transforms.
```python
# Triple the dataset size by rotation and transformation
X_train_aug, y_train_aug = batchAugment(X_train_aug, y_train_aug, 1)    

print("Augmentation complete!")
```
Output:
```
New augmented size is:  234621
Augmentation complete!
```

#### Visualisation of the Pre-processing Steps
Here's what the images looked like before, during and after processing! Note how much more clearly defined the edges are after equalisation, and the variations produced by the augmentation step.

I was initially concerned about data loss when converting to grayscale, but found it barely affected performance at all. If anything, it allwowed the model to train faster with 1/3 the colour channels!

![alt text][image3]

### Model Architecture and Testing
What follows is an implementation of the LeNet-5 architecture. Only major changes made to this model were the output shape (43 classes instead of 10) and the network parameters.

Here, I attempted to address the unequal distribution of data by implementing a weighted cross entropy loss, heavily penalising mistakes on poorly represented classes, making the model 'pay more attention' to these examples, but performance actually suffered. Will investigate this technique in future models.

My final model was constructed as follows:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, output 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  output 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, output 10x10x16      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  output 5x5x16 				|
| Fully connected		| Input 400, output 120        									|
| RELU					|												|
| Dropout					|		0.85 rate										|
| Fully connected		| Input 120, output 84        									|
| RELU					|												|
| Dropout					|		0.85 rate										|
| Fully connected		| Input 84, output 43     									|
| Softmax				| With tf.reduce_mean loss and AdamOptimizer        									|
 
#### Parameters and Training Pipline
Values were chosen after much trial and error, and observation of model behaviour. To train the model, the AdamOptimizer was used.
```python
EPOCHS = 20
BATCH_SIZE = 128
rate = 0.0009

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
```

#### Results
During initial testing, I simply threw in the stock dataset with only grayscaling and OpenCV equalisation applied. Already, the network achieved a **92.3%** accuracy on the validation set. I did not try it on the test set at the time, but I imagine it would not have done very well. The dataset was very imbalanced, and the model would likely have been poorly generalised.

Curious, I attempted a 2x augmentation with no re-balancing. There was much more data now, but bias was strong. To my surprise, the network achieved **95.8%** on the validation set!

This felt a little bit like cheating, so I decided to try penalised classification at the cross entropy stage. In theory, I was going to heavily penalise poorly represented classes in an effort to make the network 'pay more attention', and hopefully generalise better for these classes. Sadly, the model returned a paltry **50%** accuracy, and it was back to refining my pre-processing pipline. ):

After re-balancing and 3x augmentation, I wound up with a validation accuracy of **96.8%**. Not bad for some simple augmentations!

I wound up not changing the model architecture much, as I felt this implementation was sufficient to hit my target of 93% accuracy. Further exploration in this area needed.

#### Test Set Pulled From The Web
The following are 6 images I pulled from the web. The only requirements I set were that they were German, and already existed in the signnames.csv database. Throwing in a completely new, never-before-seen sign with no respective label would be rather unfair.

Note image 6, which is a composition of 13: yield and 40: roundabout. I was curious to see how the model would react to a curveball like this. Let's see.

![alt text][image4]

### Results
```
Test Accuracy = 0.167
```
Ouch! Let's see what went wrong.

![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]


Look like the model only got Bumpy road and Slippery road right. Understandable, given these were the only two clear signs.

Road works and yield/roundabout were both completely off the mark. Perhaps more training with obstructed/shadowed/compound signs would have **yielded** (sorry) better results.

Perhaps dialing up my intensity for warping during my ```batchAugment()``` stage would have helped with the Stop sign, giving better recognition to objects that were clearly no longer circular.

Not sure what happened with pedestrians either. It seemed REALLY confident about the sign being Ahead only, which would no doubt spell disaster for said pedestrians.

### Conclusion
Overall, the model performed rather well in initial testing, but faltered on images taken from the wild. To be fair, the images from the web differed greatly from the supplied training set. Further refinement, augmentation and perhaps simply more, varied data could help the model generalise and deal with the challenge better. 

Moving forward, I'd like to try implementing models from other successful papers to gain a better intuition of the design choices when building a model. I note that the LeNet model was initially built for low-res classification of the MNIST dataset, a considerably simpler and less varied challenge.

