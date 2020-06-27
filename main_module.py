
# coding: utf-8

import keras
import json
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K 

import util


# <a name="1"></a>
# # 1 Dataset
# <a name="1-1"></a>
# ## 1.1 What is an MRI?
# 
# Magnetic resonance imaging (MRI) is an advanced imaging technique that is used to observe a variety of diseases and parts of the body. 
# 
# 
# At a high level, MRI works by measuring the radio waves emitting by atoms subjected to a magnetic field. 
# 
# <img src="https://miro.medium.com/max/1740/1*yC1Bt3IOzNv8Pp7t1v7F1Q.png">
# 
# 
# <a name="1-2"></a>

# ## 1.2 MRI Data Processing
# 
# We often encounter MR images in the [DICOM format](https://en.wikipedia.org/wiki/DICOM). 
# - The DICOM format is the output format for most commercial MRI scanners. This type of data can be processed using the [pydicom](https://pydicom.github.io/pydicom/stable/getting_started.html) Python library. 
# 

# <a name="1-3"></a>
# ## 1.3 Exploring the Dataset
# 
# Our dataset is stored in the [NifTI-1 format](https://nifti.nimh.nih.gov/nifti-1/) and we will be using the [NiBabel library](https://github.com/nipy/nibabel) to interact with the files. Each training sample is composed of two separate files:
# 
# The first file is an image file containing a 4D array of MR image in the shape of (240, 240, 155, 4). 
# -  The first 3 dimensions are the X, Y, and Z values for each point in the 3D volume, which is commonly called a voxel. 
# - The 4th dimension is the values for 4 different sequences
#     - 0: FLAIR: "Fluid Attenuated Inversion Recovery" (FLAIR)
#     - 1: T1w: "T1-weighted"
#     - 2: t1gd: "T1-weighted with gadolinium contrast enhancement" (T1-Gd)
#     - 3: T2w: "T2-weighted"
# 
# The second file in each training example is a label file containing a 3D array with the shape of (240, 240, 155).  
# - The integer values in this array indicate the "label" for each voxel in the corresponding image files:
#     - 0: background
#     - 1: edema
#     - 2: non-enhancing tumor
#     - 3: enhancing tumor
# 
# We have access to a total of 484 training images which we will be splitting into a training (80%) and validation (20%) dataset.
# 
# Let's begin by looking at one single case and visualizing the data! You have access to 10 different cases via this notebook and we strongly encourage you to explore the data further on your own.

# We'll use the [NiBabel library](https://nipy.org/nibabel/nibabel_images.html) to load the image and label for a case. The function is shown below to give you a sense of how it works. 

# In[2]:


# set home directory and data directory
HOME_DIR = "./BraTS-Data/"
DATA_DIR = HOME_DIR

def load_case(image_nifty_file, label_nifty_file):
    # load the image and label file, get the image content and return a numpy array for each
    image = np.array(nib.load(image_nifty_file).get_fdata())
    label = np.array(nib.load(label_nifty_file).get_fdata())
    
    return image, label


# 
# The colors correspond to each class.
# - Red is edema
# - Green is a non-enhancing tumor
# - Blue is an enhancing tumor. 
# 

# In[3]:


image, label = load_case(DATA_DIR + "imagesTr/BRATS_003.nii.gz", DATA_DIR + "labelsTr/BRATS_003.nii.gz")
image = util.get_labeled_image(image, label)

util.plot_image_grid(image)



# In[4]:


image, label = load_case(DATA_DIR + "imagesTr/BRATS_003.nii.gz", DATA_DIR + "labelsTr/BRATS_003.nii.gz")
util.visualize_data_gif(util.get_labeled_image(image, label))



# <a name="1-4"></a>
# ## 1.4 Data Preprocessing using patches
# 
# 
# ##### Generate sub-volumes
# 
# We are going to first generate "patches" of our data which you can think of as sub-volumes of the whole MR images. 
# - The reason that we are generating patches is because a network that can process the entire volume at once will simply not fit inside our current environment's memory/GPU.
# - Therefore we will be using this common technique to generate spatially consistent sub-volumes of our data, which can be fed into our network.
# - Specifically, we will be generating randomly sampled sub-volumes of shape \[160, 160, 16\] from our images. 
# - Furthermore, given that a large portion of the MRI volumes are just brain tissue or black background without any tumors, we want to make sure that we pick patches that at least include some amount of tumor data. 
# - Therefore, we are only going to pick patches that have at most 95% non-tumor regions (so at least 5% tumor). 
# - We do this by filtering the volumes based on the values present in the background labels.
# 
# ##### Standardization (mean 0, stdev 1)
# 
# Lastly, given that the values in MR images cover a very wide range, we will standardize the values to have a mean of zero and standard deviation of 1. 
# - This is a common technique in deep image processing since standardization makes it much easier for the network to learn.
# 
# Let's walk through these steps in the following exercises.

# <a name="1-4-1"></a>
# ### 1.4.1 Sub-volume Sampling
# Fill in the function below takes in:
# - a 4D image (shape: \[240, 240, 155, 4\])
# - its 3D label (shape: \[240, 240, 155\]) arrays, 
# 
# The function returns:
#  - A randomly generated sub-volume of size \[160, 160, 16\]
#  - Its corresponding label in a 1-hot format which has the shape \[3, 160, 160, 160\]
# 


# In[10]:


def get_sub_volume(image, label, 
                   orig_x = 240, orig_y = 240, orig_z = 155, 
                   output_x = 160, output_y = 160, output_z = 16,
                   num_classes = 4, max_tries = 1000, 
                   background_threshold=0.95):
    """
    Extract random sub-volume from original images.

    Args:
        image (np.array): original image, 
            of shape (orig_x, orig_y, orig_z, num_channels)
        label (np.array): original label. 
            labels coded using discrete values rather than
            a separate dimension, 
            so this is of shape (orig_x, orig_y, orig_z)
        orig_x (int): x_dim of input image
        orig_y (int): y_dim of input image
        orig_z (int): z_dim of input image
        output_x (int): desired x_dim of output
        output_y (int): desired y_dim of output
        output_z (int): desired z_dim of output
        num_classes (int): number of class labels
        max_tries (int): maximum trials to do when sampling
        background_threshold (float): limit on the fraction 
            of the sample which can be the background

    returns:
        X (np.array): sample of original image of dimension 
            (num_channels, output_x, output_y, output_z)
        y (np.array): labels which correspond to X, of dimension 
            (num_classes, output_x, output_y, output_z)
    """
    # Initialize features and labels with `None`
    X = None
    y = None

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    tries = 0
    
    while tries < max_tries:
        # randomly sample sub-volume by sampling the corner voxel
        # hint: make sure to leave enough room for the output dimensions!
        start_x = np.random.randint(orig_x - output_x + 1)
        start_y = np.random.randint(orig_y - output_y + 1)
        start_z = np.random.randint(orig_z - output_z + 1)
        print(label)
        # extract relevant area of label
        y = label[start_x: start_x + output_x,
                  start_y: start_y + output_y,
                  start_z: start_z + output_z]
        
        # One-hot encode the categories.
        # This adds a 4th dimension, 'num_classes'
        # (output_x, output_y, output_z, num_classes)
        print(y)
        y = keras.utils.to_categorical(y, num_classes=num_classes)
        print(y)
        # compute the background ratio
        bgrd_ratio = np.sum(y[:,:,:,0])/(output_x * output_y * output_z)

        # increment tries counter
        tries += 1

        # if background ratio is below the desired threshold,
        # use that sub-volume.
        # otherwise continue the loop and try another random sub-volume
        if bgrd_ratio < background_threshold:

            # make copy of the sub-volume
            X = np.copy(image[start_x: start_x + output_x,
                              start_y: start_y + output_y,
                              start_z: start_z + output_z, :])
            
            # change dimension of X
            # from (x_dim, y_dim, z_dim, num_channels)
            # to (num_channels, x_dim, y_dim, z_dim)
            X = np.moveaxis(X,-1,0)

            # change dimension of y
            # from (x_dim, y_dim, z_dim, num_classes)
            # to (num_classes, x_dim, y_dim, z_dim)
            y = np.moveaxis(y,-1,0)
            ### END CODE HERE ###
            
            # take a subset of y that excludes the background class
            # in the 'num_classes' dimension
            y = y[1:, :, :, :]
    
            return X, y

    # if we've tried max_tries number of samples
    # Give up in order to avoid looping forever.
    print(f"Tried {tries} times to find a sub-volume. Giving up...")


# ### Test Case:

# In[11]:


np.random.seed(3)

image = np.zeros((4, 4, 3, 1))
label = np.zeros((4, 4, 3))
for i in range(4):
    for j in range(4):
        for k in range(3):
            image[i, j, k, 0] = i*j*k
            label[i, j, k] = k

print("image:")
for k in range(3):
    print(f"z = {k}")
    print(image[:, :, k, 0])
print("\n")
print("label:")
for k in range(3):
    print(f"z = {k}")
    print(label[:, :, k])


# #### Test: Extracting (2, 2, 2) sub-volume

# In[12]:


sample_image, sample_label = get_sub_volume(image, 
                                            label,
                                            orig_x=4, 
                                            orig_y=4, 
                                            orig_z=3,
                                            output_x=2, 
                                            output_y=2, 
                                            output_z=2,
                                            num_classes = 3)

print("Sampled Image:")
for k in range(2):
    print("z = " + str(k))
    print(sample_image[0, :, :, k])


# #### Expected output:
# 
# ```Python
# Sampled Image:
# z = 0
# [[0. 2.]
#  [0. 3.]]
# z = 1
# [[0. 4.]
#  [0. 6.]]
# ```

# In[92]:


print("Sampled Label:")
for c in range(2):
    print("class = " + str(c))
    for k in range(2):
        print("z = " + str(k))
        print(sample_label[c, :, :, k])


# #### Expected output:
# 
# ```Python
# Sampled Label:
# class = 0
# z = 0
# [[1. 1.]
#  [1. 1.]]
# z = 1
# [[0. 0.]
#  [0. 0.]]
# class = 1
# z = 0
# [[0. 0.]
#  [0. 0.]]
# z = 1
# [[1. 1.]
#  [1. 1.]]
# ```


# In[13]:


image, label = load_case(DATA_DIR + "imagesTr/BRATS_001.nii.gz", DATA_DIR + "labelsTr/BRATS_001.nii.gz")
X, y = get_sub_volume(image, label)
# enhancing tumor is channel 2 in the class label
# you can change indexer for y to look at different classes
util.visualize_patch(X[0, :, :, :], y[2])


# <a name="1-4-2"></a>
# ### 1.4.2 Standardization
# 

# <details>    
# <summary>
#     <font size="3" color="darkgreen"><b>Hints</b></font>
# </summary>
# <p>
# <ul>
#     <li> Check that the standard deviation is not zero before dividing by it.
# </ul>
# </p>

# In[14]:


# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def standardize(image):
    """
    Standardize mean and standard deviation 
        of each channel and z_dimension.

    Args:
        image (np.array): input image, 
            shape (num_channels, dim_x, dim_y, dim_z)

    Returns:
        standardized_image (np.array): standardized version of input image
    """
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # initialize to array of zeros, with same shape as the image
    standardized_image = np.zeros(image.shape)

    # iterate over channels
    for c in range(image.shape[0]):
        # iterate over the `z` dimension
        for z in range(image.shape[3]):
            # get a slice of the image 
            # at channel c and z-th dimension `z`
            image_slice = image[c,:,:,z]

            # subtract the mean
            centered = image_slice - np.mean(image_slice)

            # divide by the standard deviation
            if np.std(centered) != 0:
                centered_scaled = centered / np.std(centered)

            # update  the slice of standardized image
            # with the scaled centered and scaled image
            standardized_image[c, :, :, z] = centered_scaled

    ### END CODE HERE ###

    return standardized_image


# And to sanity check, let's look at the output of our function:

# In[15]:



X_norm = standardize(X)
print("standard deviation for a slice should be 1.0")
print(f"stddv for X_norm[0, :, :, 0]: {X_norm[0,:,:,0].std():.2f}")


# Let's visualize our patch again just to make sure (it won't look different since the `imshow` function we use to visualize automatically normalizes the pixels when displaying in black and white).

# In[16]:


util.visualize_patch(X_norm[0, :, :, :], y[2])


# <a name="2"></a>
# # 2 Model: 3D U-Net
# - This architecture will take advantage of the volumetric shape of MR images and is one of the best performing models for this task. 
# - Feel free to familiarize yourself with the architecture by reading [this paper](https://arxiv.org/abs/1606.06650).
# 
# <img src="https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png" width="50%">

# <a name="3"></a>
# # 3 Metrics

# <a name="3-1"></a>
# ## 3.1 Dice Similarity Coefficient
# 
# Aside from the architecture, one of the most important elements of any deep learning method is the choice of our loss function. 
# 
# A natural choice that you may be familiar with is the cross-entropy loss function. 
# - However, this loss function is not ideal for segmentation tasks due to heavy class imbalance (there are typically not many positive regions). 
# 
# A much more common loss for segmentation tasks is the Dice similarity coefficient, which is a measure of how well two contours overlap. 
# - The Dice index ranges from 0 (complete mismatch) 
# - To 1 (perfect match).
# 
# In general, for two sets $A$ and $B$, the Dice similarity coefficient is defined as: 
# $$\text{DSC}(A, B) = \frac{2 \times |A \cap B|}{|A| + |B|}.$$
# 
# Here we can interpret $A$ and $B$ as sets of voxels, $A$ being the predicted tumor region and $B$ being the ground truth. 
# 
# Our model will map each voxel to 0 or 1
# - 0 means it is a background voxel
# - 1 means it is part of the segmented region.
# 
# In the dice coefficient, the variables in the formula are:
# - $x$ : the input image
# - $f(x)$ : the model output (prediction)
# - $y$ : the label (actual ground truth)
# 
# The dice coefficient "DSC" is:
# 
# $$\text{DSC}(f, x, y) = \frac{2 \times \sum_{i, j} f(x)_{ij} \times y_{ij} + \epsilon}{\sum_{i,j} f(x)_{ij} + \sum_{i, j} y_{ij} + \epsilon}$$
# 
# - $\epsilon$ is a small number that is added to avoid division by zero
# 
# <img src="https://www.researchgate.net/publication/328671987/figure/fig4/AS:688210103529478@1541093483784/Calculation-of-the-Dice-similarity-coefficient-The-deformed-contour-of-the-liver-from.ppm" width="30%">
# 
# [Image Source](https://www.researchgate.net/figure/Calculation-of-the-Dice-similarity-coefficient-The-deformed-contour-of-the-liver-from_fig4_328671987)
# 
# 

# In[17]:


# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def single_class_dice_coefficient(y_true, y_pred, axis=(0, 1, 2), 
                                  epsilon=0.00001):
    """
    Compute dice coefficient for single class.

    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for single class.
                                    shape: (x_dim, y_dim, z_dim)
        y_pred (Tensorflow tensor): tensor of predictions for single class.
                                    shape: (x_dim, y_dim, z_dim)
        axis (tuple): spatial axes to sum over when computing numerator and
                      denominator of dice coefficient.
                      Hint: pass this as the 'axis' argument to the K.sum function.
        epsilon (float): small constant added to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_coefficient (float): computed value of dice coefficient.     
    """

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    dice_numerator = 2*(K.sum(y_true*y_pred, axis=axis))+epsilon
    dice_denominator = K.sum(y_true*y_true,axis=axis)+K.sum(y_pred*y_pred,axis=axis)+epsilon
    dice_coefficient = dice_numerator/dice_denominator
    
    ### END CODE HERE ###

    return dice_coefficient


# In[18]:


# TEST CASES
sess = K.get_session()
#sess = tf.compat.v1.Session()
with sess.as_default() as sess:
    pred = np.expand_dims(np.eye(2), -1)
    label = np.expand_dims(np.array([[1.0, 1.0], [0.0, 0.0]]), -1)

    print("Test Case #1")
    print("pred:")
    print(pred[:, :, 0])
    print("label:")
    print(label[:, :, 0])

    # choosing a large epsilon to help check for implementation errors
    dc = single_class_dice_coefficient(pred, label,epsilon=1)
    print(f"dice coefficient: {dc.eval():.4f}")

    print("\n")

    print("Test Case #2")
    pred = np.expand_dims(np.eye(2), -1)
    label = np.expand_dims(np.array([[1.0, 1.0], [0.0, 1.0]]), -1)

    print("pred:")
    print(pred[:, :, 0])
    print("label:")
    print(label[:, :, 0])

    # choosing a large epsilon to help check for implementation errors
    dc = single_class_dice_coefficient(pred, label,epsilon=1)
    print(f"dice_coefficient: {dc.eval():.4f}")


# ##### Expected output
# 
# If you get a different result, please check that you implemented the equation completely.
# ```Python
# Test Case #1
# pred:
# [[1. 0.]
#  [0. 1.]]
# label:
# [[1. 1.]
#  [0. 0.]]
# dice coefficient: 0.6000
# 
# 
# Test Case #2
# pred:
# [[1. 0.]
#  [0. 1.]]
# label:
# [[1. 1.]
#  [0. 1.]]
# dice_coefficient: 0.8333
# ```

#
# 
# 

# In[19]:


def dice_coefficient(y_true, y_pred, axis=(1, 2, 3), 
                     epsilon=0.00001):
    """
    Compute mean dice coefficient over all abnormality classes.

    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        y_pred (Tensorflow tensor): tensor of predictions for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        axis (tuple): spatial axes to sum over when computing numerator and
                      denominator of dice coefficient.
                      Hint: pass this as the 'axis' argument to the K.sum
                            and K.mean functions.
        epsilon (float): small constant add to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_coefficient (float): computed value of dice coefficient.     
    """

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    dice_numerator = 2. * K.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) + epsilon
    dice_coefficient = K.mean((dice_numerator)/(dice_denominator))
    
    ### END CODE HERE ###

    return dice_coefficient


# In[20]:


# TEST CASES
sess = K.get_session()
with sess.as_default() as sess:
    pred = np.expand_dims(np.expand_dims(np.eye(2), 0), -1)
    label = np.expand_dims(np.expand_dims(np.array([[1.0, 1.0], [0.0, 0.0]]), 0), -1)

    print("Test Case #1")
    print("pred:")
    print(pred[0, :, :, 0])
    print("label:")
    print(label[0, :, :, 0])

    dc = dice_coefficient(pred, label,epsilon=1)
    print(f"dice coefficient: {dc.eval():.4f}")

    print("\n")

    print("Test Case #2")
    pred = np.expand_dims(np.expand_dims(np.eye(2), 0), -1)
    label = np.expand_dims(np.expand_dims(np.array([[1.0, 1.0], [0.0, 1.0]]), 0), -1)


    print("pred:")
    print(pred[0, :, :, 0])
    print("label:")
    print(label[0, :, :, 0])

    dc = dice_coefficient(pred, label,epsilon=1)
    print(f"dice coefficient: {dc.eval():.4f}")
    print("\n")


    print("Test Case #3")
    pred = np.zeros((2, 2, 2, 1))
    pred[0, :, :, :] = np.expand_dims(np.eye(2), -1)
    pred[1, :, :, :] = np.expand_dims(np.eye(2), -1)
    
    label = np.zeros((2, 2, 2, 1))
    label[0, :, :, :] = np.expand_dims(np.array([[1.0, 1.0], [0.0, 0.0]]), -1)
    label[1, :, :, :] = np.expand_dims(np.array([[1.0, 1.0], [0.0, 1.0]]), -1)

    print("pred:")
    print("class = 0")
    print(pred[0, :, :, 0])
    print("class = 1")
    print(pred[1, :, :, 0])
    print("label:")
    print("class = 0")
    print(label[0, :, :, 0])
    print("class = 1")
    print(label[1, :, :, 0])

    dc = dice_coefficient(pred, label,epsilon=1)
    print(f"dice coefficient: {dc.eval():.4f}")



# 
# While the Dice Coefficient makes intuitive sense, it is not the best for training. 
# - This is because it takes in discrete values (zeros and ones). 
# - The model outputs *probabilities* that each pixel is, say, a tumor or not, and we want to be able to backpropagate through those outputs. 
# 
#
# ### Multi-Class Soft Dice Loss
# 
# We've explained the single class case for simplicity, but the multi-class generalization is exactly the same as that of the dice coefficient. 
# - Since you've already implemented the multi-class dice coefficient, we'll have you jump directly to the multi-class soft dice loss.
# 
# For any number of categories of diseases, the expression becomes:
# 
# $$\mathcal{L}_{Dice}(p, q) = 1 - \frac{1}{N} \sum_{c=1}^{C} \frac{2\times\sum_{i, j} p_{cij}q_{cij} + \epsilon}{\left(\sum_{i, j} p_{cij}^2 \right) + \left(\sum_{i, j} q_{cij}^2 \right) + \epsilon}$$
# 
# Please implement the soft dice loss below!
# 
# As before, you will use K.mean()
# - Apply the average the mean to ratio that you'll calculate in the last line of code that you'll implement.

# In[21]:


def soft_dice_loss(y_true, y_pred, axis=(1, 2, 3), 
                   epsilon=0.00001):
    """
    Compute mean soft dice loss over all abnormality classes.

    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        y_pred (Tensorflow tensor): tensor of soft predictions for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        axis (tuple): spatial axes to sum over when computing numerator and
                      denominator in formula for dice loss.
                      Hint: pass this as the 'axis' argument to the K.sum
                            and K.mean functions.
        epsilon (float): small constant added to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_loss (float): computed value of dice loss.     
    """

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    dice_numerator = 2. * K.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = K.sum(y_true**2, axis=axis) + K.sum(y_pred**2, axis=axis) + epsilon
    dice_loss = 1 - K.mean((dice_numerator)/(dice_denominator))


    ### END CODE HERE ###

    return dice_loss


# In[24]:


sess = K.get_session()
with sess.as_default() as sess:
    pred = np.expand_dims(np.expand_dims(np.eye(2), 0), -1)
    label = np.expand_dims(np.expand_dims(np.array([[1.0, 1.0], [0.0, 0.0]]), 0), -1)
    
    print("Test Case #3")
    pred = np.expand_dims(np.expand_dims(np.eye(2), 0), -1)
    label = np.expand_dims(np.expand_dims(np.array([[1.0, 1.0], [0.0, 1.0]]), 0), -1)

    print("pred:")
    print(pred[0, :, :, 0])
    print("label:")
    print(label[0, :, :, 0])

    dc = soft_dice_loss(pred, label, epsilon=1)
    print(f"soft dice loss: {dc.eval():.4f}")



# <a name="4"></a>
# # 4 Create and Train the model
# 
# Once you've finished implementing the soft dice loss, we can create the model! 
# 
# We'll use the `unet_model_3d` function in `utils` which we implemented for you.
# - This creates the model architecture and compiles the model with the specified loss functions and metrics. 

# In[26]:


model = util.unet_model_3d(loss_function=soft_dice_loss, metrics=[dice_coefficient])



# In[27]:


# run this cell if you didn't run the training cell in section 4.1
base_dir = HOME_DIR + "processed/"
with open(base_dir + "config.json") as json_file:
    config = json.load(json_file)
# Get generators for training and validation sets
train_generator = util.VolumeDataGenerator(config["train"], base_dir + "train/", batch_size=3, dim=(160, 160, 16), verbose=0)
valid_generator = util.VolumeDataGenerator(config["valid"], base_dir + "valid/", batch_size=3, dim=(160, 160, 16), verbose=0)


# In[28]:


model.load_weights(HOME_DIR + "model_pretrained.hdf5")


# In[29]:


model.summary()


# <a name="5"></a>
# # 5 Evaluation
# 
# Now that we have a trained model, we'll learn to extract its predictions and evaluate its performance on scans from our validation set.

# <a name="5-1"></a>
# ## 5.1 Overall Performance

# First let's measure the overall performance on the validation set. 
# - We can do this by calling the keras [evaluate_generator](https://keras.io/models/model/#evaluate_generator) function and passing in the validation generator, created in section 4.1. 
# 
# #### Using the validation set for testing
# - Note: since we didn't do cross validation tuning on the final model, it's okay to use the validation set.
# - For real life implementations, however, you would want to do cross validation as usual to choose hyperparamters and then use a hold out test set to assess performance
# 
# Python Code for measuring the overall performance on the validation set:
# 
# ```python
# val_loss, val_dice = model.evaluate_generator(valid_generator)
# 
# print(f"validation soft dice loss: {val_loss:.4f}")
# print(f"validation dice coefficient: {val_dice:.4f}")
# ```
# 
# In[30]:


util.visualize_patch(X_norm[0, :, :, :], y[2])


# #### Add a 'batch' dimension
# We can extract predictions by calling `model.predict` on the patch. 
# - We'll add an `images_per_batch` dimension, since the `predict` method is written to take in batches. 
# In[31]:


X_norm_with_batch_dimension = np.expand_dims(X_norm, axis=0)
patch_pred = model.predict(X_norm_with_batch_dimension)



# In[32]:


# set threshold.
threshold = 0.5

# use threshold to get hard predictions
patch_pred[patch_pred > threshold] = 1.0
patch_pred[patch_pred <= threshold] = 0.0


# Now let's visualize the original patch and ground truth alongside our thresholded predictions.

# In[33]:


print("Patch and ground truth")
util.visualize_patch(X_norm[0, :, :, :], y[2])
plt.show()
print("Patch and prediction")
util.visualize_patch(X_norm[0, :, :, :], patch_pred[0, 2, :, :, :])
plt.show()


# #### Sensitivity and Specificity
# 
# The model is covering some of the relevant areas, but it's definitely not perfect. 
# - To quantify its performance, we can use per-pixel sensitivity and specificity. 
# # 
# $$\text{sensitivity} = \frac{\text{true positives}}{\text{true positives} + \text{false negatives}}$$
# 
# $$\text{specificity} = \frac{\text{true negatives}}{\text{true negatives} + \text{false positives}}$$
# 
# Below let's write a function to compute the sensitivity and specificity per output class.

# 
# <details>    
# <summary>
#     <font size="3" color="darkgreen"><b>Hints</b></font>
# </summary>
# <p>
# <ul>
#     <li>Recall that a true positive occurs when the class prediction is equal to 1, and the class label is also equal to 1</li>
#     <li>Use <a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html" > numpy.sum() </a> </li>
# 
# </ul>
# </p>

# In[34]:


# UNQ_C6 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def compute_class_sens_spec(pred, label, class_num):
    """
    Compute sensitivity and specificity for a particular example
    for a given class.

    Args:
        pred (np.array): binary arrary of predictions, shape is
                         (num classes, height, width, depth).
        label (np.array): binary array of labels, shape is
                          (num classes, height, width, depth).
        class_num (int): number between 0 - (num_classes -1) which says
                         which prediction class to compute statistics
                         for.

    Returns:
        sensitivity (float): precision for given class_num.
        specificity (float): recall for given class_num
    """

    # extract sub-array for specified class
    class_pred = pred[class_num]
    class_label = label[class_num]

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # compute true positives, false positives, 
    # true negatives, false negatives
    tp = np.sum((class_pred == 1) & (class_label == 1))
    tn = np.sum((class_pred == 0) & (class_label == 0))
    fp = np.sum((class_pred == 1) & (class_label == 0))
    fn = np.sum((class_pred == 0) & (class_label == 1))

    # compute sensitivity and specificity
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    ### END CODE HERE ###

    return sensitivity, specificity


# In[35]:


# TEST CASES
pred = np.expand_dims(np.expand_dims(np.eye(2), 0), -1)
label = np.expand_dims(np.expand_dims(np.array([[1.0, 1.0], [0.0, 0.0]]), 0), -1)

print("Test Case #1")
print("pred:")
print(pred[0, :, :, 0])
print("label:")
print(label[0, :, :, 0])

sensitivity, specificity = compute_class_sens_spec(pred, label, 0)
print(f"sensitivity: {sensitivity:.4f}")
print(f"specificity: {specificity:.4f}")


# #### Expected output:
# 
# ```Python
# Test Case #1
# pred:
# [[1. 0.]
#  [0. 1.]]
# label:
# [[1. 1.]
#  [0. 0.]]
# sensitivity: 0.5000
# specificity: 0.5000
# ```

# In[ ]:


print("Test Case #2")

pred = np.expand_dims(np.expand_dims(np.eye(2), 0), -1)
label = np.expand_dims(np.expand_dims(np.array([[1.0, 1.0], [0.0, 1.0]]), 0), -1)

print("pred:")
print(pred[0, :, :, 0])
print("label:")
print(label[0, :, :, 0])

sensitivity, specificity = compute_class_sens_spec(pred, label, 0)
print(f"sensitivity: {sensitivity:.4f}")
print(f"specificity: {specificity:.4f}")


# In[ ]:


from IPython.display import display
print("Test Case #3")

df = pd.DataFrame({'y_test': [1,1,0,0,0,0,0,0,0,1,1,1,1,1],
                   'preds_test': [1,1,0,0,0,1,1,1,1,0,0,0,0,0],
                   'category': ['TP','TP','TN','TN','TN','FP','FP','FP','FP','FN','FN','FN','FN','FN']
                  })

display(df)
pred = np.array( [df['preds_test']])
label = np.array( [df['y_test']])

sensitivity, specificity = compute_class_sens_spec(pred, label, 0)
print(f"sensitivity: {sensitivity:.4f}")
print(f"specificity: {specificity:.4f}")


# #### Sensitivity and Specificity for the patch prediction
# 
# Next let's compute the sensitivity and specificity on that patch for expanding tumors. 

# In[36]:


sensitivity, specificity = compute_class_sens_spec(patch_pred[0], y, 2)

print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")


# #### Expected output:
# 
# ```Python
# Sensitivity: 0.7891
# Specificity: 0.9960
# ```

# We can also display the sensitivity and specificity for each class.

# In[37]:


def get_sens_spec_df(pred, label):
    patch_metrics = pd.DataFrame(
        columns = ['Edema', 
                   'Non-Enhancing Tumor', 
                   'Enhancing Tumor'], 
        index = ['Sensitivity',
                 'Specificity'])
    
    for i, class_name in enumerate(patch_metrics.columns):
        sens, spec = compute_class_sens_spec(pred, label, i)
        patch_metrics.loc['Sensitivity', class_name] = round(sens,4)
        patch_metrics.loc['Specificity', class_name] = round(spec,4)

    return patch_metrics


# In[38]:


df = get_sens_spec_df(patch_pred[0], y)

print(df)



# In[ ]:


image, label = load_case(DATA_DIR + "imagesTr/BRATS_003.nii.gz", DATA_DIR + "labelsTr/BRATS_003.nii.gz")
pred = util.predict_and_viz(image, label, model, .5, loc=(130, 130, 77))                


# #### Check how well the predictions do
# 
# We can see some of the discrepancies between the model and the ground truth visually. 
# - We can also use the functions we wrote previously to compute sensitivity and specificity for each class over the whole scan.
# - First we need to format the label and prediction to match our functions expect.

# In[ ]:


whole_scan_label = keras.utils.to_categorical(label, num_classes = 4)
whole_scan_pred = pred

# move axis to match shape expected in functions
whole_scan_label = np.moveaxis(whole_scan_label, 3 ,0)[1:4]
whole_scan_pred = np.moveaxis(whole_scan_pred, 3, 0)[1:4]


# Now we can compute sensitivity and specificity for each class just like before.

# In[ ]:


whole_scan_df = get_sens_spec_df(whole_scan_pred, whole_scan_label)

print(whole_scan_df)


# 
# 
# 
# 
