# Brain-Tumour-Segmentation-on-fMRI-data-using-U-nets.




[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Built an MRI data processing module and standardized the voxel data, trained the model by taking random sub-samples from the 3D image and applied the U-net model to segment tumor regions in 3D brain MRI image. 
- Implemented a custom loss function for model training (soft dice loss) and evaluated model performance by calculating sensitivity and specificity.

# Model Input!

 What is an MRI?
Magnetic resonance imaging (MRI) is an advanced imaging technique that is used to observe a variety of diseases and parts of the body.

 - Neural networks can analyze these images individually (as a radiologist would) or combine them into a single 3D volume to make predictions.

 - At a high level, MRI works by measuring the radio waves emitting by atoms subjected to a magnetic field.:
 
### Dataset

Our dataset is stored in the NifTI-1 format and we will be using the NiBabel library to interact with the files. Each training sample is composed of two separate files:

The first file is an image file containing a 4D array of MR image in the shape of (240, 240, 155, 4).

The first 3 dimensions are the X, Y, and Z values for each point in the 3D volume, which is commonly called a voxel.
The 4th dimension is the values for 4 different sequences
> 0: FLAIR: "Fluid Attenuated Inversion Recovery" (FLAIR)
> 1: T1w: "T1-weighted"
> 2: t1gd: "T1-weighted with gadolinium contrast enhancement" (T1-Gd)
> 3: T2w: "T2-weighted"

The second file in each training example is a label file containing a 3D array with the shape of (240, 240, 155).

The integer values in this array indicate the "label" for each voxel in the corresponding image files:
>0: background
1: edema
2: non-enhancing tumor
3: enhancing tumor


### Data Set preprocessing 
 Data Preprocessing using patches

- Generate sub-volumes
We are going to first generate "patches" of our data which you can think of as sub-volumes of the whole MR images.

 - The reason that we are generating patches is because a network that can process the entire volume at once will simply not fit inside our current environment's memory/GPU.
Therefore we will be using this common technique to generate spatially consistent sub-volumes of our data, which can be fed into our network.
Specifically, we will be generating randomly sampled sub-volumes of shape [160, 160, 16] from our images.

### Metrics
The Dice similarity coefficient, also known as the Sørensen–Dice index or simply Dice coefficient, is a statistical tool which measures the similarity between two sets of data. This index has become arguably the most broadly used tool in the validation of image segmentation algorithms created with AI, but it is a much more general concept which can be applied sets of data for a variety of applications including NLP.

The equation for this concept is:
> DC(X,Y) = 2 * |X| ∩ |Y| / (|X| + |Y|)

where X and Y are two sets
 - a set with vertical bars either side refers to the cardinality of the set, i.e. the number of elements in that set, e.g. |X| means the number of elements in set X
 - ∩ is used to represent the intersection of two sets, and means the elements that are common to both sets
 
### Loss function

The loss function used is also based on dice coefiicent.
The intuition here is that higher dice coefficient implies more overlap between the predicted and ground truth value. So we want to provide a high loss for low overlap and hence we do 1-DC to obtain the loss.

```sh
Loss(X,Y) = 1 - DC(X,Y)
```

#### Results
```              Edema Non-Enhancing Tumor Enhancing Tumor
Sensitivity  0.9085              0.9505          0.7891
Specificity  0.9848              0.9961           0.996
```
