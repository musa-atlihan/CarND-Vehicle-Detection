# Vehicle Detection Project

The aim of this project is to detect the vehicles in images using HOG features.

## HOG Features

When the class of objects varies in color, structural features like gradients or edges might give a more robost representation. This way, different signatures for different shapes can be obtained. However, the signature for a shape should have enough flexibility to accommodate small variations in orientation, size, etc. The problem with using the gradient values directly is that the signature becomes too sensitive. To allow some variability the histogram of oriented gradients (HOG), which is a more flexible approach, will be implemented.

![hog images for different cell sizes](./images/car-hog-npixels.png)

Representations of the HOG features of a car with different numbers of pixels in each cell.  

To get the HOG features, a helper function named `get_hog` is [implemented](./feature-extraction.ipynb) using the [scikit-image](http://scikit-image.org/) package to compute the histogram of gradient directions (orientations) of an image. The gradient samples are distributed into a nine orientation bins as default. Nine bins were enough for the accuracy in small scales. Here in the implementation of the histogram, it is not used just the count of the samples in each direction. Instead, the gradient magnitudes of each sample are summed up. This way, the stronger gradients contribute more weight to their orientation bin and thus, the effect of noise is reduced. By dividing the images into different size of cells (e.g. 2x2, 8x8 and 16x16) and by getting the histogram for the directions and the gradient magnitudes of the pixels in these cells, the representations of the structures emerge. The main advantage is that HOG gives the ability to accept small variations in the shape, while keeping the signature distinct enough. The sensitivity for the features can be tweaked by varying parameters such as the number of orientation bins, grid of cells, cell sizes, the overlap between cells, block normalization, etc ([Navneet Dalal and Bill Triggs, 2005](http://ieeexplore.ieee.org/document/1467360/)).


## The Dataset

The dataset provided for the project is a combination of datasets from various sources. These sources are [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html) (GTI), [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/) and extra images extracted from [Udacity](https://www.udacity.com/)'s [project video](./project_video.mp4). The dataset has two classes of images (all have a size of 64x64 pixels) with the labels *vehicle* and *non-vehicle*. The number of images with the *vehicle* label and their sources can be seen below. The sets of *GTI Far*, *GTI left*, *GTI Right* and *GTI MidClose* are from the same source of GTI with different viewpoints. Together with the *KITTI* examples they constitute the whole set of *vehicle* images.

![vehicle image numbers](./images/vehicle-sets.png)

Some of the examples from *vehicle* sources are given below.

![gti far set](./images/gti_far.png)

![gti right set](./images/gti_right.png)

![gti left set](./images/gti_left.png)

![gti midclose set](./images/gti_midclose.png)

![gti kitti set](./images/kitti.png)

The *non-vehicle* images are taken from GTI with extra captured images from the project video. The number of examples for each source is given below.

![non-vehicle image numbers](./images/non-vehicle-sets.png)

Some of the examples from *non-vehicle* sources are given below.

![gti non vehicle set](./images/non_vehicle_gti.png)

![extras non vehicle set](./images/non_vehicle_extras.png)

To check the balance between *vehicle* and *non-vehicle* classes the total number of examples for each class is given below.

![vehicle non-vehicle image numbers](./n-classes.png)

As can be seen from the figure, the classes of the dataset seem to be roughly balanced, thus no augmentation will be applied for balancing.

## Support Vector Machines



## Feature Exploration


There are several ways to extract the HOG features. Most important feature extracting parameters are pixels per cell (`pix_per_cell`), cells per block (`cell_per_block`) and the number of orientation bins (`orient`). `pix_per_cell` defines the number of pixels in cells, in each cell the gradient orientations are computed and the directions are grouped into orientation bins (the number of bins is controlled by the parameter `orient`). It is reported to be convenient to choose a number between 4 to 9 for the number of orientation bins ([Navneet Dalal and Bill Triggs, 2005](http://ieeexplore.ieee.org/document/1467360/), and thus, the default is set to nine. However, it is hard to figure out the convenient values for the pixel number in each cell and the cell number per block. These parameters depends on the dataset in use. For instance, choosing larger cell sizes increases the magnitudes with a decrease in the resolution of the representations, however the efficency of the representations depends of the objects in the dataset. In addition, a block of cells (defined with the parameter `cell_per_block`) are chosen for the block normalization and reduces the effects of the lighting varions on the images. However, choosing large block sizes does not mean a better normalization. It is reported that for the human detection dataset, choosing the block sizes as wide as the human limbs performs best ([Navneet Dalal and Bill Triggs, 2005](http://ieeexplore.ieee.org/document/1467360/)). Thus, these parameters depend on the object sizes in a dataset and need exploration. To search for the best parameters, various sub-datasets (datasets with smaller number of examples) of different parameters are prepared for an exploration. Since it is important to keep a balance between the accuracy and the computational load, the sizes of cells and the blocks are chosen in a limited range (small cell sizes and larger blocks are computationally expensive). A table is given for the explored sub-datasets with different features extracted from RGB, GRAY and HSV color spaces.


| Dataset | color space   | Hist eqz* | pixels per cell | cells per block | bins | size**   | accuracy |
|:-------:|:-------------:|:---------:|:---------------:|:---------------:|:----:|:--------:|---------:|
| 1       | GRAY          | yes       | 8x8             | 1x1             | 9    | 4000     | 0.951    |
| 1       | GRAY          | No        | 8x8             | 1x1             | 9    | 4000     | 0.927    |
| 1       | GRAY          | yes       | 8x8             | 2x2             | 9    | 4000     | 0.977    |
| 1       | GRAY          | No        | 8x8             | 2x2             | 9    | 4000     | 0.952    |
| 1       | GRAY          | yes       | 16x16           | 1x1             | 9    | 4000     | 0.968    |
| 1       | GRAY          | No        | 16x16           | 1x1             | 9    | 4000     | 0.956    |
| **1     | GRAY          | yes       | 16x16           | 2x2             | 9    | 4000     | 0.984**  |
| 1       | GRAY          | No        | 16x16           | 2x2             | 9    | 4000     | 0.975    |
| 1       | RGB           | yes       | 8x8             | 1x1             | 9    | 4000     | 0.961    |
| 1       | RGB           | No        | 8x8             | 1x1             | 9    | 4000     | 0.935    |
| 1       | RGB           | yes       | 8x8             | 2x2             | 9    | 4000     | 0.983    |
| 1       | RGB           | No        | 8x8             | 2x2             | 9    | 4000     | 0.963    |
| 1       | RGB           | yes       | 16x16           | 1x1             | 9    | 4000     | 0.971    |
| 1       | RGB           | No        | 16x16           | 1x1             | 9    | 4000     | 0.955    |
| **1     | RGB           | yes       | 16x16           | 2x2             | 9    | 4000     | 0.989**  |
| 1       | RGB           | No        | 16x16           | 2x2             | 9    | 4000     | 0.977    |
| 1       | HSV           | yes       | 8x8             | 1x1             | 9    | 4000     | 0.946    |
| 1       | HSV           | No        | 8x8             | 1x1             | 9    | 4000     | 0.961    |
| 1       | HSV           | yes       | 8x8             | 2x2             | 9    | 4000     | 0.966    |
| 1       | HSV           | No        | 8x8             | 2x2             | 9    | 4000     | 0.972    |
| 1       | HSV           | yes       | 16x16           | 1x1             | 9    | 4000     | 0.967    |
| 1       | HSV           | No        | 16x16           | 1x1             | 9    | 4000     | 0.966    |
| 1       | HSV           | yes       | 16x16           | 2x2             | 9    | 4000     | 0.976    |
| 1       | HSV           | No        | 16x16           | 2x2             | 9    | 4000     | 0.980    |



![false positives non vehicle set](./images/non_vehicle_false_positives.png)