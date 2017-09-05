# Vehicle Detection Project

The aim of this project is to detect the vehicles in images using HOG features.

For the code implementations please refer to [this notebook](./project-code.ipynb).

## HOG Features

When the class of objects varies in color, structural features like gradients or edges might give a more robost representation. This way, different signatures for different shapes can be obtained. However, the signature for a shape should have enough flexibility to accommodate small variations in orientation, size, etc. The problem with using the gradient values directly is that the signature becomes too sensitive. To allow some variability the histogram of oriented gradients (HOG feature), which is a more flexible approach, will be implemented.

For to get the HOG features, a helper function named `get_hog` is implemented using the [scikit-image](http://scikit-image.org/) package to compute the histogram of gradient directions (orientations) of an image. The gradient samples are distributed into a nine orientation bins as default. Nine bins were enough for the accuracy in small scales. Here in the implementation of the histogram, it is not used just the count of the samples in each direction. Instead, the gradient magnitudes of each sample are summed up. This way, the stronger gradients contribute more weight to their orientation bin and thus, the effect of noise is reduced. By dividing the images into 8x8 cells and by getting the histogram for the directions and the gradient magnitudes of the pixels in these cells, the representations of the structures emerge. The main advantage is that HOG gives the ability to accept small variations in the shape, while keeping the signature distinct enough. The sensitivity for the features can be tweaked by varying parameters such as the number of orientation bins, grid of cells, cell sizes, the overlap between cells, block normalization, etc ([Navneet Dalal and Bill Triggs, 2005](http://ieeexplore.ieee.org/document/1467360/)).

A car is cropped out from the test image with the name [test1](./test_images/test1.jpg) and the HOG features for different cell sizes are visualized in the images given below.

![hog images for different cell sizes](./images/car-hog-npixels.png)

As can be seen from the images, a 8x8 pixels of cell size seems convenient to get an acceptable representation of the car. Thus, a 8x8 cell size will be taken as the default value.

## The Dataset

The dataset provided for the project is a combination of datasets from various sources. These sources are [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html) (GTI), [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/) and extra images extracted from the [project video](./project_video.mp4). The dataset has two classes of images with the labels *vehicle* and *non-vehicle*. The number of images with the *vehicle* label and their sources can be seen below. The sets of *GTI Far*, *GTI left*, *GTI Right* and *GTI MidClose* are from the same source GTI with different viewpoints. Together with the *KITTI* examples they constitute the set of *vehicle* images.

![vehicle image numbers](./images/vehicle-sets.png)

In the images below, some of the examples are given from *vehicle* sources.

![gti far set](./images/gti_far.png)
![gti right set](./images/gti_right.png)
![gti left set](./images/gti_left.png)
![gti midclose set](./images/gti_midclose.png)
![gti kitti set](./images/gti_kitti.png)

The *non-vehicle* images are taken from GTI with extra captured images from the project video. The number of examples for each source is given below.

![non-vehicle image numbers](./images/non-vehicle-sets.png)

In the images below, some of the examples are given from *non-vehicle* sources.

![gti non vehicle set](./images/non_vehicle_gti.png)
![extras non vehicle set](./images/non_vehicle_extras.png)

To check the balance between *vehicle* and *non-vehicle* classes the figure of total number of examples for each class is given below.

![vehicle non-vehicle image numbers](./n-classes.png)

As can be seen from the figure, the classes of the dataset seem to be roughly balanced, thus no augmentation will be applied.

