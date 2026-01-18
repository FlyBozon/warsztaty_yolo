
red_ball - v1 2026-01-18 1:57pm
==============================

This dataset was exported via roboflow.com on January 18, 2026 at 1:01 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 33 images.
Ball are annotated in COCO format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 512x512 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise
* Randomly crop between 0 and 80 percent of the image
* Random shear of between -35° to +35° horizontally and -31° to +31° vertically
* Random Gaussian blur of between 0 and 2.5 pixels
* Salt and pepper noise was applied to 2.44 percent of pixels


