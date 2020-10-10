![RSAN](1.png?raw=true "RSAN")
# [Residual Spatial Attention Network for Retinal Vessel Segmentation(ICONIP2020)](https://arxiv.org/abs/2009.08829)
This code is for the paper: Residual Spatial Attention Network for Retinal Vessel Segmentation. We report state-of-the-art performances on DRIVE and CHASE DB1 datasets.

Code written by Changlu Guo, Budapest University of Technology and Economics(BME).


We train and evaluate on Ubuntu 16.04, it will also work for Windows and OS.

## Spatial Attention (SA)
Spatial Attention (SA) was introduced as a part of the convolutional block attention module for classification and detection. SA employs the inter-spatial relationship between features to produce a spatial attention map, the code can be found in [attention_module.py](attention_module.py).
![SA](2.png?raw=true "SA")

## Residual Spatial Attention Block (RSAB)
In this paper, we proposed RSAB, because it integrates DropBlock to prevent overfitting and the advantages of spatial attention, we believe it has great potential in the field of medical image processing for building deep networks even for processing small data sets .
![RSAB](4.png?raw=true "RSAB")



## Quick start 

Train:
Run [train_drive.py](train_drive.py) or [train_chase.py](train_chase.py)

Test:
Run [eval_drive.py](eval_drive.py) or [eval_chase.py](eval_chase.py)

## Results

![Results](5.png?raw=true "Results")
Row 1 is for DRIVE dataset. Row 2 is for CHASE DB1 dataset. (a) Color fundus images, (b) segmentation results of Backbone, (c) segmentation results of Backbone+DropBlock, (d) segmentation results of RSAN, (e) corresponding ground truths.

## Environments
Keras 2.3.1  <br>
Tensorflow==1.14.0 <br>


## If you are inspired by our work, please cite this paper.

@misc{guo2020residual, <br>
    title={Residual Spatial Attention Network for Retinal Vessel Segmentation}, <br>
    author={Changlu Guo and Márton Szemenyei and Yugen Yi and Wei Zhou and Haodong Bian}, <br>
    year={2020}, <br>
    eprint={2009.08829}, <br>
    archivePrefix={arXiv}, <br>
    primaryClass={eess.IV} <br>
}

