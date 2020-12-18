# SuperGlue with Physarum Dynamics

## Introduction

 This repository contains the implementation of SuperGlue with its original Sinkhorn Algorithm replaced by Physarum Dynamics solver. Physarum Dynamics is a very efficient  differentiable solver for general linear programming problems. Physarum Dynamics can be used in a plug and play manner within deep neural networks as a layer, which converges quickly without the need for a feasible initial point. 

For more details, please see:
* Physarum Dynamics full paper: [Physarum Powered Differentiable Linear Programming Layers and Applications](https://arxiv.org/abs/2004.14539).
* SuperGlue full paper: [SuperGlue: Learning Feature Matching with Graph Neural Networks](https://arxiv.org/abs/1911.11763).


 The SuperGlue network is a Graph Neural Network combined with an Optimal Matching layer that is trained to perform matching on two sets of sparse image features. SuperGlue operates as a "middle-end," performing context aggregation, matching, and filtering in a single end-to-end architecture. Correspondences across images have some constraints:
 * A keypoint can have at most a single correspondence in the another image.
 * Some keypoints will be unmatched due to occlusion and failure of the detector.

<p align="center">
  <img src="assets/superglue1.png" width="400"/>
</p>


 SuperGlue aims to find all correspondences between reprojections of the same points and identifying keypoints that have no matches. There are two main components in SuperGlue architecture: Attentional Graph Neural Network and Optimal Matching Layer.

<p align="center">
  <img src="assets/superglue2.png"/>
</p>

## Dependencies
* Python 3
* PyTorch >= 1.1
* OpenCV >= 3.4 (4.1.2.30 recommended for best GUI keyboard interaction, see this [note](#additional-notes))
* Matplotlib >= 3.1
* NumPy >= 1.18

Simply run the following command: `pip3 install numpy opencv-python torch matplotlib`

Or create a conda environment by `conda install --name myenv --file superglue.txt`

## Contents
There are two main top-level scripts in this repo:

1. `train.py` : trains the superglue model.
2. `load_data.py`: reads images from files and creates pairs. It generates keypoints, descriptors and ground truth matches which will be used in training.

### Download Data
Download the COCO2014 dataset files for training
```
wget http://images.cocodataset.org/zips/train2014.zip
```
Download the validation set
```
wget http://images.cocodataset.org/zips/val2014.zip
```
Download the test set
```
wget http://images.cocodataset.org/zips/test2014.zip
```

### Training Directions

To train the SuperGlue with default parameters, run the following command:

```sh
python train.py
```

### Additional useful command line parameters
* Use `--epoch` to set the number of epochs (default: `20`).
* Use `--train_path` to set the path to the directory of training images.
* Use `--eval_output_dir` to set the path to the directory in which the visualizations is written (default: `dump_match_pairs/`).
* Use `--show_keypoints` to visualize the detected keypoints (default: `False`).
* Use `--viz_extension` to set the visualization file extension (default: `png`). Use pdf for highest-quality.

### Visualization Demo
Due to space constraints, the experimentations bellow allow a small number of keypoints. This constraint will possibly be lifted if using sparse-sparse matrix multiplication.
The matches are colored by their predicted confidence in a jet colormap (Red: more confident, Blue: less confident). 

You should see images like this inside of `dump_match_pairs/`

<img src="assets/49_matches.png" width="800">
<img src="assets/219_matches2.png" width="800">
<img src="assets/149_matches.png" width="800">
<img src="assets/439_matches2.png" width="800">
<img src="assets/699_matches.png" width="800">

