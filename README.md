# Edge Detection and Object Outline Extraction with U-Net

This project shows my implementation of an edge detection system that combines traditional methods (Canny and Sobel) with a deep learning approach using a U-Net model. The U-Net model pre-trained on the Carvana dataset is fine-tuned on the BSDS500 dataset to produce filled binary segmentation masks. These masks are then post-processed to extract crisp outlines of the object of interest.

## Overview 

The main goals of this project are to:
 - Load and preprocess the BSDS500 dataset.
 - Generate baseline edge maps using traditional methods (Canny and Sobel).
 - Fine-tune a pre-trained U-Net model to output a filled binary segmentation mask of the target object from the BSDS500 dataset.
 - Extract a crisp outline from the predicted segmentation mask using post-processing techniques.
 - Compare the results between Canny, Sobel, and the U-Net model.
 
## Approach and Model Selection

### Traditional Methods
 - **Canny Edge Detector**: Provides baseline edge maps by detecting gradients.
 - **Sobel Filter**: Combines gradients in the x and y directions to yield edge maps.
 
### Deep Learning Approach
 - **U-Net Model**: 
   - I used the pre-trained 'unet_carvana' model from [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet), which was originally used on a Carvana dataset.
   - Given the differences between the target objects (cars vs. natural images) and the task difference (semantic segmentation vs. edge detection), I fine-tuned the model on the BSDS500 dataset using binary segmentation masks.
   
 - **Binary Masks vs. Edge Maps**:
   - Instead of training the U-Net to predict edge maps (tends to be ambiguous), I converted the ground truth edge annotations into filled binary masks. This lets the network to learn to segment the entire object.
   - After prediction, Canny is applied to the segmentation mask to produce a crisp outline.
   
### Fine-Tuning Strategy
 - **Training Setup**:
   - Batch size is 1
   - Learning rate is 1e-5
   - Number of epochs is 10

## Project Structure

 - **Cell 1**: Imports and setup
   This cell contains all the necessary libraries, disables SSL verification for torch.hub, and sets up the Python environment.
   
 - **Cell 2**: Dataset and Pre-processing
   This cell defines the 'BSDSDataset' class, which loads images and ground truth annotations from BSDS500, converts images to RGB, and fills edge maps to create binary segmentation masks.
   
 - **Cell 3**: Traditional Edge Detection
   This cell implements functions for Canny and Sobel and visualizes their outputs.
   
 - **Cell 4**: Loading the Pre-Trained U-Net Model
   This cell loads the pre-trained U-Net model from torch.hub (the Carvana version), applying a monkey-patch to force the CPU.
   
 - **Cell 5**: Fine-Tuning the U-Net
   This cell fine-tunes the pre-trained U-Net on BSDS500 using the filled binary segmentation masks.
   
- **Cell 6**: Visualization
   This cell displays side-by-side comparisons of the original image, Canny, Sobel, and the U-Net output.
   
## Results and Analysis

 - **Traditional Methods**:
   - Canny and Sobel generate clear, fragmented edge maps.
   
 - **U-Net Approach**:
   - The fine-tuned U-Net outputs a filled segmentation mask that isolates the target object.
   - During my development of this notebook, the U-Net initially produced a black screen. Then it produced a white segmentation of the whole image. In order to reach the output that I have currrently, I had to turn the edge map into a binary segmentation. 

 - **Comparison**:
   - The final visualization demonstrates the differences between Canny, Sobel, and U-Net approach.
   - While the U-Net output may require further fine-tuning for perfection, it shows promise in accurately segmenting and outlining the object of interest.
   
## Video Demo

A video demonstration is available that walks through the code and describes the project by showing:
 - The pre-processing of the BSDS500 dataset.
 - Baseline edge detection outputs (Canny and Sobel).
 - The fine-tuning process on the U-Net.
 - A comparison of final edge detection results.
 
*Please use this link [video](https://youtu.be/X5sLSqk7hjw) for a detailed walkthrough*

## How to Run

1. **Dataset**:
  Ensure that the BSDS500Dataset is downloaded and in your project directory.

2. **Dependencies**:
  Install the required Python libraries:
  ```bash
  pip install torch torchvision opencv-python numpy scipy matplotlib
  ```
  
3. **Run the Notebook**:
  Open the Jupyter Notebook file `edge_detection.ipynb` and execute the cells in order. Ajust any paths in the code if needed.
  
# Acknowledgments

### Pre-trained Model:
Thanks to the developers of the [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) for providing a pre-trained U-Net model.

### Dataset:
Thanks to the creators of the [BSDS500dataset](https://figshare.com/articles/dataset/BSR_bsds500_data_set/13139684?utm_source=chatgpt.com&file=25236740) for providing a dataset for edge detection. 
**Credits**:
陈, 军宇 (2020). BSR_bsds500  data set. figshare. Dataset. https://doi.org/10.6084/m9.figshare.13139684.v1

### Traditional Edge Detectors:
Canny and Sobel are the edge detectors that are used.
