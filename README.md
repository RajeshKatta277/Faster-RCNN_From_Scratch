# Faster-RCNN_From_Scratch
This is a simplified implementation of Faster-RCNN from scratch and training it for brain tumor detection. In this project, we have developed a complete implementation of the Faster R-CNN (Region-based Convolutional Neural Network) architecture from scratch using PyTorch. This implementation includes all components of the Faster R-CNN framework, including the Region Proposal Network (RPN) and the classification and regression heads. 

## 1. Dataset and Data Loading:

Training data consists of MRI images with corresponding bounding boxes (center coordinates, normalized width, and height) and labels (0: negative, 1: positive).
- Dataset link: https://www.kaggle.com/datasets/davidbroberts/brain-tumor-object-detection-datasets
### TumorDataset class:
- Takes an index as input.
- Loads the corresponding image.
- Scales the image from normalized to original shape.
- Generates RPN training data (anchor boxes, labels, regression deltas). 
- Returns: image, anchor boxes, anchor box labels, regression deltas, class (positive or negative).
### Dataloaders:
- trainloader: Loads batches of training data (batch size = 1).
- testloader: Loads batches of test data (batch size = 1).

## 2. Model Architecture:

### Backbone (Feature Extraction):

- Used a pre-trained ResNet-50 model (selected first 8 layers) to extract features from the input image.
- Freezed the backbone layers during training (not updated). 

### generate_anchor_boxes function:
- Generates pre-defined anchor boxes of different sizes and aspect ratios at each location on the feature map.
- Used for RPN objectness score prediction and bounding box refinement.
### Region Proposal Network (RPN):

#### RegionProposalNetwork class:
- Takes the extracted feature map from the backbone as input.
- conv layer: Applies a 3x3 convolution to the feature map.
- cls_logits layer: Predicts objectness score (foreground or background) for each anchor box using  1x1 convolutions of n_anchor units(outputs n_anchors x 1 sized tensor).
- bbox_pred layer: Predicts adjustments (deltas) to the anchor boxes for better bounding box localization using 1x1 convolutions of n_anchor*4 units (outputs number of anchors x 4 sized tensor, where 4 represents the delta values for x1, y1, x2, and y2 coordinates).

###  Region of Interest (ROI) Pooling:

extract_regions function: Takes feature map, regression deltas, labels, and anchor boxes as input.
-   Convert regression deltas to box coordinates for boxes with objectness score 1 and projects these boxes on the feature map to get proposals.
-   Applies roi pooling to those regions to convert them to fixed size.
-   Outputs pooled regions and refined bounding boxes.

### Head (Classification and Regression):

- head: A sequence of fully connected layers to process the pooled features.
- cls_layer: Final layer predicting the class probabilities (tumor or no tumor) using a linear layer.
- Skipped the bounding box regression at this stage because, i tried but it doesn't make any difference. 

## Training and Loss Function:

Training mode: model.train() sets the model to training mode.
Evaluation mode: model.eval() sets the model to evaluation mode (e.g., for testing).

During training:
- Processes the input image through the backbone.
- Passes the feature map through RPN for objectness scores and bounding box deltas.
- Extracts regions and refines bounding boxes based on RPN outputs.
- Calculates classification scores for tumor/no tumor using the processed regions.
- Returns RPN scores, deltas, and class scores.

During evaluation:
- Similar to training, but returns RPN scores, refined bounding boxes instead of deltas, and class scores.
- Apples non-maximum-suppression (NMS) and then averages the selected boxes to get the mean box. 

### Loss Function (faster_rcnn_loss): 

Combines three losses:
- Objectness classification loss: Binary cross-entropy between predicted objectness scores and ground truth labels (foreground/background).
- Bounding box regression loss: Smooth L1 loss between predicted bounding box adjustments and ground truth bounding box adjustments (only for foreground samples).
- Classification loss: Binary cross-entropy between predicted class probabilities (tumor or no tumor) and ground truth class labels.
#### Imbalance Handling: 
- During training, a sampler(RandomSampler) class is used to balance foreground and background samples (3 foreground and 3 background samples in my case) for RPN loss calculation. This addresses the bias towards negative samples due to the presence of many background regions in the images.  

