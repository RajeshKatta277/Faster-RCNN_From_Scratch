class TumorDataset(Dataset):
    def __init__(self, transform=None,train=True):
        self.transform = transform
        self.train=train
        if self.train:
            self.data=train_labels_df 
        else:
            self.data=test_labels_df
        

    def __len__(self):
        return len(self.data)
    
    def generate_anchor_boxes(self,feature_map_size, anchor_scales, anchor_ratios,stride):
        '''Generates anchor boxes given anchor scales, anchor ratios and stride(stride: 1 unit of lenfth or breadth in the feature space is equal to how many units of lenght or breadth in the image space)
           Inputs: tuple:feature map size
                   list:anchor_scales
                   list:anchor_ratios
                   int: stride
           Output: numpy array: anchor boxes (feature_map_size[0] x feature_map_size[1] x number of anchor boxes per location x 4) '''
        
        num_anchors = len(anchor_scales) * len(anchor_ratios)
        anchors = np.zeros((feature_map_size[0], feature_map_size[1], num_anchors, 4))  # (H, W, num_anchors, 4)

        for i in range(feature_map_size[0]):
            for j in range(feature_map_size[1]):
                for k, scale in enumerate(anchor_scales):
                    for l, ratio in enumerate(anchor_ratios):
                        # Calculate anchor box coordinates
                        w = scale * np.sqrt(ratio) # ratio= width/height
                        h = scale / np.sqrt(ratio) #1/ratio =height/width
                        x1 = j * stride - w / 2
                        y1 = i * stride - h / 2
                        x2 = j * stride + w / 2
                        y2 = i * stride + h / 2
                        
                        anchors[i, j, k * len(anchor_ratios) + l] = convertor.image_to_feature(x1, y1, x2, y2)

        return anchors

    def assign_anchor_labels(self,anchors, gt_boxes, pos_iou_thresh=0.7, neg_iou_thresh=0.3, ignore_thresh=0.4):
        '''Labels the anchor boxes as positive, nagative and ignore boxes based on iou with ground truth box.'''
        
        num_anchors = np.prod(anchors.shape[:3])
        anchors_reshaped = anchors.reshape(num_anchors, 4)
        gt_boxes=torch.tensor(gt_boxes)
        num_gt_boxes = 1
        gt_boxes_reshaped = gt_boxes.reshape(num_gt_boxes, 4)
        iou = self.box_iou(torch.tensor(anchors_reshaped), gt_boxes_reshaped).numpy()


        #iou = box_iou(torch.tensor(anchors.reshape(-1, 4)), torch.tensor(gt_boxes)).numpy() # reshapes to size (h*w*number of anchor boxes , 4)
        iou = iou.reshape(anchors.shape[:3]+ (-1,) ) 
        labels = np.zeros(anchors.shape[:3], dtype=int)

    
        max_iou = np.amax(iou, axis=3)
        argmax_iou = np.argmax(iou, axis=3)
        labels[max_iou < neg_iou_thresh] = 0  # Negative label
        labels[max_iou >= pos_iou_thresh] = 1  # Positive label
        labels[np.logical_and(max_iou >=neg_iou_thresh, max_iou < pos_iou_thresh)] = -1  # Ignore label

        # Set positive label for anchors with the highest IoU for each ground-truth box
        for gt_idx in range(gt_boxes.shape[0]):
            if iou.shape[3] == 1:
                anchor_idx = np.unravel_index(np.argmax(iou), iou.shape[:3])
            else:
                anchor_idx = np.unravel_index(np.argmax(iou[:, :, :, gt_idx]), iou.shape[:3])

            labels[anchor_idx] = 1
    
        return labels 
    
    def generate_regression_targets(self,anchors, gt_boxes, labels):
        num_anchors = len(anchors)
        regression_targets = np.zeros((num_anchors, 4), dtype= float)

        gt_boxes=torch.tensor(gt_boxes).reshape(1,4)

        for i in range(num_anchors):
            if labels[i] == 1:  # Positive anchor box
                anchor = anchors[i]
                gt_box = gt_boxes[np.argmax(self.box_iou(torch.tensor(anchor.reshape(-1, 4)), gt_boxes))]  # Corresponding ground-truth box
                regression_targets[i] = self.compute_regression_target(anchor, gt_box)  # Calculate regression targets

        return regression_targets
    
    def compute_regression_target(self,anchor, gt_box):
        """
        Compute bounding box regression targets (deltas) between an anchor box and a ground-truth bounding box.
        """
        ax1, ay1, ax2, ay2 = anchor
        gx1, gy1, gx2, gy2 = gt_box
        
        eps = 1e-10
        width_anchor = max(np.abs(ax2 - ax1), eps)
        height_anchor = max(np.abs(ay2 - ay1), eps)
        width_gt = max(np.abs(gx2 - gx1),eps)
        height_gt = max(np.abs(gy2 - gy1),eps)
       
        
        dx = (gx1 - ax1) / width_anchor
        dy = (gy1 - ay1) / height_anchor
        dw = np.log(width_gt / width_anchor)
        dh = np.log(height_gt / height_anchor) 
    
        return np.array([dx, dy, dw, dh], dtype=float)
    
    def box_iou(self,anchors, gt_box):
        """
        Calculate IoU between anchor boxes and a single ground truth box.

        Inputs:
            anchors (torch.Tensor): Anchor boxes of shape (7, 7, num_anchors, 4) containing (x1, y1, x2, y2) coordinates.(7x7 is the feature map size).
            gt_box (torch.Tensor): Ground truth box of shape (1, 4) containing (x1, y1, x2, y2) coordinates.

        Outputs:
            Tensor: IoU between each anchor box and the ground truth box, shaped (7, 7, num_anchors, 1).
        """

        anchor_x1 = anchors[..., 0]
        anchor_y1 = anchors[..., 1]
        anchor_x2 = anchors[..., 2]
        anchor_y2 = anchors[..., 3]

        gt_x1, gt_y1, gt_x2, gt_y2 = gt_box[0]


        inter_x1 = torch.max(anchor_x1, gt_x1)
        inter_y1 = torch.max(anchor_y1, gt_y1)
        inter_x2 = torch.min(anchor_x2, gt_x2)
        inter_y2 = torch.min(anchor_y2, gt_y2)

        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0) #intersection area


        anchor_area = (anchor_x2 - anchor_x1) * (anchor_y2 - anchor_y1)
        gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)

    
        union_area = anchor_area + gt_area - inter_area


        iou = inter_area / (union_area + 1e-6)  
        return iou.view(-1, 1)
    
    def generate_rpn_training_data(self,image_shape, gt_boxes, anchor_scales, anchor_ratios, stride=32, pos_iou_thresh=0.5, neg_iou_thresh=0.1, ignore_thresh=0.4):
        """
        Generate training data for the Region Proposal Network (RPN) in our Faster R-CNN.

        Inputs:
            image_shape (tuple): Shape of the input image in the format (height, width).
            gt_boxes (numpy array): Ground truth bounding boxes of shape (num_gt_boxes, 4) in format (x1, y1, x2, y2).
            anchor_scales (list): List of scales for anchor boxes.
            anchor_ratios (list): List of aspect ratios for anchor boxes.
            stride (int): Stride of the feature map.
            pos_iou_thresh (float): IoU threshold for positive anchors.
            neg_iou_thresh (float): IoU threshold for negative anchors.
            ignore_thresh (float): IoU threshold for anchors to be ignored.

        Outputs:
            numpy array: Anchor boxes of shape (H, W, num_anchors, 4).
            numpy array: Anchor labels of shape (H, W, num_anchors) where 0 represents background, 1 represents foreground, and -1 represents ignore.
            numpy.array: Regression targets of shape (H, W, num_anchors, 4).
        """

        feature_map_size = (image_shape[0] // stride, image_shape[1] // stride)
        anchors = self.generate_anchor_boxes(feature_map_size, anchor_scales, anchor_ratios,stride=stride)
        gt_boxes=convertor.image_to_feature(*gt_boxes)

        labels = self.assign_anchor_labels(anchors, gt_boxes, pos_iou_thresh, neg_iou_thresh, ignore_thresh)


        regression_targets = np.zeros((feature_map_size[0], feature_map_size[1], len(anchor_scales) * len(anchor_ratios), 4), dtype=float)
        for i in range(feature_map_size[0]):
            for j in range(feature_map_size[1]):
                regression_targets[i, j] = self.generate_regression_targets(anchors[i, j], gt_boxes, labels[i, j])

        return anchors, labels, regression_targets 


    def __getitem__(self, idx):
      
        if self.train:
            image = Image.open(os.path.join(path2data, "images/train", 
                                    train_labels_df["ID"][idx] + ".jpg")).convert("RGB")

            x_size, y_size = image.size
            x = train_labels_df["x_cent"][idx] * x_size
            y = train_labels_df["y_cent"][idx] * y_size
            w = train_labels_df["width"][idx] * x_size
            h = train_labels_df["height"][idx] * y_size
        else:
            image = Image.open(os.path.join(path2data, "images/test", 
                                    test_labels_df["ID"][idx] + ".jpg")).convert("RGB")

            x_size, y_size = image.size
            x = test_labels_df["x_cent"][idx] * x_size
            y = test_labels_df["y_cent"][idx] * y_size
            w = test_labels_df["width"][idx] * x_size
            h = test_labels_df["height"][idx] * y_size
        
            
            

        
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w/ 2
        y2 = y + h/ 2

        if self.transform:
            image = self.transform(image)
            _,resized_x,resized_y=image.shape
            scale_x = resized_x / x_size
            scale_y = resized_y / y_size
            x1=scale_x*x1
            y1=scale_y*y1
            x2=scale_x*x2
            y2=scale_y*y2

        bbox = [x1, y1, x2, y2]
        anc,lab,reg=self.generate_rpn_training_data(image_shape=(224,224),gt_boxes=bbox,anchor_scales=[32,36,56,64,72,128],anchor_ratios=[0.5,1,2],stride=32)
        sample={'image':image,'anc':anc,'labels':lab, 'reg':reg,'class':self.data['class'][idx]}
            
        return sample
