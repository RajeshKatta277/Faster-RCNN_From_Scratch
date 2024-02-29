class FasterRCNN(nn.Module):
    def __init__(self):
        super(FasterRCNN, self).__init__()
        model=torchvision.models.resnet50(weights=torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V1)
        layers=list(model.children())[:8]
        self.backbone=nn.Sequential(*layers).eval()
        self.rpn = RegionProposalNetwork(in_channels=2048,num_anchors=18)
        self.head = nn.Sequential(
            nn.Linear(2048 * 5 * 5, 1024),
            nn.Tanh(),
            nn.Linear(1024,512),
            nn.Tanh(),
            
        )
        self.cls_layer = nn.Linear(512, 1) 
        self.anchors=None
        self.train=True
        
    def generate_anchor_boxes(self,feature_map_size=(7,7),anchor_scales=[32,36,56,64,72,128],anchor_ratios=[0.5,1,2],stride=32):
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
        
    def extract_regions(self,feature_map,deltas, labels, anchor_boxes):
        """
        Convert regression deltas to box coordinates for boxes with objectness score 1 -
        and applies roi pooling to those regions to convert them to fixed size.

        Args:
            deltas: (numpy array) Regression deltas of shape (7, 7, 18, 4).
            labels: (numpy array) Classification labels of shape (7, 7, 18).
            anchor_boxes: (numpy array) Anchor boxes of shape (7, 7, 18, 4).

        Returns:
            boxes: (numpy array) roi pooled box coordinates for boxes with objectness score 1 of shape (num_boxes_with_object, 4).
        """

        deltas_flat = deltas.reshape((-1, 4))
        labels_flat = labels.flatten()
        anchor_boxes_flat = anchor_boxes.reshape((-1, 4))

        object_indices = np.where(labels_flat == 1)[0]

        boxes = []

        for idx in object_indices:
            delta = deltas_flat[idx]
            anchor_box = anchor_boxes_flat[idx]
            ax1, ay1, ax2, ay2 = anchor_box
            dx, dy, dw, dh = delta

            width_anchor = max(np.abs(ax2 - ax1), 1e-10)
            height_anchor = max(np.abs(ay2 - ay1), 1e-10)

            gx1 = dx * width_anchor + ax1
            gy1 = dy * height_anchor + ay1
            gx2 = np.exp(dw.detach().cpu().numpy()) * width_anchor + ax1
            gy2 = np.exp(dh.detach().cpu().numpy())* height_anchor + ay1
            boxes.append([gx1, gy1, gx2, gy2])

        boxes = torch.tensor(boxes, dtype=float)
        batch_index = torch.zeros((boxes.shape[0], 1), dtype=torch.float32)
        boxes_with_batch = torch.cat((batch_index,boxes), dim=1)
        feature_map=feature_map.to(torch.float32)
        boxes_with_batch=boxes_with_batch.to(torch.float32)

        pooled_regions = torchvision.ops.roi_pool(feature_map,boxes_with_batch, output_size=(5,5))
        return pooled_regions,boxes 
    
    def train(self):
        self.train=True
        
    def eval(self):
        self.train=False
    
    def forward(self, image):
        if self.anchors is None:
            self.anchors=self.generate_anchor_boxes()
        feature_map = self.backbone(image)
        rpn_scores, rpn_deltas = self.rpn(feature_map) 
        proposals,boxes = self.extract_regions(feature_map.cpu(), rpn_deltas.cpu(),(torch.sigmoid(rpn_scores)>0.5).cpu(),self.anchors)
    
        flattened_pooled_regions = proposals.view(proposals.size(0), -1).to(device)
        f = self.head(flattened_pooled_regions) 
        class_scores = self.cls_layer(f)
        
        if self.train:
            return rpn_scores,rpn_deltas,class_scores 
        else:
            return rpn_scores,boxes,class_scores
