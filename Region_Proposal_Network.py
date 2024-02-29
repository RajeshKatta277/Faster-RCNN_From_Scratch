class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=512, num_anchors=9):
        super(RegionProposalNetwork, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)
    
        
    def forward(self, features):
        features = F.relu(self.conv(features))
        logits = self.cls_logits(features)  # Class logits
        bbox_deltas = self.bbox_pred(features)  # Bounding box deltas
        
        return logits, bbox_deltas

