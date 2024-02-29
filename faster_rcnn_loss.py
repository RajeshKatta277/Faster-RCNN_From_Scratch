def faster_rcnn_loss(logits, bbox_deltas,class_scores, objectness_labels,regression_deltas,true_classes, anchor_indices):
    '''Loss function: region proposal network loss + fast rcnn loss
                    = objectness loss + box adjustments loss + regions classification loss'''
    pred_logits = logits.view(-1, 1)[anchor_indices]
    pred_bbox_deltas = bbox_deltas.view(-1, 4)[anchor_indices]


    true_labels = objectness_labels.view(-1,1)[anchor_indices].float()
    true_reg =    regression_deltas.view(-1,4)[anchor_indices]
    
    classification_loss = F.binary_cross_entropy_with_logits(pred_logits,true_labels)
    positive_indices = torch.where(true_labels.squeeze(1) > 0)[0].to(device)
    
    
    regression_loss = F.smooth_l1_loss(pred_bbox_deltas[positive_indices],true_reg[positive_indices])
    
    class_loss=F.binary_cross_entropy_with_logits(class_scores,true_classes)

    return 0.2*classification_loss + 0.4*regression_loss + 0.4*class_loss
