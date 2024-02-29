class RandomSampler:
    def __init__(self):
        pass
       
    def sample(self, num_positive_samples, num_negative_samples, labels):
        """
        Model returns the regions given by the rpn and correpsonding classifications. 
        Most of the regions are negative except the regions where tumor is present. 
        This causes imbalance and to handle this imbalance we sample indices from the dataset randomly,a fixed number of positives and negatives to ensure class balance.

        Inputs:
            num_positive_samples (int): Number of positive samples to include in the sampled indices.
            num_negative_samples (int): Number of negative samples to include in the sampled indices.
            labels (numpy array): Array containing the labels for each sample.
                                     Assumes binary labels (0 or 1), where 1 indicates a positive sample and 0 indicates a negative sample.
            

        Outputs:
            numpy.ndarray: Array containing the sampled indices, ensuring class balance in the mini-batches.
        """
        labels = labels.reshape(-1, 1)
        positive_indices = np.where(labels == 1)[0]
        negative_indices = np.where(labels == 0)[0] 

        # Randomly sample positive and negative indices 
        sampled_positive_indices = np.random.choice(positive_indices, min(num_positive_samples, len(positive_indices)), replace=False)
        sampled_negative_indices = np.random.choice(negative_indices, min(num_negative_samples, len(negative_indices)), replace=False)

        # Pad remaining slots with negative samples
        remaining_samples = max(num_positive_samples + num_negative_samples - len(sampled_positive_indices) - len(sampled_negative_indices), 0)
        remaining_negative_samples = np.random.choice(negative_indices, remaining_samples, replace=False)
        
        # Concatenate sampled positive and negative indices
        sampled_indices = np.concatenate((sampled_positive_indices, sampled_negative_indices, remaining_negative_samples))

        return sampled_indices 
