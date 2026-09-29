import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """

    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    
    
    sizes = np.ones((k,)) * (n_samples // k)
    # distribute extra slots among the first folds 
    for i in range(k - (n_samples//k)*k):
        sizes[sizes % i] += 1

    index_mask = np.array([[f_i] * int(size) for f_i,size in enumerate(sizes)]).flatten()

    folds = []

    for k_i in range(k):
        train = indices[index_mask != k_i].tolist()
        test = indices[index_mask == k_i].tolist()

        folds.append((train, test))

    return folds
