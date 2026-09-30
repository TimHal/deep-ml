import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    
    # standardize the data first
    means, stds = np.mean(data, axis=0), np.std(data, axis=0)
    data = (data - means) / stds  # assuming here stds is never 0

    # after some annoying debugging... numpy has a different opinion on what
    # should be the observations and features in the matrix (rows and cols swapped)
    covars = np.cov(data, rowvar=False)

    # now we find the evs and ews and normalize the evs according to the problem
    evalues, evectors  = np.linalg.eigh(covars)
    evectors = evectors.T


    for (i, vec) in enumerate(evectors):
        # get the first element > 1e-10
        first_i = np.where(np.abs(vec) > 1e-10)[0][0]
        if evectors[i][first_i] < 0:
            evectors[i] *= -1 
    
    # Now to the PCA part. Select the k highest eigenvalues
    pca_i = np.argsort(np.abs(evalues))[::-1][:k]

    return np.round(evectors[pca_i], 4).reshape(-1,k)


