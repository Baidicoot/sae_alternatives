import torch

class WhiteningEstimator:
    def __init__(self, mean, device=None):
        self.mean = mean
        self.dim = mean.size(0)
        self.cov = torch.zeros(self.dim, self.dim, device=device)
        self.n = 0

    def update(self, X):
        """
        Update the covariance with a batch of new data
        """
        batch_size = X.size(0)
        self.cov *= self.n / (self.n + batch_size)
        self.cov += torch.matmul(X.t(), X) / (self.n + batch_size)
        self.n += batch_size
    
    def get_whitening_matrix(self):
        """
        Get the whitening matrix
        """
        eigvals, eigvecs = torch.linalg.eigh(self.cov)
        return torch.matmul(eigvecs, torch.diag(1.0 / eigvals.sqrt()))

    def get_unwhitening_matrix(self):
        """
        Get the unwhitening matrix
        """
        eigvals, eigvecs = torch.linalg.eigh(self.cov)
        return torch.matmul(torch.diag(eigvals.sqrt()), eigvecs.t())

class MeanEstimator:
    def __init__(self):
        self.mean = None
        self.n = 0

    def update(self, X):
        """
        Update the mean with a batch of new data
        """
        if self.mean is None:
            self.mean = X.mean(dim=0)
        else:
            self.mean *= self.n / (self.n + X.size(0))
            self.mean += X.mean(dim=0) / (self.n + X.size(0))
        self.n += X.size(0)

    def get_mean(self):
        """
        Get the mean
        """
        return self.mean