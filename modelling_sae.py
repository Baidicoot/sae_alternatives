import torch

def normalize_dict(m):
    norms = torch.norm(m, p=2, dim=-1).unsqueeze(-1)
    return m / norms

class BasicSAE(torch.nn.Module):
    def __init__(self, d_in, d_hidden, alpha):
        super().__init__()
        initial_matrix = torch.randn(d_hidden, d_in)
        initial_matrix = normalize_dict(initial_matrix)
        self.embed = torch.nn.Parameter(initial_matrix.clone())
        self.unembed = torch.nn.Parameter(initial_matrix.clone())
        self.embed_bias = torch.nn.Parameter(torch.zeros(d_hidden))
        self.unembed_bias = torch.nn.Parameter(torch.zeros(d_in))

        self.alpha = alpha
    
    def forward(self, x):
        h = torch.einsum('ij,bj->bi', self.embed, x) + self.embed_bias.unsqueeze(0)
        h = torch.nn.functional.relu(h)
        x_hat = torch.einsum('ij,bi->bj', normalize_dict(self.unembed), h) + self.unembed_bias.unsqueeze(0)

        mse = (x - x_hat).pow(2).mean()
        sparsity = torch.norm(h, 1, dim=-1).mean() * self.alpha

        return mse + sparsity, mse, h, x_hat

    @property
    def dictionary(self):
        return normalize_dict(self.unembed).detach()

class SoftplusSAE(torch.nn.Module):
    def __init__(self, d_in, d_hidden, alpha):
        super().__init__()
        initial_matrix = torch.randn(d_hidden, d_in)
        initial_matrix = normalize_dict(initial_matrix)
        self.embed = torch.nn.Parameter(initial_matrix.clone())
        self.unembed = torch.nn.Parameter(initial_matrix.clone())
        self.embed_bias = torch.nn.Parameter(torch.zeros(d_hidden))
        self.unembed_bias = torch.nn.Parameter(torch.zeros(d_in))

        self.alpha = alpha
    
    def forward(self, x):
        h = torch.einsum('ij,bj->bi', self.embed, x) + self.embed_bias.unsqueeze(0)
        h = torch.nn.functional.softplus(h)
        x_hat = torch.einsum('ij,bi->bj', normalize_dict(self.unembed), h) + self.unembed_bias.unsqueeze(0)

        mse = (x - x_hat).pow(2).mean()
        sparsity = torch.norm(h, 1, dim=-1).mean() * self.alpha

        return mse + sparsity, mse, h, x_hat

    @property
    def dictionary(self):
        return normalize_dict(self.unembed).detach()

class RICA(torch.nn.Module):
    def __init__(self, d_in, d_hidden, alpha):
        super().__init__()
        initial_matrix = torch.randn(d_hidden, d_in)
        initial_matrix = normalize_dict(initial_matrix)
        self.embed = torch.nn.Parameter(initial_matrix.clone())
        self.embed_bias = torch.nn.Parameter(torch.zeros(d_in))

        self.alpha = alpha
    
    def forward(self, x):
        h = torch.einsum('ij,bj->bi', self.embed, x + self.embed_bias.unsqueeze(0))
        x_hat = torch.einsum('ij,bi->bj', self.embed, h) - self.embed_bias.unsqueeze(0)

        mse = (x - x_hat).pow(2).mean()
        sparsity = torch.norm(h, 1, dim=-1).mean() * self.alpha

        return mse + sparsity, mse, h, x_hat

    @property
    def dictionary(self):
        return normalize_dict(self.embed).detach()

class UntiedRICA(torch.nn.Module):
    def __init__(self, d_in, d_hidden, alpha):
        super().__init__()
        initial_matrix = torch.randn(d_hidden, d_in)
        initial_matrix = normalize_dict(initial_matrix)
        self.embed = torch.nn.Parameter(initial_matrix.clone())
        self.unembed = torch.nn.Parameter(initial_matrix.clone())
        self.embed_bias = torch.nn.Parameter(torch.zeros(d_hidden))
        self.unembed_bias = torch.nn.Parameter(torch.zeros(d_in))

        self.alpha = alpha
    
    def forward(self, x):
        h = torch.einsum('ij,bj->bi', self.embed, x) + self.embed_bias.unsqueeze(0)
        x_hat = torch.einsum('ij,bi->bj', normalize_dict(self.unembed), h) + self.unembed_bias.unsqueeze(0)

        mse = (x - x_hat).pow(2).mean()
        sparsity = torch.norm(h, 1, dim=-1).mean() * self.alpha

        return mse + sparsity, mse, h, x_hat

    @property
    def dictionary(self):
        return normalize_dict(self.unembed).detach()

def shrinkage(x, a, threshold=20):
    exponent = torch.pow(x, 2) / torch.clamp(torch.pow(a, 2), min=1e-6)
    exponent = torch.clamp(exponent, max=threshold)
    scale = 1 - torch.exp(-exponent)
    return x * scale

class ShrinkageSAE(torch.nn.Module):
    def __init__(self, d_in, d_hidden, alpha):
        super().__init__()
        initial_matrix = torch.randn(d_hidden, d_in)
        initial_matrix = normalize_dict(initial_matrix)
        self.embed = torch.nn.Parameter(initial_matrix.clone())
        self.embed_bias = torch.nn.Parameter(torch.zeros(d_in))
        self.embed_shrinkage = torch.nn.Parameter(torch.ones(d_hidden))

        self.alpha = alpha
    
    def forward(self, x):
        h = torch.einsum('ij,bj->bi', self.embed, x + self.embed_bias.unsqueeze(0))
        h = shrinkage(h, self.embed_shrinkage.unsqueeze(0))
        x_hat = torch.einsum('ij,bi->bj', self.embed, h) - self.embed_bias.unsqueeze(0)

        mse = (x - x_hat).pow(2).mean()
        sparsity = torch.norm(h, 1, dim=-1).mean() * self.alpha

        return mse + sparsity, mse, h, x_hat
    
    @property
    def dictionary(self):
        return normalize_dict(self.embed).detach()