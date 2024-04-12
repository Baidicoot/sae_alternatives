import torch

def normalize_dict(m):
    norms = torch.norm(m, p=2, dim=-1).unsqueeze(-1)
    return m / norms

def huber_loss(x, x_hat, delta):
    raw_loss = (x - x_hat)
    loss = raw_loss.pow(2) * 0.5
    loss[loss > delta] = delta * (raw_loss.abs() - 0.5 * delta)
    return loss.mean()

def approx_overcomplete_gram_schmidt(feats, alpha=1, n_ortho=None, nonneg=False):
    if n_ortho is None:
        n_ortho = feats.shape[1]
    rand_idxs = torch.randperm(feats.shape[0])[:n_ortho]
    feat_subset = feats[rand_idxs]
    orth_matrix = feat_subset @ feat_subset.T
    # orth_matrix -= torch.eye(orth_matrix.shape[0], device=orth_matrix.device)
    orth_matrix = torch.tril(orth_matrix, diagonal=-1)
    if nonneg:
        orth_matrix = orth_matrix.clamp(min=0)
    new_feat_subset = feat_subset - alpha * (orth_matrix @ feat_subset)

    output = feats.clone()
    output[rand_idxs] = new_feat_subset

    return normalize_dict(output)

def reinitialize_similar_features(feats, thresh=0.9, n_ortho=None, nonneg=False):
    if n_ortho is None:
        n_ortho = feats.shape[0]
    rand_idxs = torch.randperm(feats.shape[0])[:n_ortho]
    feat_subset = feats[rand_idxs]
    
    similarities = feat_subset @ feat_subset.T
    similarities = torch.tril(similarities, diagonal=-1)

    if nonneg:
        similarities = similarities.clamp(min=0)
    else:
        similarities = similarities.abs()

    reinit_mask = similarities > thresh

    n_reinit = 0

    for i in range(n_ortho):
        if reinit_mask[i].sum() > 0:
            feat_subset[i] = torch.randn_like(feat_subset[i])
            n_reinit += 1

    output = feats.clone()
    output[rand_idxs] = feat_subset

    return output

class KurtosisICA(torch.nn.Module):
    def __init__(self, d_in, d_hidden, alpha, nonneg=False):
        super().__init__()
        self.embed = torch.nn.Parameter(torch.empty(d_hidden, d_in))
        torch.nn.init.kaiming_normal_(self.embed)

        self.alpha = alpha
        self.nonneg = nonneg
    
    def forward(self, x):
        h = torch.einsum('ij,bj->bi', self.embed, x)
        h_mean = h.mean(dim=0)
        h_centered = h - h_mean.unsqueeze(0)
        if self.nonneg:
            h_centered = torch.nn.functional.relu(h_centered)

        kurtosis = h_centered.pow(4).mean(dim=0) / h_centered.pow(2).mean(dim=0).pow(2)
        excess_kurtosis = kurtosis - 3
        gaussianity_loss = torch.exp(-excess_kurtosis.clamp(min=0).pow(2) * self.alpha).mean()

        # x_hat = torch.einsum('ij,bi->bj', self.embed, h)
        x_hat = torch.einsum('ij,bi->bj', self.embed, h)
        h_hat = torch.einsum('ij,bj->bi', self.embed, x_hat)
        ortho_loss = (h - h_hat).pow(2).mean()

        return ortho_loss + gaussianity_loss, h, x_hat

    @property
    def dictionary(self):
        return normalize_dict(self.embed).detach()

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
        sparsity = h.abs().mean()

        return mse + sparsity * self.alpha, h, x_hat

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
        sparsity = h.abs().mean()

        return mse + sparsity * self.alpha, h, x_hat

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
        h = torch.einsum('ij,bj->bi', self.embed, x - self.embed_bias.unsqueeze(0))
        x_hat = torch.einsum('ij,bi->bj', self.embed, h) + self.embed_bias.unsqueeze(0)

        mse = (x - x_hat).pow(2).mean()
        sparsity = h.abs().mean()

        return mse + sparsity * self.alpha, h, x_hat

    @property
    def dictionary(self):
        return normalize_dict(self.embed).detach()

class HuberRICA(torch.nn.Module):
    def __init__(self, d_in, d_hidden, alpha):
        super().__init__()
        initial_matrix = torch.randn(d_hidden, d_in)
        initial_matrix = normalize_dict(initial_matrix)
        self.embed = torch.nn.Parameter(initial_matrix.clone())
        self.embed_bias = torch.nn.Parameter(torch.zeros(d_in))

        self.alpha = alpha
    
    def forward(self, x):
        h = torch.einsum('ij,bj->bi', self.embed, x - self.embed_bias.unsqueeze(0))
        x_hat = torch.einsum('ij,bi->bj', self.embed, h) + self.embed_bias.unsqueeze(0)

        huber = huber_loss(x, x_hat, 1)
        mse = (x - x_hat).pow(2).mean()
        sparsity = h.abs().mean()

        return huber + sparsity * self.alpha, h, x_hat

    @property
    def dictionary(self):
        return normalize_dict(self.embed).detach()

class ConstrainedRICA(torch.nn.Module):
    def __init__(self, d_in, d_hidden, alpha):
        super().__init__()
        initial_matrix = torch.randn(d_hidden, d_in)
        initial_matrix = normalize_dict(initial_matrix)
        self.embed = torch.nn.Parameter(initial_matrix.clone())
        self.embed_bias = torch.nn.Parameter(torch.zeros(d_in))

        self.alpha = alpha
    
    def forward(self, x):
        with torch.no_grad():
            self.embed.data = normalize_dict(self.embed.data)

        h = torch.einsum('ij,bj->bi', self.embed, x - self.embed_bias.unsqueeze(0))
        x_hat = torch.einsum('ij,bi->bj', self.embed, h) + self.embed_bias.unsqueeze(0)

        mse = (x - x_hat).pow(2).mean()
        sparsity = h.abs().mean()

        return mse + sparsity * self.alpha, h, x_hat

    @property
    def dictionary(self):
        return normalize_dict(self.embed).detach()

class NonNegativeRICA(torch.nn.Module):
    def __init__(self, d_in, d_hidden, alpha):
        super().__init__()
        initial_matrix = torch.randn(d_hidden, d_in)
        initial_matrix = normalize_dict(initial_matrix)
        self.embed = torch.nn.Parameter(initial_matrix.clone())
        self.embed_bias = torch.nn.Parameter(torch.zeros(d_in))

        self.alpha = alpha
 
    def forward(self, x):
        h = torch.einsum('ij,bj->bi', self.embed, x - self.embed_bias.unsqueeze(0))
        h = torch.nn.functional.softplus(h)
        x_hat = torch.einsum('ij,bi->bj', self.embed, h) + self.embed_bias.unsqueeze(0)

        mse = (x - x_hat).pow(2).mean()
        sparsity = h.abs().mean()

        return mse + sparsity * self.alpha, h, x_hat

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
        sparsity = h.abs().mean()

        return mse + sparsity * self.alpha, h, x_hat

    @property
    def dictionary(self):
        return normalize_dict(self.unembed).detach()

def shrinkage(x, a, threshold=20):
    exponent = torch.pow(x, 2) / torch.clamp(torch.pow(a, 2), min=1e-6)
    exponent = torch.clamp(exponent, max=threshold)
    scale = 1 - torch.exp(-exponent)
    return x * scale

class NNMeanCenteredICA(torch.nn.Module):
    def __init__(self, d_in, d_hidden, alpha):
        super().__init__()
        initial_matrix = torch.randn(d_hidden, d_in)
        initial_matrix = normalize_dict(initial_matrix)
        self.embed = torch.nn.Parameter(initial_matrix.clone())

        self.alpha = alpha
    
    def forward(self, x):
        with torch.no_grad():
            self.embed.data = normalize_dict(self.embed.data)

        # mean center the batch
        x_mean = x.mean(dim=0)
        x_centered = x - x_mean

        # project the batch onto the dictionary
        h = torch.einsum('ij,bj->bi', self.embed, x_centered)
        h = torch.nn.functional.relu(h)

        # reconstruct the batch
        x_hat = torch.einsum('ij,bi->bj', self.embed, h) + x_mean

        mse = (x - x_hat).pow(2).mean()
        sparsity = h.abs().mean()

        return mse + sparsity * self.alpha, h, x_hat

    @property
    def dictionary(self):
        return normalize_dict(self.embed).detach()

class MeanCenteredICA(torch.nn.Module):
    def __init__(self, d_in, d_hidden, alpha):
        super().__init__()
        initial_matrix = torch.randn(d_hidden, d_in)
        initial_matrix = normalize_dict(initial_matrix)
        self.embed = torch.nn.Parameter(initial_matrix.clone())

        self.alpha = alpha
    
    def forward(self, x):
        with torch.no_grad():
            self.embed.data = normalize_dict(self.embed.data)

        # mean center the batch
        x_mean = x.mean(dim=0)
        x_centered = x - x_mean

        # project the batch onto the dictionary
        h = torch.einsum('ij,bj->bi', self.embed, x_centered)
        h = shrinkage(h, self.embed)

        # reconstruct the batch
        x_hat = torch.einsum('ij,bi->bj', self.embed, h)

        mse = (x_centered - x_hat).pow(2).mean()
        huber = huber_loss(x_centered, x_hat, 1)        

        # scale by mean activation (but detach to avoid backprop)
        sparsity = h.abs().mean()

        return huber + sparsity * self.alpha, h, x_hat

class ShrinkageRICA(torch.nn.Module):
    def __init__(self, d_in, d_hidden, alpha):
        super().__init__()
        initial_matrix = torch.randn(d_hidden, d_in)
        initial_matrix = normalize_dict(initial_matrix)
        self.embed = torch.nn.Parameter(initial_matrix.clone())
        self.embed_bias = torch.nn.Parameter(torch.zeros(d_in))
        self.embed_shrinkage = torch.nn.Parameter(torch.ones(d_hidden))

        self.alpha = alpha
    
    def forward(self, x):
        h = torch.einsum('ij,bj->bi', self.embed, x - self.embed_bias.unsqueeze(0))
        h = shrinkage(h, self.embed_shrinkage.unsqueeze(0))
        x_hat = torch.einsum('ij,bi->bj', self.embed, h) + self.embed_bias.unsqueeze(0)

        mse = (x - x_hat).pow(2).mean()
        sparsity = h.abs().mean()

        return mse + sparsity * self.alpha, h, x_hat
    
    @property
    def dictionary(self):
        return normalize_dict(self.embed).detach()