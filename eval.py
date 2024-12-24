import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import polynomial_kernel
import numpy as np
import torch.distributed as dist
def compute_mmd(feat_real, feat_gen, n_subsets=100, subset_size=1000, **kernel_args):
    m = min(feat_real.shape[0], feat_gen.shape[0])
    subset_size = min(subset_size, m)
    mmds = np.zeros(n_subsets)
    choice = np.random.choice

    # with range(n_subsets) as bar:
    for i in range(n_subsets):
        g = feat_real[choice(len(feat_real), subset_size, replace=False)]
        r = feat_gen[choice(len(feat_gen), subset_size, replace=False)]
        o = compute_polynomial_mmd(g, r, **kernel_args) 
        mmds[i] = o
        # bar.set_postfix({'mean': mmds[:i+1].mean()})
    return mmds


def compute_polynomial_mmd(feat_r, feat_gen, degree=3, gamma=None, coef0=1):
    # use  k(x, y) = (gamma <x, y> + coef0)^degree
    # default gamma is 1 / dim
    X = feat_r
    Y = feat_gen

    K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

    return _mmd2_and_variance(K_XX, K_XY, K_YY)


def _mmd2_and_variance(K_XX, K_XY, K_YY):
    # based on https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
    # but changed to not compute the full kernel matrix at once
    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)

    diag_X = np.diagonal(K_XX)
    diag_Y = np.diagonal(K_YY)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m-1))
    mmd2 -= 2 * K_XY_sum / (m * m)
    return mmd2

class Eval:
    def __init__(self):
        # initialize dinov2 model
        from transformers import AutoModel, AutoImageProcessor
        local_rank = torch.distributed.get_rank()
        if local_rank == 0:
            self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
            self.model = AutoModel.from_pretrained('facebook/dinov2-large').bfloat16()
        dist.barrier()
        if local_rank != 0:
            self.processor =  AutoImageProcessor.from_pretrained('facebook/dinov2-large')
            self.model =  AutoModel.from_pretrained('facebook/dinov2-large').bfloat16()
        dist.barrier()
        self.model.to(f"cuda:{local_rank}")
        self.load_precomputed_features()

    def load_precomputed_features(self, dataset_path="./inet/dinov2_inet_feats.pt"):
        self.precomputed_features = torch.load(dataset_path).float().cpu().numpy()

    @torch.no_grad()
    def eval(self, val_images):
        """
        val_images: Tensor of shape (N, 3, 256, 256) , from 0 to 255, in uint8
        """
        # preprocess the images
        inputs = self.processor(images=val_images, return_tensors="pt").to(self.model.device)
        # forward pass
        outputs = self.model(**inputs)
        # get the embeddings
        embeddings = outputs.pooler_output
        # normalize the embeddings
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        embeddings = embeddings.float().cpu().numpy()

        # compute the mmd
        mmd = compute_mmd(self.precomputed_features, embeddings)
        mmd = max(0, mmd.mean())
        return mmd
