import torch
import numpy as np
import math

import torch.nn.functional as F
from random import sample, shuffle
from scipy import stats
from itertools import combinations, tee
from collections import defaultdict
from random import choices
from sparsemax import Sparsemax

from torch.distributions import Uniform

#from seq2sign.measures.regularity.get_labels import get_encodings, get_token_encodings, get_pos

@torch.jit.script
def linspace(start, stop, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)
    
    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)
    
    # the output starts at 'start' and increments until 'stop' in each dimension
    bins = (start[None] + steps*(stop - start)[None]).T
    
    bin_widths = (bins[:, 1:] - bins[:, :-1])
    centers = bins[:, :-1] + (bin_widths/2)
        
    return bins, centers

def mh_euclidean_uniform(all_representations, n_bins, n_heads=8, online_bins=None):
    d_hidden = all_representations.shape[-1]
    #all_representations = F.normalize(all_representations)
    all_representations = all_representations.view(-1, n_heads, int(d_hidden/n_heads))
    
    if online_bins is None:
        uni = Uniform(
            all_representations.min().repeat(int(d_hidden/n_heads)),
            all_representations.max().repeat(int(d_hidden/n_heads))
        )
        
        bins = uni.sample([n_bins]).T
    
    else:
        bins= online_bins
    
    scores = -torch.cdist(
        all_representations, 
        bins.T,
        p=2,
    ).clamp(min=1e-9)
    
    return scores, bins

def mh_norm_dot_uniform(
    all_representations, n_bins, n_heads=8, 
    uniform_sphere=True, radians=True, online_bins=None
):
    d_hidden = all_representations.shape[-1]
    rep_min, rep_max = all_representations.min(), all_representations.max()
    
    all_representations = all_representations.view(-1, n_heads, int(d_hidden/n_heads))
    all_representations = F.normalize(all_representations, dim=-1)
    
    if online_bins is None:
        if uniform_sphere:
            bins = F.normalize(
                torch.randn(
                    (
                        int(d_hidden/n_heads), 
                        n_bins
                    )
                ).to(all_representations.device),
                dim=0
            )
        else:
            uni = Uniform(
                rep_min.repeat(int(d_hidden/n_heads)), 
                rep_max.repeat(int(d_hidden/n_heads))
            )
            
            bins = uni.sample([n_bins]).T
    else:
        bins = online_bins
    
    scores = (
        torch.matmul(
            all_representations,
            bins
        )
    )
    
    if radians:
        scores = math.pi-torch.arccos(scores)
        
    else:
        scores = scores
    
    return scores, bins

def mh_dot(
    all_representations, n_bins, n_heads=8, 
    uniform=False, radians=True, online_bins=None
):
    d_hidden = all_representations.shape[-1]
    rep_min, rep_max = all_representations.min(), all_representations.max()
    
    all_representations = all_representations.view(-1, n_heads, int(d_hidden/n_heads))
    
    if online_bins is None:
        if uniform:
            uni = Uniform(
                rep_min.repeat(int(d_hidden/n_heads)), 
                rep_max.repeat(int(d_hidden/n_heads))
            )
            
            bins = uni.sample([n_bins]).T
              
        else:
            bins = torch.randn(
                    (
                        int(d_hidden/n_heads), 
                        n_bins
                    )
                ).to(all_representations.device)
    else:
        bins = online_bins
    
    scores = (
        torch.matmul(
            all_representations,
            bins
        )
    )
    
    
    return scores, bins

def soft_bin(
    all_representations, n_bins, bins=None, centers=None, 
    temp=1.0, dist_fn='sphere_uniform', sub_mean=False, n_heads=4,
    smoothing_fn="softmax", online_bins=None
    
):
    
    if sub_mean:
        all_representations = all_representations-all_representations.mean(0)
        
    maxxes = all_representations.max(0).values
    minns = all_representations.min(0).values
    
    bins, centres = linspace(minns, maxxes, n_bins+1)

    if dist_fn == 'sphere_uniform':
        scores, used_bins = mh_norm_dot_uniform(
            all_representations, n_bins, n_heads, uniform_sphere=True,
            radians=True, online_bins=online_bins
        )
        
    elif dist_fn == 'cosine_uniform':
        scores, used_bins  = mh_norm_dot_uniform(
            all_representations, n_bins, n_heads, uniform_sphere=True,
            radians=False, online_bins=online_bins
        )
        
    elif dist_fn == 'norm_uniform':
        scores, used_bins  = mh_norm_dot_uniform(
            all_representations, n_bins, n_heads, uniform_sphere=False,
            online_bins=online_bins
        )
        
    elif dist_fn == 'euclidean_uniform':
        scores, used_bins  = mh_euclidean_uniform(
            all_representations, n_bins, n_heads, online_bins=online_bins
        )
        
    elif dist_fn == 'dot':
        scores, used_bins  = mh_dot(
            all_representations, n_bins, n_heads, online_bins=online_bins
        )

    scores = smoothing(scores, temp, smoothing_fn)
    
    return scores, bins, centres, used_bins 


def smoothing(scores, temp, smoothing_fn, coeff=16):
    
    if smoothing_fn == "sparsemax":
        sparsemax = Sparsemax(dim=-1)
        scores = sparsemax(scores/temp)
    elif smoothing_fn == "softmax":
        scores = F.softmax(scores/temp, dim=-1)

    else:
        scores = scores**coeff
        scores = scores/scores.sum(-1, keepdim=True)
        
    return scores


def conditional_counts(labels, bin_scores, min_labels=5):
    label_counts = []
    for l_set in labels:
        label_dists, sub_label_dists = [], []
        for label in labels[l_set]:
            if type(labels[l_set][label]) != list:
                sub_counts = nested_conditional_counts(
                    labels[l_set][label], bin_scores, min_labels
                )
                if sub_counts is not None:
                    sub_label_dists.append(
                        sub_counts
                    )
                continue
            
            if len(labels[l_set][label])>=min_labels:
                label_dists.append(
                    bin_scores[labels[l_set][label],:].sum(0)
                )
                
        if label_dists:
            label_counts.append(torch.stack(label_dists))
        if sub_label_dists:
            label_counts.append(sub_label_dists)
            
    return label_counts

def nested_conditional_counts(labels, bin_scores, min_labels):
    label_dists = []
    for sub_label in labels:
        if len(labels[sub_label])>=min_labels:
            label_dists.append(
                bin_scores[labels[sub_label],:].sum(0)
            )
    if label_dists:
        return torch.stack(label_dists)

def conditional_h(counts):
    conditionals = []
    for count in counts:
        if type(count) != list:
            conditionals.append(
                entropy(count).mean()
            )
        else:
            sub_conditional = []
            for sub_count in count:
                sub_conditional.append(
                    entropy(sub_count).mean()
                )
            conditionals.append(torch.stack(sub_conditional).mean())
            
    return torch.stack(conditionals)

def disentanglement(counts):
    
    def disentangle(class_count):
        p_not_x = normalise(class_count.sum(0)-class_count)
        p_x = normalise(class_count)
        return js_divergence(p_not_x, p_x).mean()
    
    disentanglements = []
    for count in counts:
        if type(count) != list:
            
            disentanglements.append(
                disentangle(count)
            )
        else:
            sub_disentanglement = []
            for sub_count in count:
                sub_disentanglement.append(
                    disentangle(sub_count)
                )
            disentanglements.append(torch.stack(sub_disentanglement).mean())
            
    return torch.stack(disentanglements)

def js_divergence(p, q):
    
    p = (p/(p.sum(-1, keepdim=True).clamp(min=1e-9)))
    q = (q/(q.sum(-1, keepdim=True).clamp(min=1e-9)))
    
    m = (p+q)*0.5
    
    p = 0.5*(p*(p/m.clamp(min=1e-9)).clamp(min=1e-9).log()).sum(-1)
    q = 0.5*(q*(q/m.clamp(min=1e-9)).clamp(min=1e-9).log()).sum(-1)
    
    del m
    
    return p+q

def residual_h(mutual_info, h_y):
    mutual_info = mutual_info.__reversed__()
    residuals = []
    for i, x in enumerate(mutual_info[:-1]):
        residuals.append(
            x - mutual_info[i+1]
        )
    residuals.append(mutual_info[-1]) #add in the largest class
    y_residual = (h_y - mutual_info[0])/h_y
    return torch.stack(residuals).__reversed__()/h_y, y_residual

def normalise(dist):
    return dist/dist.sum(-1, keepdim=True).clamp(min=1e-9)

def entropy(dist):
    p = normalise(dist)
    return (-p*p.clamp(min=1e-9).log()).sum(-1)
    
def soft_measure(
    encodings, examples, n_bins=100, temp=1e-9, dist_fn='cosine_uniform', 
    pos_labels=None, label_sets=['pos','token','bigram','trigram'],
    n_heads=1, smoothing_fn="softmax"
):
    
    enc = get_encodings(encodings, incremental=True)
    label_ids, all_representations = get_token_encodings(enc, examples, pos_dict=pos_labels, labels=label_sets)
    label_sets = list(label_ids.keys())
    
    scores, bins, centres = soft_bin(
        all_representations, 
        n_bins, 
        temp=temp, 
        dist_fn=dist_fn,
        n_heads=n_heads,
        smoothing_fn=smoothing_fn,
    ) 
    #bin_dist = smoothing(scores, temp, sparse=sparse)
    label_counts = conditional_counts(label_ids, scores)
    
    total_scores =scores.sum(0)
    h_y, max_h_y = entropy(total_scores).mean(), entropy(torch.ones_like(total_scores)).mean()
    
    
    conditional = conditional_h(label_counts)
    mutual_info = h_y - conditional
    dis = disentanglement(label_counts)
    
    residuals, residual_y = residual_h(mutual_info, h_y)
    
    results = {
        'entropy/y':h_y.item(),
        'entropy/var_y':all_representations.var().item(),
        'efficiency/y':(h_y/max_h_y).item(),
        'residual/y':residual_y.item()
    }
    
    for i, l in enumerate(label_sets):
        results[f'entropy/{l}'] = conditional[i].item()
        results[f'variation/{l}'] = (conditional[i]/max_h_y).item()
        results[f'mutual_info/{l}'] = mutual_info[i].item()
        results[f'regularity/{l}'] = (mutual_info[i]/max_h_y).item()
        results[f'disentanglement/{l}'] = dis[i].item()
        results[f'residual/{l}'] = residuals[i].item()
    
    
    
    return results, h_y, conditional, mutual_info, dis