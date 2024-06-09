import torch
import h.soft_h as sh

from collections import defaultdict


class Information(object):
    def __init__(
        self, n_bins=20, 
        temp=1e-9,
        dist_fn='cosine_uniform',
        n_heads=12,
        smoothing_fn="softmax",
        count_device='cpu',
        inference_device='cpu',
    ) -> None:
       
        self.temp = temp
        self.dist_fn=dist_fn
        self.n_heads=n_heads
        self.smoothing_fn = smoothing_fn
        self.n_bins= n_bins
        self.count_device=count_device
        self.inference_device=inference_device
        
        self.y_counts = torch.zeros([n_heads, n_bins]).to(self.count_device)
        self.conditionals = defaultdict(lambda: defaultdict(lambda: torch.zeros([n_heads, n_bins]).to(self.count_device)))
        self.sub_conditionals = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: torch.zeros([n_heads, n_bins]).to(self.count_device)
                )
            )
        )
        
        self.cached_bins = None
        
    def reset(self, keep_bins=True):
        self.y_counts = torch.zeros([self.n_heads, self.n_bins]).to(self.count_device)
        self.conditionals = defaultdict(lambda: defaultdict(lambda: torch.zeros([self.n_heads, self.n_bins]).to(self.count_device)))
        self.sub_conditionals = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: torch.zeros([self.n_heads, self.n_bins]).to(self.count_device)
                )
            )
        )
        
        if keep_bins ==False:
            self.cached_bins = None
        
    def batch_count(self, token_encodings, batch_labels):
        
        scores, bins, centres, used_bins  = sh.soft_bin(
            token_encodings, 
            self.n_bins, 
            temp=self.temp, 
            dist_fn=self.dist_fn,
            n_heads=self.n_heads,
            smoothing_fn=self.smoothing_fn,
            online_bins=self.cached_bins
        )
        
        scores = scores.to(self.count_device)
        
        if self.cached_bins is None:
            self.cached_bins = used_bins
            
        self.y_counts += scores.sum(0)
        self.conditional_counts(batch_labels, scores)
        
    def analyse(self):
        
        label_counts, label_sets = self.coallate_conditionals()
        h_y, max_h_y = sh.entropy(self.y_counts).mean(), sh.entropy(torch.ones_like(self.y_counts)).mean()
        
        
        conditional = sh.conditional_h(label_counts)
        mutual_info = h_y - conditional
        dis = sh.disentanglement(label_counts)
        
        residuals, residual_y = sh.residual_h(mutual_info, h_y)
        
        results = {
            'entropy/overall':h_y.item(),
            'efficiency/overall':(h_y/max_h_y).item(),
            'variation/overall':(h_y/max_h_y).item(),
            'residual/overall':residual_y.item(),
            'disentanglement/overall':0,
            'regularity/overall':0
        }
        
        for i, l in enumerate(label_sets):
            results[f'entropy/{l}'] = conditional[i].item()
            results[f'variation/{l}'] = (conditional[i]/max_h_y).item()
            results[f'mutual_info/{l}'] = mutual_info[i].item()
            results[f'regularity/{l}'] = (mutual_info[i]/max_h_y).item()
            results[f'disentanglement/{l}'] = dis[i].item()
            results[f'residual/{l}'] = residuals[i].item()
        
        
        
        return results #, h_y, conditional, mutual_info, dis
        
    def conditional_counts(self, labels, scores, min_labels=5):
        label_counts = []
        for l_set in labels:
            label_dists, sub_label_dists = [], []
            for label in labels[l_set]:
                if type(labels[l_set][label]) != list:
                    sub_counts = self.nested_conditional_counts(
                        labels, l_set, label, scores, min_labels
                    )
                    if sub_counts is not None:
                        sub_label_dists.append(
                            sub_counts
                        )
                    continue
                
                if len(labels[l_set][label])>=min_labels:
                    #label_dists.append(
                    #    bin_scores[labels[l_set][label],:].sum(0)
                    #)
                    
                    self.conditionals[l_set][label] += scores[labels[l_set][label],:].sum(0)
            if label_dists:
                label_counts.append(torch.stack(label_dists))
            if sub_label_dists:
                label_counts.append(sub_label_dists)
                

    def nested_conditional_counts(self, labels, l_set, super_label, scores, min_labels):
        label_dists = []
        for sub_label in labels[l_set][super_label]:
            if len(labels[l_set][super_label][sub_label])>=min_labels:
                self.sub_conditionals[l_set][super_label][sub_label] += scores[labels[l_set][super_label][sub_label],:].sum(0)
                #label_dists.append(
                #    scores[labels[sub_label],:].sum(0)
                #)
        if label_dists:
            return torch.stack(label_dists)
        
    def coallate_conditionals(self):
        conditionals, label_set_names = [], []
        for l_set in self.conditionals:
            label_set_names.append(l_set)
            set_conditional = []
            for label in self.conditionals[l_set]:
                set_conditional.append(
                    self.conditionals[l_set][label]
                )
            conditionals.append(set_conditional)
            
        for l_set in self.sub_conditionals:
            label_set_names.append(l_set)
            set_conditional = []
            for super_label in self.sub_conditionals[l_set]:
                sub_conditional = []
                for sub_label in self.sub_conditionals[l_set][super_label]:
                    sub_conditional.append(
                        self.sub_conditionals[l_set][super_label][sub_label]
                    )
                set_conditional.append(torch.stack(sub_conditional))
            conditionals.append(set_conditional)
                
        return conditionals, label_set_names
                    
                    
        
        