import torch
from torch.utils.data.dataloader import DataLoader

import attr
import gc
import json
import os
import bisect
from functools import partial

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spacy

from datetime import datetime
from itertools import combinations, tee
from collections import defaultdict

from datasets import load_dataset
from tqdm.notebook import tqdm as nb_tqdm
from tqdm import tqdm

from transformers import BertTokenizerFast, BertModel, AutoTokenizer, AutoModel
from h.h_estimator import Information as Estimator

@attr.s
class EncoderState:
    '''
    An an object for storing the output of An Encoder model
    '''
    src_memory = attr.ib()
    tokens = attr.ib()
    mask = attr.ib()
    labels = attr.ib(default=[None])
        
class BERTAnalyser(object):
    def __init__(
        self, device:str ='cpu', h_estimator:Estimator=None, model_id:str='google/multiberts-seed_13',
        results_file_name:str='results', cache_dir:str=None, max_len:int=256,
        seed:int=999999, is_jupyter_nb:bool=True, 
    ) -> None:
        '''
        Class for Loading and Analysing a BERT Model
        
        This init function sets object variables, builds a defailt estimator if none is provided
        Sets global plot aesthetics, Sets torch random seed, Then attempts to load the specified
        model using HuggingFace Transformers
        
        device: str - pytorch device to load model into. usually cpu or cuda:0
        
        h_estimator: Estimator - a class imported from the h_estimator file in this directory
        it's a class for fast pytorch-based soft entropy estimation
        
        model_id: str - hugging face model path to load (needs to be a BERTmodel, an error is thrown otherwise)
        
        results_file_name:str - the relative path where a json file with any results will be automatically written
        
        cache_dir: str - path to where hugging face should cache resources
        
        max_len: int - maximum length the tokenizer will allow, any longer sequences this will be truncated
        
        seed: int - sets pytorch random seed for controlling randomness in data shuffle and estimation
        
        is_jupyter_nb : bool - renders progress bars in a notebook-friendly widget
    
        '''
        
        torch.cuda.empty_cache()
        gc.collect()
        
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.dataset=None
        self.cache_dir=cache_dir
        
        self.save_path = f'{results_file_name}.json'
        self.model_name = model_id
        self.is_notebook = is_jupyter_nb
        
        self.spacy_tagger = spacy.load("en_core_web_sm")
        
        if h_estimator is None:
            h_estimator = Estimator(n_bins=20, count_device='cpu', inference_device='cpu', n_heads=12)
            
        else:
            self.h = h_estimator
            
        sns.set(style='whitegrid', font='Arial')
        
        torch.manual_seed(seed)
        
        if model_id is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                cache_dir=f"{self.cache_dir}/hub",
                model_max_length=max_len,
                truncation=True, max_length=max_len,
                
            )
            self.model = AutoModel.from_pretrained(model_id, cache_dir=f"{self.cache_dir}/hub").to(device)
            
    def get_dataset(self, repo:str="nyu-mll/glue", subtask:str="mnli"):
        '''
        loads the specified dataset and subtask from hugging face or local cache
        defaults to loading MNLI entailment task from the GLUE benchmark
        
        repo: str - hugging face repo name
        subtask: str (optional) - repo subtask name
        '''
        self.dataset_name = repo if subtask is None else repo+'/'+subtask
        if subtask is not None:
            self.dataset = load_dataset(repo, subtask, cache_dir=f"{self.cache_dir}/datasets")
        else:
            self.dataset = load_dataset(repo, cache_dir=f"{self.cache_dir}/datasets")

        self.data_repo = repo
        self.data_subtask = subtask
        
    def get_example(self, batch):
        '''
        This method is called every batch and formats the data for ech specific dataset into
        the batch format needed by the get batch method. It returns a list of lists with each
        example sentence or pair
        '''
        
        if self.data_subtask in ['mnli-pos']:
            texts = [' '.join([i['premise'], i['hypothesis']]) for i in batch]
            return texts, self.tokens_pos_mapping(texts)
        
        if self.data_subtask in ['mnli', 'ax']:
            #return [i['premise'] for i in batch]
            return [' '.join([i['premise'], i['hypothesis']]) for i in batch], None
        
        if self.data_subtask in ['rte', 'mrpc', 'stsb', 'wnli']:
            return [' '.join([i['sentence1'], i['sentence2']]) for i in batch], None
        
        if self.data_subtask in ['qnli', 'sst2']:
            return [' '.join([i['question'], i['sentence']]) for i in batch], None
        
        if self.data_subtask in ['qqp']:
            return [' '.join([i['question1'], i['question2']]) for i in batch], None
        
        if self.data_subtask in ['cola', 'sst2']:
            return [i['sentence'] for i in batch], None
        
        if self.data_repo =="sentence-transformers/wikipedia-en-sentences":
            return [i['sentence'] for i in batch], None
         
        if 'mix' in self.data_repo:
            return [i['sentence'] for i in batch], [[i['language']] for i in batch]
        
        if 'hcoxec' in self.data_repo:
            return [i['sentence'] for i in batch], None
        
        raise NotImplementedError
        
    def get_batch(self, batch, pos_labels=False):
        words, labels = self.get_example(batch)
        tokenized = self.tokenizer(words, return_tensors='pt', padding=True, truncation=True, max_length=256).to(self.device)
        tokens = [tokenized['input_ids'][i][tokenized['input_ids'][i].nonzero()].T[0].tolist() 
                    for i in range(len(tokenized['input_ids']))]
        
        if labels is not None:
            lens = tokenized.attention_mask.sum(-1).tolist()
            labels = [x*lens[i] for i, x in enumerate(labels)]
            
        if pos_labels:
            labels = [self.tokens_pos_mapping(s) for s in words]
        
        return words, tokenized, tokens, labels
    
    def match_labels_to_tokens(self, labels, tokenized):
        if labels is None:
            return None
        
        word_ids = tokenized.word_ids()
        tags = []
        for i in tokenized.word_ids():
            if i is None:
                tags.append(99999999)
            else:
                tags.append(sent['pos_tags'][i])
                
    def tokens_pos_mapping(self, text, use_large_pos_tags=False):
        doc = self.spacy_tagger(text)
        
        token_stuff = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        tokens = token_stuff.input_ids
        mapping = token_stuff.offset_mapping
        decoded_tokens = [self.tokenizer.decode(token) for token in tokens]

        starts = [s if tok[0] != ' ' else s+1 for (s,_), tok in zip(mapping, decoded_tokens)]
        idxs = [tok.idx for tok in doc]
        locations = [bisect.bisect_right(idxs, start) - 1 for start in starts]
        spacy_tokens = ['[CLS]'] + [doc[location] for location in locations] + ['[SEP]']
        
        small_pos_tags = [doc.pos_ for doc in spacy_tokens]
        large_pos_tags = [doc.tag_ for doc in spacy_tokens]
        
        return large_pos_tags if use_large_pos_tags else small_pos_tags
        
        
    def get_dataloader(self, split="train", bs=512, repo="nyu-mll/glue", subtask="mnli", pos_labels=False):
        if self.dataset is None:
            self.get_dataset(repo, subtask)
        
        
        split_loader = DataLoader(
            self.dataset[split], 
            batch_size=bs, 
            shuffle=True, 
            collate_fn=partial(self.get_batch, pos_labels=pos_labels)
        )
        return split_loader
    
    def get_all_token_representations(self, encodings):
        
        all_representations = torch.cat(
            [batch.src_memory[batch.mask.gt(0)] for batch in encodings]
        )
        
        all_tokens = [ i for batch in encodings for i in batch.tokens]
        try:
            all_labels = [ i for batch in encodings for i in batch.labels]
        except:
            all_labels=None
        
        return all_representations, all_tokens, all_labels
    
    def encode(self, dataloader, max_batches=None):
        encodings = []
        iterator = nb_tqdm(dataloader) if self.is_notebook else tqdm(dataloader)
        for ib, data_batch in enumerate(iterator):
            if (max_batches is not None) and (ib > max_batches):
                break
                            
            encodings.append(
                self.forward(data_batch)
            )
        return encodings
            
    def forward(self, x):
        
        words, tokenized, tokens, labels = x
        with torch.no_grad():
            output = self.model(**tokenized)
            enc_state = EncoderState(
                src_memory=output.last_hidden_state,
                tokens=tokens,
                mask=tokenized['attention_mask'],
                labels=labels
            )
            
        return enc_state
    
    def online_estimation(self, dataloader, max_batches=None, additional_logs={}, label_types=['token','bigram','trigram']):
        if self.is_notebook:
            iterator = nb_tqdm(dataloader, total=max_batches, desc=f"Encoding | {self.dataset_name}")
        else:
            iterator = tqdm(dataloader, total=max_batches, desc=f"Encoding | {self.dataset_name}")

        for ib, data_batch in enumerate(iterator):
            if (max_batches is not None) and (ib >= max_batches):
                break
                
            encodings = [self.forward(data_batch)]
            
            all_representations, all_tokens, all_labels = self.get_all_token_representations(encodings)
            
            if label_types == ['token','bigram','trigram']:
                label_ids = get_token_encodings(examples=all_tokens, labels=label_types)
            elif all_labels is not None:
                label_ids = get_token_encodings(examples=all_tokens, labels=label_types, seq_labels=all_labels)
            self.h.batch_count(all_representations, label_ids)
            
            del all_representations
            del encodings
            torch.cuda.empty_cache()
            gc.collect()
        
        results = self.filter_results(self.h.analyse())
        results_df = self.build_df(results)
        self.record(results, additional=additional_logs)
        
        return results, results_df
    
    def filter_results(
        self, results:dict, 
        filters=['regularity','variation','disentanglement', 'residual']
    ) -> dict:
        '''
        returns only results whose keys in the results dict contain at least
        one of the filter strings. This is done for readability, in the event that
        the number of results from the analysis is too large
        '''
        to_return = {}
        for item in results:
            for filt in filters:
                if filt in item:
                    to_return[item] = results[item]
                    break
        
        return to_return
    
    def build_df(self, results:dict) -> pd.DataFrame:
        '''
        Constructs a Pandas DataFrame object from a dictionary of results
        Assumes that the results dictionary keys a strings with layout
        'measure/label' where measure is the 
        '''
        items = defaultdict(lambda: defaultdict(lambda: []))
        labels = []
        for i in results:
            measure, label = i.split('/')
            items[measure][label] = results[i]
            #items['label'][label] = label
            labels.append(label)
        df = pd.DataFrame.from_dict(items)
        df['label'] = df.index.values
        
        return df
    
    def record(self, data: dict, additional: dict = {}) -> None:
        '''
        Writes data to outfile, after adding any additional keys
        passed in from the additional dict. These can be parameters from
        the analysis - like number of samples, or a readable name of the model
        
        Note: the model id and dataset saved in the analyser object are 
        automatically added
        
        '''
        additional['model_id'] = self.model_name
        additional['dataset'] = self.dataset_name

        additional.update(data)

        with open(self.save_path, "a") as outfile:
            outfile.write(json.dumps(additional)+'\n')
            
    def get_time(self):
        return datetime.now().strftime("%H_%M_%S")
            
    def plot_residual(
        self, result_df:pd.DataFrame, filename:str ='residual', title='Proportions',
        palette_name:str = "BuGn"
    )-> None:
        ''''
        Generates a Pie-Plot visualing the proportions of representational space
        explained by different labels
        accepts a dataframe of results, the directory to save the file in, plot title
        and the name of the seaborn color palette to use (see all of them here: https://www.practicalpythonfordatascience.com/ap_seaborn_palette)
        '''
        residual_values = result_df['residual'].values
        residual_labels = result_df['residual'].index.values
        residual_labels = [x if x != 'overall' else 'residual' for x in residual_labels]
        colors = sns.color_palette(palette_name, len(residual_labels))


        plt.pie(residual_values, labels = residual_labels, colors=colors, autopct='%.2f%%')
        plt.title(**{'label':title, 'loc':'left', 'fontsize':16, 'fontname':'Helvetica'})
        
        path = os.path.join('visuals', filename)
        os.makedirs(path, exist_ok= True)
        
        plt.savefig(f'{path}/{filename}_{self.get_time()}.png',  dpi=400)
        
    def plot(
        self, result_df:pd.DataFrame, measure='variation', filename:str ='plot',
        palette_name:str = "PuBuGn"
    )-> None:
        ax = sns.barplot(
            result_df, 
            x=measure,  
            y='label',  
            hue='label', 
            palette=palette_name, 
            legend=False, 
        )
        
        for container in ax.containers:
            ax.bar_label(container, fontsize=10)
            plt.title(**{'label':measure, 'loc':'left', 'fontsize':16, 'fontname':'Helvetica'})
        
        path = os.path.join('visuals', measure)
        os.makedirs(path, exist_ok= True)
        
        plt.savefig(f'{path}/{filename}_{self.get_time()}.png',  dpi=400)
                
                    
    
def n_gram(sequence, n):
    iterables = tee(sequence, n)

    for i, sub_iterable in enumerate(iterables):  # For each window,
        for _ in range(i):  # iterate through every order of ngrams
            next(sub_iterable, None)  # generate the ngrams within the window.
            
    return list(zip(*iterables))  # Unpack and flattens the iterables.

def idx_ngrams(max_len, n):
    idxs = n_gram(range(max_len), n=n)
    overlap_matrix = []
    for i in range(max_len):
        ii = []
        for inn, ngr in enumerate(idxs):
            if i in ngr:
                ii.append(inn)
        overlap_matrix.append(ii)
        
    return overlap_matrix

def get_token_encodings(
    examples,
    seq_labels=None,
    labels=['token','bigram','trigram'],
    pos_dict=None
):
    token_ids = defaultdict(lambda: [])
    pos_ids = defaultdict(lambda: [])
    general_label_ids = defaultdict(lambda: [])
    bigram_ids = defaultdict(lambda: defaultdict(lambda: []))
    trigram_ids = defaultdict(lambda: defaultdict(lambda: []))
    bow_ids = defaultdict(lambda: defaultdict(lambda: []))
    token_info = []
    
    token_vectors = []
    
    i = 0 #assign an id to each token instance in the dataset
    for i_e, example in enumerate(examples):
        tokens = example
        bigrams, trigrams = n_gram(tokens, n=2), n_gram(tokens, n=3)
        bigram_idxs, trigram_idxs = idx_ngrams(len(tokens), n=2), idx_ngrams(len(tokens), n=3)
        for i_t, token in enumerate(tokens):
            #token = token.lower() if uncased else token
            token_ids[token].append(i)
            bis, tris = [], []
            
            if ('pos' in labels) and (pos_dict is not None):
                token_pos = pos_dict[token]
                pos_ids[token_pos].append(i)
                
            if ('language' in labels) and (seq_labels is not None):
                general_label_ids[seq_labels[i_e][i_t]].append(i)
            
                
            if 'bigram' in labels:
                for bi in bigram_idxs[i_t]:
                    bigram_ids[token][bigrams[bi]].append(i)
                    bis.append(bigrams[bi])
            
            if 'trigram' in labels:
                for tri in trigram_idxs[i_t]:
                    trigram_ids[token][trigrams[tri]].append(i)
                    tris.append(trigrams[tri])
            
            if 'bow' in labels:
                for i_tt, other_token in enumerate(tokens):
                    if i_t != i_tt:
                        bow_ids[token][other_token].append(i)
                            
            i+=1
            
    ids = {
        'token': token_ids,
        'pos':pos_ids,
        'bigram':bigram_ids,
        'trigram':trigram_ids,
        'bow':bow_ids,
        'language':general_label_ids,
    }
    
    label_ids = {}
    for id_type in labels:
        label_ids[id_type] = ids[id_type]
    
    return label_ids