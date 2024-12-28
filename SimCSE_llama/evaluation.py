import sys
import io, os
import numpy as np
import logging
import argparse
from prettytable import PrettyTable
import torch
import transformers
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
from sentence_transformers import models, SentenceTransformer

class SimEmbedding(nn.Module):
    def __init__(self, vocab_size=32000, embedding_dim=4096, dropout_prob=0.1):
        super(SimEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.pooling = models.Pooling(4096, pooling_mode_cls_token=True)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pooling({'token_embeddings': x})
        x = x['sentence_embedding']
        output1 = self.dropout(x)
        output2 = self.dropout(x)
        return output1, output2

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = '/home/aiuser/SimCSE/SentEval'
PATH_TO_DATA = '/home/aiuser/SimCSE/SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, 
            help="Transformers' model name or path")
    parser.add_argument("--pooler", type=str, 
            choices=['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last'], 
            default='cls', 
            help="Which pooler to use")
    parser.add_argument("--mode", type=str, 
            choices=['dev', 'test', 'fasttest'],
            default='test', 
            help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--task_set", type=str, 
            choices=['sts', 'transfer', 'full', 'na'],
            default='sts',
            help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument("--tasks", type=str, nargs='+', 
            default=['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                     'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC',
                     'SICKRelatedness', 'STSBenchmark'], 
            help="Tasks to evaluate on. If '--task_set' is specified, this will be overridden")
    
    args = parser.parse_args()
    
    # Load transformers' model checkpoint
    # model = AutoModel.from_pretrained(args.model_name_or_path)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # Load llama embedding checkpoint
    # model = nn.Embedding(num_embeddings=32000, embedding_dim=4096)
    model = SimEmbedding()
    # model_state_dict = torch.load('/home/aiuser/SimCSE_ori/llama/llama-2-7b/pytorch_model-00001-of-00002.bin')
    # embedding_state_dict = {'weight': value for key, value in model_state_dict.items() if 'model.embed_tokens' in key}
    # model_state_dict = torch.load('/home/aiuser/SimCSE_ori/finetune/model_weights2.pth')
    model_state_dict = torch.load('/home/aiuser/SimCSE/finetune/best_model_0.7046846693211393.pth')
    embedding_state_dict = {'weight': value for key, value in model_state_dict.items() if 'embedding.weight' in key}
    model.load_state_dict(model_state_dict)
    tokenizer = AutoTokenizer.from_pretrained("/home/aiuser/SimCSE/llama/llama-2-7b")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                         'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                         'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return
    
    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]

        # Tokenization
        # if max_length is not None:
        #     batch = tokenizer.batch_encode_plus(
        #         sentences,
        #         return_tensors='pt',
        #         padding=True,
        #         max_length=max_length,
        #         truncation=True
        #     )
        # else:
        #     batch = tokenizer.batch_encode_plus(
        #         sentences,
        #         return_tensors='pt',
        #         padding=True,
        #     )

        batch_num = len(sentences)
        embeddings = []
        for sentence in sentences:
            input = tokenizer(sentence, return_tensors="pt")
            embeddings.append((model(input.input_ids.cuda())[0]).view(-1))
            # embeddings.append(torch.randn(4096*100).cuda())

        # print(embeddings[0].shape)

        max_length = 4096 * 2
        padded_embeddings = [torch.cat((embedding, torch.zeros(max_length - embedding.size(0), dtype=embedding.dtype).cuda())) for embedding in embeddings]
        output = torch.cat(padded_embeddings).reshape(batch_num, -1)
        return output.detach().cpu()

        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device)
        
        # Get raw embeddings
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True, return_dict=True)
            last_hidden = outputs.last_hidden_state
            pooler_output = outputs.pooler_output
            hidden_states = outputs.hidden_states

        # Apply different poolers
        if args.pooler == 'cls':
            # There is a linear+activation layer after CLS representation
            return pooler_output.cpu()
        elif args.pooler == 'cls_before_pooler':
            return last_hidden[:, 0].cpu()
        elif args.pooler == "avg":
            return ((last_hidden * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)).cpu()
        elif args.pooler == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        elif args.pooler == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        else:
            raise NotImplementedError

    results = {}

    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result
    
    # Print evaluation results
    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))    
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

    elif args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['acc']))    
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)


if __name__ == "__main__":
    main()