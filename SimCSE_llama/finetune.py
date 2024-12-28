import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AdamW
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from sentence_transformers import models, SentenceTransformer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import sys
# 加载预训练的LLaMA-2-7b模型和分词器
model_path = "/home/aiuser/SimCSE/llama/llama-2-7b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
PATH_TO_SENTEVAL = '/home/aiuser/SimCSE/SentEval'
PATH_TO_DATA = '/home/aiuser/SimCSE/SentEval/data'
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
import numpy as np
from datetime import datetime
from filelock import FileLock
import logging

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

model = SimEmbedding()
# model = nn.Embedding(num_embeddings=32000, embedding_dim=4096)
model_state_dict = torch.load('/home/aiuser/SimCSE/llama/llama-2-7b/pytorch_model-00001-of-00002.bin')
embedding_state_dict = {'weight': value for key, value in model_state_dict.items() if 'model.embed_tokens' in key}
model.embedding.load_state_dict(embedding_state_dict)
# 数据集加载
data_files = {'train': '/home/aiuser/SimCSE/data/wiki1m_for_simcse.txt'}
dataset = load_dataset('text', data_files=data_files)

print(f'begin to deal dataset')

best_eval = 0

def evaluate(
) -> Dict[str, float]:

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch):
        sentences = [' '.join(s) for s in batch]
        batch_num = len(sentences)
        embeddings = []
        with torch.no_grad():
            for sentence in sentences:
                input = tokenizer(sentence, return_tensors="pt")
                embeddings.append((model(input.input_ids.cuda())[0]).view(-1))
            max_length = 4096 * 2
            padded_embeddings = [torch.cat((embedding, torch.zeros(max_length - embedding.size(0), dtype=embedding.dtype).cuda())) for embedding in embeddings]
            output = torch.cat(padded_embeddings).reshape(batch_num, -1)
        return output.detach().cpu()

    # Set params for SentEval (fastmode)
    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
    params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                        'tenacity': 3, 'epoch_size': 2}

    se = senteval.engine.SE(params, batcher, prepare)
    tasks = ['STSBenchmark', 'SICKRelatedness']
    results = se.eval(tasks)
    
    stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
    sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]

    metrics = {"eval_stsb_spearman": stsb_spearman, "eval_sickr_spearman": sickr_spearman, "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2} 
    return metrics

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
print(tokenized_datasets)

def collate_fn(batch):
    # 提取批次的 input_ids 和 attention_mask
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]

    # 将 input_ids 和 attention_mask 填充到相同的长度
    # padding_value 通常设置为模型的 PAD_TOKEN_ID，例如对于 BERT 是 0
    input_ids_padded = pad_sequence([torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence([torch.tensor(mask) for mask in attention_mask], batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded
    }

# DataLoader
train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=64, collate_fn=collate_fn)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# SimCSE损失函数
def simcse_loss(embeddings, temperature=0.05):
    batch_size = embeddings.size(0) // 2
    labels = torch.arange(batch_size).to(embeddings.device)
    sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    sim_matrix = sim_matrix / temperature
    sim_matrix = sim_matrix[:batch_size, batch_size:]
    sim_matrix = torch.exp(sim_matrix)
    sum = torch.sum(sim_matrix, dim=1)
    diag = torch.diag(sim_matrix)
    d = diag / sum
    logd = torch.log(d)
    loss = torch.mean(-logd)
    return loss

# 训练循环
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.train()

print(f'begin with {device} ...')

num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    print(f'len: {len(train_dataloader)}')
    batch_num = 0
    for batch in train_dataloader:
        batch_num += 1
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 正常的前向传播
        outputs1, outputs2 = model(batch['input_ids'])
        embeddings1 = outputs1[:, :].reshape(batch['input_ids'].shape[0], -1)
        embeddings2 = outputs2[:, :].reshape(batch['input_ids'].shape[0], -1)

        # 复制嵌入以形成正样本对
        embeddings = torch.cat([embeddings1, embeddings2], dim=0)
        
        # 计算SimCSE损失
        loss = simcse_loss(embeddings)
        total_loss += loss.item()
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_num % 125 == 0:
            model.eval()
            eval = evaluate()['eval_stsb_spearman']
            if eval > best_eval:
                best_eval = eval
                torch.save(model.state_dict(), f"./finetune/best_model_{eval}.pth")
            model.train()
            print(f'batch: {batch_num} eval score: {best_eval}')
    
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")
    torch.save(model.state_dict(), f"./finetune/model_best_{epoch}.pth")

# 保存微调后的模型
