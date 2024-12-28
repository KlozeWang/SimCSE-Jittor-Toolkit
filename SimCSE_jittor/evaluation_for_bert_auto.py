import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_from_disk
from scipy.stats import pearsonr, spearmanr
import numpy as np
import os
from simcse.models_bert import Similarity, BertForCL
from transformers import BertConfig
#import jittor
from argparse import Namespace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "./Task2/bert"
#model_name = './result/my-unsup-simcse-bert-base-uncased'
model = AutoModel.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#args = Namespace(cache_dir=None, config_name=None, dataset_config_name=None, dataset_name=None, do_eval=True, do_train=True, eval_steps=125, eval_transfer=False, gradient_accumulation_steps=1, hard_negative_weight=0, learning_rate=3e-05, load_best_model_at_end=True, max_seq_length=32, metric_for_best_model='stsb_spearman', mlp_only_train=True, model_name_or_path='Task2/bert', model_revision='main', model_type=None, num_train_epochs=1, output_dir='Task2/result/zh', overwrite_cache=False, overwrite_output_dir=True, pad_to_max_length=False, per_device_train_batch_size=1, pooler_type='cls', preprocessing_num_workers=None, seed=555, temp=0.05, tokenizer_name=None, train_file='Task2/zh.txt', use_auth_token=False, use_fast_tokenizer=False, validation_split_percentage=5)
#model = BertForCL(BertConfig(vocab_size=119547), model_kargs=args)
#print(model.state_dict()['encoder.layer.11.output.LayerNorm.weight'])
checkpoint = torch.load('./Task2/result/es_large/checkpoint-7000/model.pt', map_location=device)
new_state_dict = {k[5:]: v for k, v in checkpoint.items() if k.startswith('bert.')}
#print(model.state_dict()['encoder.layer.11.output.LayerNorm.bias'])
#exit()
#new_state_dict = jittor.load('./Task2/result/zh_large/model.pt')
#print(new_state_dict['encoder.layer.11.output.LayerNorm.weight'])
weight = model.state_dict()
pretrained_weights = {k: v for k, v in new_state_dict.items() if k in weight}
weight.update(pretrained_weights)
model.load_state_dict(weight)
#print(model.state_dict()['encoder.layer.11.output.LayerNorm.weight'])
#model.load_state_dict(checkpoint)
model.eval()

dataset = load_from_disk('./Task2/sts22')

def get_embeddings(sentences, batch_size=32):
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        batch = tokenizer.batch_encode_plus(
            batch_sentences,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=None
        )
        #for key in batch:
        #    batch[key] = batch[key].unsqueeze(0)
        #_batch = {}
        #_batch["input_ids"] = jittor.array(batch["input_ids"].detach().numpy())
        #_batch["token_type_ids"] = jittor.array(batch["token_type_ids"].detach().numpy())
        #_batch["attention_mask"] = jittor.array(batch["attention_mask"].detach().numpy())
        for key in batch:
            batch[key] = batch[key].to(device)

        # print(_batch)
        # print(len(batch_sentences))
        # print(_batch["token_type_ids"])
        with torch.no_grad():
            outputs = model(**batch)
            batch_embeddings = outputs.pooler_output.cpu().numpy()

        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

def evaluate_by_language(data, languages, batch_size=32):
    results = {}
    for lang in languages:
        lang_data = data.filter(lambda x: x['lang'] == lang)
        sentence1 = lang_data['sentence1']
        sentence2 = lang_data['sentence2']
        scores = lang_data['score']

        embeddings1 = get_embeddings(sentence1, batch_size)
        embeddings2 = get_embeddings(sentence2, batch_size)

        cosine_similarities = np.array([
            np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
            for e1, e2 in zip(embeddings1, embeddings2)
        ])

        pearson_corr, _ = pearsonr(cosine_similarities, scores)
        spearman_corr, _ = spearmanr(cosine_similarities, scores)

        results[lang] = {
            "Pearson": round(pearson_corr, 4),
            "Spearman": round(spearman_corr, 4),
            "Num_Samples": len(scores)
        }
    return results

languages = ["es", "es-en", "zh", "zh-en", "en"]

print("Evaluating test data...")
#jittor.flags.use_cuda = 1
test_results = evaluate_by_language(dataset['test'], languages, batch_size=30)

def print_results(results, split_name):
    print(f"Results for {split_name}:")
    for lang, metrics in results.items():
        print(f"  Language: {lang}")
        print(f"    Pearson: {metrics['Pearson']}")
        print(f"    Spearman: {metrics['Spearman']}")
        print(f"    Number of Samples: {metrics['Num_Samples']}")

print_results(test_results, "test")

'''
model_name = "./result/my-unsup-simcse-bert-base-uncased"
Results for test:
  Language: es
    Pearson: 0.4864
    Spearman: 0.5751
    Number of Samples: 200
  Language: es-en
    Pearson: 0.264
    Spearman: 0.256
    Number of Samples: 365
  Language: zh
    Pearson: 0.4125
    Spearman: 0.5258
    Number of Samples: 637
  Language: zh-en
    Pearson: 0.0253
    Spearman: 0.0795
    Number of Samples: 161
  Language: en
    Pearson: 0.5234
    Spearman: 0.5889
    Number of Samples: 197

model_name = "./Task2/bert"
Results for test:
  Language: es
    Pearson: -0.0507
    Spearman: 0.0239
    Number of Samples: 200
  Language: es-en
    Pearson: 0.0041
    Spearman: 0.0223
    Number of Samples: 365
  Language: zh
    Pearson: -0.0245
    Spearman: -0.0125
    Number of Samples: 637
  Language: zh-en
    Pearson: 0.0599
    Spearman: 0.0246
    Number of Samples: 161
  Language: en
    Pearson: -0.0012
    Spearman: -0.0329
    Number of Samples: 197
'''