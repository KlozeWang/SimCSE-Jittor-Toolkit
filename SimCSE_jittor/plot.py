import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from datasets import load_from_disk
import matplotlib.pyplot as plt
import numpy as np

'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "./Task2/bert"
#model_name = './result/my-unsup-simcse-bert-base-uncased'
model = AutoModel.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#args = Namespace(cache_dir=None, config_name=None, dataset_config_name=None, dataset_name=None, do_eval=True, do_train=True, eval_steps=125, eval_transfer=False, gradient_accumulation_steps=1, hard_negative_weight=0, learning_rate=3e-05, load_best_model_at_end=True, max_seq_length=32, metric_for_best_model='stsb_spearman', mlp_only_train=True, model_name_or_path='Task2/bert', model_revision='main', model_type=None, num_train_epochs=1, output_dir='Task2/result/zh', overwrite_cache=False, overwrite_output_dir=True, pad_to_max_length=False, per_device_train_batch_size=1, pooler_type='cls', preprocessing_num_workers=None, seed=555, temp=0.05, tokenizer_name=None, train_file='Task2/zh.txt', use_auth_token=False, use_fast_tokenizer=False, validation_split_percentage=5)
#model = BertForCL(BertConfig(vocab_size=119547), model_kargs=args)
#print(model.state_dict()['encoder.layer.11.output.LayerNorm.weight'])
checkpoint = torch.load('./Task2/result/es_large/checkpoint-7000/model.pt', map_location='cpu')
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

data = []
with open('./data/wiki1m_for_simcse.txt', 'r') as file:
    for i, line in enumerate(file):
        if i == 5000:
            break
        data.append(line)
sts22_data = [line.strip() for line in data]

print('ok')

def get_sentence_embedding(sentence):
    """通过 BERT 提取句子的嵌入向量（使用池化输出）。"""
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.squeeze().cpu().numpy()

sentence_embeddings = []
similarity_scores = []
for sent1 in sts22_data:
    emb1 = get_sentence_embedding(sent1)
    sentence_embeddings.append(emb1)  # 句子 1 的嵌入

sentence_embeddings = np.array(sentence_embeddings)

def reduce_dimensions(embeddings, method="pca"):
    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        raise ValueError("Unsupported method: choose 'pca' or 'tsne'.")
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings

reduced_embeddings = reduce_dimensions(sentence_embeddings, method="tsne")

with open('./embedding.txt', 'a') as file:
    for x, y in zip(reduced_embeddings[:, 0], reduced_embeddings[:, 1]):
        file.write(str(x) + ' ' + str(y) + '\n')
exit()


def plot_embeddings(embeddings, similarity_scores, title="Semantic Space"):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings[:, 0], embeddings[:, 1], color='green', s=30, alpha=0.8
    )
    plt.colorbar(scatter, label="Similarity Score")
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()
    plt.savefig('va.png', bbox_inches='tight')

plot_embeddings(reduced_embeddings, similarity_scores, title="Semantic Space of STS22 Data")
'''
x_points = []
y_points = []
with open('./embedding.txt', 'r') as file:
    for line in file:
        xx, yy = line.strip().split()
        x_points.append(float(xx))
        y_points.append(float(yy))
        
plt.figure(figsize=(10, 8))
plt.title('Semantic Space of Wiki')

#plt.scatter(x_points[:5000], y_points[:5000], color='red', label='vanilla', alpha = 0.2, s=20)

# 中间5000个点画成红色
#plt.scatter(x_points[5000:10000], y_points[5000:10000], color='blue', label='zh', alpha = 0.2, s=20)

# 后5000个点画成蓝色
plt.scatter(x_points[10000:15000], y_points[10000:15000], color='green', label='es', alpha = 0.2, s=20)

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('es-new.png', bbox_inches='tight')