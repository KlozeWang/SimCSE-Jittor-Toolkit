#from datasets import load_from_disk
#
#dataset = load_from_disk('./Task2/sts22')
#
#score = dataset['test']['score']
#print(score)

from datasets import load_from_disk
dataset = load_from_disk('./Task2/es_large')

#print(dataset['train']['text'][0])
file = dataset['train']['text'][:600000]

output_filename = './Task2/es_large.txt'

with open(output_filename, 'w', encoding='utf-8') as f:
    for d in file:
        f.write(d + '\n')

#from transformers import AutoModel, BertConfig

#config = BertConfig.from_pretrained('./Task2/bert')