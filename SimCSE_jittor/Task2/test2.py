#from datasets import load_from_disk
#
#dataset = load_from_disk('./Task2/sts22')
#
#score = dataset['test']['score']
#print(score)

from datasets import load_from_disk
dataset = load_from_disk('./Task2/zh_large')

#print(dataset)
#print(dataset['train']['text'][0])
file = dataset['train']['instruction_zh']
file += dataset['train']['input_zh']
file += dataset['train']['output_zh']

file = [line for line in file if line]

split_lines = []
for line in file:
    for block in line.split():
        parts = block.split("。")
        for i, part in enumerate(parts):
            stripped_part = part.strip()
            if stripped_part:
                if i < len(parts) - 1:
                    split_lines.append(part + "。")
                else:
                    if parts[-1] == '。':
                        split_lines.append(part+'。')
                    else:
                        split_lines.append(part)

file = [line for line in split_lines if line.strip()]

output_filename = './Task2/zh_large.txt'

with open(output_filename, 'w', encoding='utf-8') as f:
    for d in file:
        f.write(d + '\n')

#from transformers import AutoModel, BertConfig

#config = BertConfig.from_pretrained('./Task2/bert')