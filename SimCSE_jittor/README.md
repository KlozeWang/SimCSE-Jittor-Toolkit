## SimCSE: Simple Contrastive Learning of Sentence Embeddings 复现工作

本项目使用Jittor框架复现SimCSE，以下为框架的使用方法：
进入项目文件夹``cd SimCSE_jittor``

##### Unsupervised Learning
1. 下载数据
```
bash ./data/download_wiki.sh
```
2. 运行训练程序
```
bash jittor_unsup_example.sh
```

3. 运行格式转换程序
```
python simcse_to_huggingface.py --path [MODEL_PATH]
```

4. 测试，注意修改eval.sh中的文件夹路径
```
bash eval.sh
```

##### Supervised Learning
1. 下载数据
```
bash ./data/download_nli.sh
```
2. 运行训练程序
```
bash jittor_sup_example.sh
```

3. 运行格式转换程序
```
python simcse_to_huggingface.py --path [MODEL_PATH]
```

4. 测试，注意修改eval.sh中的文件夹路径
```
bash eval.sh
```

## SimCSE 跨语言迁移训练

我们提供了简单的脚本以供训练和评测。

在训练之前，创建 `Task2` 目录，并在 `Task2/bert` 中放入预训练模型，并在其中放入需要训练的语料（一个文件，里面是若干行句子）。从 https://huggingface.co/datasets/mteb/sts22-crosslingual-sts 中下载 datasets，并放入 `Task2/sts22` 中

修改 `train_for_task2.sh` 中对应的目录和需要放置 ckpt 的目录，运行：
```bash
bash train_for_task2.sh
```

对于获得的 ckpt，在 `evaluation_for_bert_auto.py` 的修改模型的目录和 ckpt 的目录，并运行：
```bash
bash train_for_task2.sh
```

sts22 的 zh，es，en，zh-en，es-en 五个部分的两个相关性系数会被输出。