## LLaMA-SimCSE

使用SimCSE微调LLaMA Embedding

直接运行 `bert_eval.sh`，可以获得 `BERT` 没有经过微调的 STS 分数

下载 LLaMA-2-7b 预训练模型至 `./SimCSE_llama/llama/llama-2-7b`，运行 `python finetune.py` 或 `python finetune_jt.py` 即可在 torch 和 jittor 架构上对 LLaMA Embedding 进行微调，微调之后的模型会保存到 `./finetune/` 中

然后运行 `sh evaluation.sh` 即可获得微调之后的 STS 分数，若要选择测试的模型参数，在 `evaluation.py` 中修改 model 地址即可