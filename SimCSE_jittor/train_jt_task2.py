import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
import random
import jittor
from jittor.dataset import DataLoader, Dataset
from transformers import BertTokenizer, BertConfig

from datasets import load_dataset
import argparse
from simcse.models_bert import BertForCL
from simcse.trainers_jt_task2 import CLTrainer

def set_seed(seed):
    random.seed(seed)
    jittor.misc.set_global_seed(seed)

logger = logging.getLogger(__name__)

def getModelArguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Where do you want to store the pretrained models downloaded from huggingface.co"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where do you want to store the pretrained models to local"
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        help="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
    )
    parser.add_argument(
        "--model_revision",
        type=str,
        default="main",
        help="The specific model version to use (can be a branch name, tag name or commit id)."
    )
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        help="Will use the token generated when running `transformers-cli login` (necessary to use this script with private models)."
    )

    parser.add_argument(
        "--temp",
        type=float,
        default=0.05,
        help="Temperature for softmax."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=555,
        help="seed for training"
    )
    parser.add_argument(
        "--pooler_type",
        type=str,
        default="cls",
        help="What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
    )
    parser.add_argument(
        "--hard_negative_weight",
        type=float,
        default=0,
        help="The **logit** of weight for hard negatives (only effective if hard negatives are used)."
    )



def getDataTrainingArguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library)."
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library)."
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the output dir training and evaluation sets"
    )
    parser.add_argument(
        "--validation_split_percentage",
        type=int,
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="gradient_accumulation_steps"
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="The learning rate of training"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=128,
        help="The batch size of training"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=125,
        help="eval steps"
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="stsb_spearman",
        help="metric_for_best_model"
    )
    parser.add_argument(
        "--load_best_model_at_end",
        action="store_true",
        default=True,
        help="load_best_model_at_end"
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        default=True,
        help="do train"
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        default=True,
        help="do eval"
    )

    # SimCSE's arguments
    parser.add_argument(
        "--train_file",
        type=str,
        default=None, 
        help="The training data file (.txt or .csv)."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=32,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when batching to the maximum length in the batch."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Training epochs."
    )
    parser.add_argument(
        "--mlp_only_train",
        action="store_true",
        default=False,
        help="MLP only train"
    )

    # def __post_init__(self):
    #     if self.dataset_name is None and self.train_file is None and self.validation_file is None:
    #         raise ValueError("Need either a dataset name or a training/validation file.")
    #     else:
    #         if self.train_file is not None:
    #             extension = self.train_file.split(".")[-1]
    #             assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."


def getOurTrainingArguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--eval_transfer",
        action="store_true",
        default=False,
        help="Evaluate transfer task dev sets (in validation)."
    )

    # def _setup_devices(self) -> str:
    #     logger.info("PyTorch: setting up devices")
    #     if self.no_cuda:
    #         device = "cpu"
    #         self._n_gpu = 0
    #     else:
    #         jittor.misc.cuda(0)
    #         device = "cuda:0" if jittor.flags.use_cuda else "cpu"
    #         self._n_gpu = 1
        
    #     return device

class JittorDataset(Dataset):
    def __init__(self, train_data, tokenizer, padding: Union[bool, str] = True, max_length: Optional[int] = None, pad_to_multiple_of: Optional[int] = None):
        super().__init__()
        self.data = train_data
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def collate_batch(self, features):
        special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels']
        bs = len(features)
        if bs > 0:
            num_sent = len(features[0]['input_ids'])
        else:
            return
        flat_features = []
        for feature in features:
            for i in range(num_sent):
                flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})
        # print(flat_features[0])

        batch = self.tokenizer.pad(
            flat_features,
            padding='max_length',
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        return batch

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    logger.info("PyTorch: setting up devices")
    jittor.misc.cuda(0)
    device = "cuda:0" if jittor.flags.use_cuda else "cpu"
    parser = argparse.ArgumentParser()
    getModelArguments(parser)
    getDataTrainingArguments(parser)
    getOurTrainingArguments(parser)
    training_args = parser.parse_args()
    data_args = parser.parse_args()
    model_args = parser.parse_args()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    extension = data_args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    if extension == "csv":
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/", delimiter="\t" if "tsv" in data_args.train_file else ",")
    else:
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/")

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = BertConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = BertConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        raise NotImplementedError
        # config = CONFIG_MAPPING[model_args.model_type]()
        # logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = BertTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # if model_args.model_name_or_path:
    #     if 'roberta' in model_args.model_name_or_path:
    #         raise NotImplementedError
    #         model = RobertaForCL.from_pretrained(
    #             model_args.model_name_or_path,
    #             from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #             config=config,
    #             cache_dir=model_args.cache_dir,
    #             revision=model_args.model_revision,
    #             use_auth_token=True if model_args.use_auth_token else None,
    #             model_args=model_args                  
    #         )
    #     elif 'bert' in model_args.model_name_or_path:
    #         model = BertForCL.from_pretrained(
    #             model_args.model_name_or_path,
    #             from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #             config=config,
    #             cache_dir=model_args.cache_dir,
    #             revision=model_args.model_revision,
    #             use_auth_token=True if model_args.use_auth_token else None,
    #             model_args=model_args
    #         )
    #     else:
    #         raise NotImplementedError
    # else:
    #     raise NotImplementedError
    #     logger.info("Training new model from scratch")
    #     model = AutoModelForMaskedLM.from_config(config)

    # model.resize_token_embeddings(len(tokenizer))

    # Prepare features
    column_names = datasets["train"].column_names
    sent2_cname = None
    if len(column_names) == 2:
        # Pair datasets
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
    elif len(column_names) == 3:
        # Pair datasets with hard negatives
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
        sent2_cname = column_names[2]
    elif len(column_names) == 1:
        # Unsupervised datasets
        sent0_cname = column_names[0]
        sent1_cname = column_names[0]
    else:
        raise NotImplementedError

    def prepare_features(examples):
        # padding = longest (default)
        #   If no sentence in the batch exceed the max length, then use
        #   the max sentence length in the batch, otherwise use the 
        #   max sentence length in the argument and truncate those that
        #   exceed the max length.
        # padding = max_length (when pad_to_max_length, for pressure test)
        #   All sentences are padded/truncated to data_args.max_seq_length.
        total = len(examples[sent0_cname])

        # Avoid "None" fields 
        for idx in range(total):
            if examples[sent0_cname][idx] is None:
                examples[sent0_cname][idx] = " "
            if examples[sent1_cname][idx] is None:
                examples[sent1_cname][idx] = " "
        
        sentences = examples[sent0_cname] + examples[sent1_cname]

        # If hard negative exists
        if sent2_cname is not None:
            for idx in range(total):
                if examples[sent2_cname][idx] is None:
                    examples[sent2_cname][idx] = " "
            sentences += examples[sent2_cname]

        sent_features = tokenizer(
            sentences,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        features = {}
        if sent2_cname is not None:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total], sent_features[key][i+total*2]] for i in range(total)]
        else:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]
            
        return features

    if training_args.do_train:
        train_dataset = datasets["train"].map(
            prepare_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    # @dataclass
    # class OurDataCollatorWithPadding:

    #     tokenizer: PreTrainedTokenizerBase
    #     padding: Union[bool, str, PaddingStrategy] = True
    #     max_length: Optional[int] = None
    #     pad_to_multiple_of: Optional[int] = None
    #     mlm: bool = True
    #     mlm_probability: float = data_args.mlm_probability

    #     def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    #         special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels']
    #         bs = len(features)
    #         if bs > 0:
    #             num_sent = len(features[0]['input_ids'])
    #         else:
    #             return
    #         flat_features = []
    #         for feature in features:
    #             for i in range(num_sent):
    #                 flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

    #         batch = self.tokenizer.pad(
    #             flat_features,
    #             padding=self.padding,
    #             max_length=self.max_length,
    #             pad_to_multiple_of=self.pad_to_multiple_of,
    #             return_tensors="pt",
    #         )

    #         batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}

    #         if "label" in batch:
    #             batch["labels"] = batch["label"]
    #             del batch["label"]
    #         if "label_ids" in batch:
    #             batch["labels"] = batch["label_ids"]
    #             del batch["label_ids"]

    #         return batch
    train_dataset = JittorDataset(train_dataset, 
                                  tokenizer=tokenizer, 
                                  padding="max_length" if data_args.pad_to_max_length else False, 
                                  max_length=data_args.max_seq_length,
                                  pad_to_multiple_of=8
                                )

    trainer = CLTrainer(
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer
    )
    trainer.model_args = model_args

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        train_result = trainer.train(model_path=model_path)
        # trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result["metrics"].items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")


    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate(eval_senteval_transfer=True)

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key, value in sorted(results.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

    return results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
