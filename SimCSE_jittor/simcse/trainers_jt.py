import collections
import inspect
import math
import sys
import os
import re
import json
import shutil
import time
import warnings
from pathlib import Path
import importlib.util
from packaging import version
import jittor
from jittor import nn
from jittor.optim import LRScheduler
import random
import tqdm
PREFIX_CHECKPOINT_DIR = "checkpoint"
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from .models_bert import Similarity, BertForCL
from transformers import BertConfig
from torch.utils.tensorboard import SummaryWriter
import copy
# Set path to SentEval
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
import numpy as np
from datetime import datetime
from filelock import FileLock
import logging

logger = logging.getLogger(__name__)

class LinearLR(LRScheduler):
    def __init__(self, optimizer, total_iters):
        self.total_iters = total_iters
        super().__init__(optimizer)

    def get_lr(self):
        current_step = self._step_count
        num_training_steps = self.total_iters
        r = max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps))
        )
        return [base_lr * r for base_lr in self.base_lrs]

class CLTrainer:
    def __init__(self, args, train_dataset, tokenizer):
        self.args = args
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.best_metric = None

    def evaluate(
        self,
        eval_senteval_transfer: bool = False,
    ) -> Dict[str, float]:

        # SentEval prepare and batcher
        def prepare(params, samples):
            return

        def batcher(params, batch):
            sentences = [' '.join(s) for s in batch]
            batch = self.tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )
            
            _inputs = {}
            _inputs["input_ids"] = jittor.array(batch["input_ids"].detach().numpy())
            _inputs["token_type_ids"] = jittor.array(batch["token_type_ids"].detach().numpy())
            _inputs["attention_mask"] = jittor.array(batch["attention_mask"].detach().numpy())

            with jittor.no_grad():
                outputs = self.model(**_inputs, output_hidden_states=True, return_dict=True, sent_emb=True)
                pooler_output = outputs["pooler_output"]
            return jittor.misc.cpu(pooler_output)

        # Set params for SentEval (fastmode)
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                            'tenacity': 3, 'epoch_size': 2}

        se = senteval.engine.SE(params, batcher, prepare)
        tasks = ['STSBenchmark', 'SICKRelatedness']
        # if eval_senteval_transfer or self.args.eval_transfer:
        #     tasks = ['STSBenchmark', 'SICKRelatedness', 'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']
        self.model.eval()
        results = se.eval(tasks)
        
        stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
        sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]

        metrics = {"eval_stsb_spearman": stsb_spearman, "eval_sickr_spearman": sickr_spearman, "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2} 
        # if eval_senteval_transfer or self.args.eval_transfer:
        #     avg_transfer = 0
        #     for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
        #         avg_transfer += results[task]['devacc']
        #         metrics['eval_{}'.format(task)] = results[task]['devacc']
        #     avg_transfer /= 7
        #     metrics['eval_avg_transfer'] = avg_transfer
        print(metrics)

        return metrics
        
    def _save_checkpoint(self, model, metrics=None):
        """
        Compared to original implementation, we change the saving policy to
        only save the best-validation checkpoints.
        """

        # In all cases, including ddp/dp/deepspeed self.model is always a reference to the model we
        # want to save.
        # assert _model_unwrap(model) is self.model, "internal model should be a reference to self.model"

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater
            
            print("comparing....")
            if (
                self.best_metric is None
                or self.best_model_checkpoint is None
                or operator(metric_value, self.best_metric)
            ):
                output_dir = self.args.output_dir
                self.best_metric = metric_value
                self.best_model_checkpoint = output_dir
                print("saving best model...")

                jittor.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
        else:
            # Save model checkpoint
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}"
            output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
            
            jittor.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
    
    def speed_metrics(self, split, start_time, num_samples=None):
        """
        Measure and return speed performance metrics.

        This function requires a time snapshot `start_time` before the operation to be measured starts and this function
        should be run immediately after the operation to be measured has completed.

        Args:

        - split: name to prefix metric (like train, eval, test...)
        - start_time: operation start time
        - num_samples: number of samples processed
        """
        runtime = time.time() - start_time
        result = {f"{split}_runtime": round(runtime, 4)}
        if num_samples is not None:
            samples_per_second = 1 / (runtime / num_samples)
            result[f"{split}_samples_per_second"] = round(samples_per_second, 3)
        return result

    def train(self, model_path: Optional[str] = None):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        
        The main difference between ours and Huggingface's original implementation is that we 
        also load model_args when reloading best checkpoints for evaluation.
        """

        # Keeping track whether we can can len() on the dataset or not
        writer = SummaryWriter("./runs/jittor_sup_float16")
        train_dataloader = jittor.dataset.DataLoader(self.train_dataset, batch_size=self.args.per_device_train_batch_size, shuffle=True)
        # Data loader and number of training steps

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        num_update_steps_per_epoch = len(train_dataloader) //self.args.per_device_train_batch_size // self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
        num_train_epochs = math.ceil(self.args.num_train_epochs)

        
        # Check if saved optimizer or scheduler states exist
        # self._load_optimizer_and_scheduler(model_path)

        model = BertForCL(BertConfig(), model_kargs=self.model_args)
        print(f"Load from Model path: {os.path.join(model_path,'pytorch_model.bin')}")
        import torch
        param = torch.load(os.path.join(model_path,"pytorch_model.bin"))
        param_new = {}
        for key, value in param.items():
            if key.endswith("gamma"):
                key_new = key.split(".")
                key_new[-1] = "weight"
                key_new = ".".join(key_new)
            elif key.endswith("beta"):
                key_new = key.split(".")
                key_new[-1] = "bias"
                key_new = ".".join(key_new)
            else:
                key_new = key
            param_new[key_new] = value
        model.load_state_dict(param_new)
        self.optimizer = jittor.optim.AdamW(model.parameters(), lr=self.args.learning_rate)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), DDP(Deepspeed(Transformers Model)), etc.

        # Train!
        total_train_batch_size = self.args.per_device_train_batch_size

        gas = self.args.gradient_accumulation_steps
        num_examples = (
            len(train_dataloader)  * num_train_epochs
        )
        self.scheduler = LinearLR(self.optimizer, total_iters=num_examples//total_train_batch_size)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = jittor.array(0.0)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = 0
        # self._total_flos = self.state.total_flos
        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        model.train()
        self.model = model
        self.global_step = 0
        nowstep = 0
        # for epoch in range(epochs_trained, num_train_epochs):
        #     epoch_iterator = train_dataloader

        #     # Reset the past mems state at the beginning of each epoch if necessary.
        #     # if self.args.past_index >= 0:D
        #     #     self._past = None

        #     steps_in_epoch = len(train_dataloader)
        #     inputs = None
        #     last_inputs = None
        #     for step, inputs in enumerate(epoch_iterator):
        #         nowstep = nowstep + 1
        #         _inputs = {}
        #         _inputs["input_ids"] = jittor.array(inputs["input_ids"].detach().numpy())
        #         _inputs["token_type_ids"] = jittor.array(inputs["token_type_ids"].detach().numpy())
        #         _inputs["attention_mask"] = jittor.array(inputs["attention_mask"].detach().numpy())
        #         outputs = self.model(**_inputs)
        #         tr_loss = outputs["loss"]
        #         # tr_loss = self.compute_loss(outputs, self.args.temp)
        #         # zero grad & backward in jittor
                
        #         self.optimizer.zero_grad()
        #         self.optimizer.backward(tr_loss)
        #         self.optimizer.clip_grad_norm(1.0, 2)
            
        #         self.optimizer.step()
        #         # self.optimizer.step(loss = tr_loss)
        #         self.scheduler.step()
        #         self.global_step += 1

        #         if self.global_step % self.args.eval_steps == 0:
        #             print(f"{self.global_step} {inputs['input_ids'].shape} {self.scheduler.get_last_lr()[0]}")
        #             self.model.eval()
        #             metrics = self.evaluate()
                    
        #             writer.add_scalar('loss', tr_loss.detach().numpy(), nowstep)
        #             writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], nowstep)
        #             writer.add_scalar('avg_sts', metrics["eval_avg_sts"], nowstep)
        #             writer.add_scalar('stsb_spearman', metrics["eval_stsb_spearman"], nowstep)
        #             writer.add_scalar('sickr_spearman', metrics["eval_sickr_spearman"], nowstep)

        #             self._save_checkpoint(self.model, metrics)
        #             self.model.train()

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            # if self.args.past_index >= 0:D
            #     self._past = None

            steps_in_epoch = len(train_dataloader)
            inputs = None
            last_inputs = None
            for step, inputs in enumerate(epoch_iterator):
                nowstep = nowstep + 1
                _inputs = {}
                _inputs["input_ids"] = jittor.array(inputs["input_ids"].detach().numpy())
                _inputs["token_type_ids"] = jittor.array(inputs["token_type_ids"].detach().numpy())
                _inputs["attention_mask"] = jittor.array(inputs["attention_mask"].detach().numpy())
                outputs = self.model(**_inputs)
                tr_loss = outputs["loss"]
                # tr_loss = self.compute_loss(outputs, self.args.temp)
                # zero grad & backward in jittor
                
                if self.global_step % gas == 0:
                    self.optimizer.zero_grad()
                self.optimizer.backward(tr_loss / gas)
                self.global_step += 1
                
                if self.global_step % gas == 0:
                    self.optimizer.clip_grad_norm(1.0, 2)
                
                    self.optimizer.step()
                    # self.optimizer.step(loss = tr_loss)
                    self.scheduler.step()

                    if (self.global_step / gas) % self.args.eval_steps == 0:
                        print(f"{self.global_step / gas} {inputs['input_ids'].shape} {self.scheduler.get_last_lr()[0]}")
                        self.model.eval()
                        metrics = self.evaluate()
                        
                        writer.add_scalar('loss', tr_loss.detach().numpy(), nowstep / gas)
                        writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], nowstep / gas)
                        writer.add_scalar('avg_sts', metrics["eval_avg_sts"], nowstep / gas)
                        writer.add_scalar('stsb_spearman', metrics["eval_stsb_spearman"], nowstep / gas)
                        writer.add_scalar('sickr_spearman', metrics["eval_sickr_spearman"], nowstep / gas)

                        self._save_checkpoint(self.model, metrics)
                        self.model.train()

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        # if self.args.load_best_model_at_end and self.best_model_checkpoint is not None:
        #     logger.info(
        #         f"Loading best model from {self.best_model_checkpoint} (score: {self.best_metric})."
        #     )
        #     state_dict = jittor.load(os.path.join(self.best_model_checkpoint, "pytorch_model.bin"))
        #     self.model.load_state_dict(state_dict)

        metrics = self.speed_metrics("train", start_time, max_steps)
        print(metrics)
        writer.add_scalar('train_runtime', metrics["train_runtime"], 0)
        if "train_samples_per_second" in metrics:
            writer.add_scalar('train_samples_per_second', metrics["train_samples_per_second"], 0)

        self._total_loss_scalar += tr_loss.item()

        exit(0)

        return {
            "global_step":self.global_step, 
            "training_loss": self._total_loss_scalar / self.global_step, 
            "metrics":metrics
        }