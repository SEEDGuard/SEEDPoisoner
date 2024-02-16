# TODO: An object oriented approach for fine-tuning and model evaluation


import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
# from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, AdamW,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer)

from utils import (compute_metrics, convert_examples_to_features,
                   output_modes, processors)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)}


class SequenceClassifier:
    def __init__(self, data_dir, model_type, model_name_or_path, task_name, output_dir):
        # Model parameters
        self.data_dir = data_dir
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.output_dir = output_dir

        # Training parameters
        self.max_seq_length = 128
        self.do_train = False
        self.do_eval = False
        self.do_predict = False
        self.per_gpu_train_batch_size = 8
        self.per_gpu_eval_batch_size = 8
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 3.0
        self.max_steps = -1
        self.warmup_steps = 0
        self.logging_steps = 50
        self.save_steps = 50
        self.seed = 42

        # Setup CUDA, GPU & distributed training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        # Set seed for reproducibility
        self.set_seed()

        # Model and Tokenizer initialization
        self.config_class, self.model_class, self.tokenizer_class = RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
        self.config = self.config_class.from_pretrained(model_name_or_path, num_labels=self.num_labels, finetuning_task=task_name)
        self.tokenizer = self.tokenizer_class.from_pretrained(model_name_or_path)
        self.model = self.model_class.from_pretrained(model_name_or_path, config=self.config)

        self.model.to(self.device)

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)

    def train(self, train_dataset, model, tokenizer, optimizer):
        self.train_batch_size = self.per_gpu_train_batch_size * max(1, self.n_gpu) #TODO: CHECK ARGS
        train_sampler = RandomSampler(train_dataset) if self.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.train_batch_size)

        if self.max_steps > 0:
            t_total = self.max_steps
            self.num_train_epochs = self.max_steps // (len(train_dataloader) // self.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.gradient_accumulation_steps * self.num_train_epochs

        scheduler = get_linear_schedule_with_warmup(optimizer, self.warmup_steps, t_total)

        # checkpoint_last = os.path.join(self.output_dir, 'checkpoint-last')
        # scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
        # if os.path.exists(scheduler_last):
        #     scheduler.load_state_dict(torch.load(scheduler_last))

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", self.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", self.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    self.train_batch_size * self.gradient_accumulation_steps * (
                        torch.distributed.get_world_size() if self.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = self.start_step
        tr_loss, logging_loss = 0.0, 0.0
        best_acc = 0.0
        model.zero_grad()
        train_iterator = trange(self.start_epoch, int(self.num_train_epochs), desc="Epoch",
                                disable=self.local_rank not in [-1, 0])
        set_seed()  # Added here for reproductibility (even between python 2 and 3)  #TODO: CALL SET SEED CORRECTLY
        model.train()
        for idx, _ in enumerate(train_iterator):
            tr_loss = 0.0
            for step, batch in enumerate(train_dataloader):

                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if self.model_type in ['bert', 'xlnet'] else None,
                          # XLM don't use segment_ids
                          'labels': batch[3]}
                ouputs = model(**inputs)
                loss = ouputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                if self.fp16:
                    try:
                        from apex import amp
                    except ImportError:
                        raise ImportError(
                            "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

                tr_loss += loss.item()
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if self.local_rank in [-1, 0] and self.logging_steps > 0 and global_step % self.logging_steps == 0:
                        # Log metrics
                        if self.local_rank == -1 and self.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                            results = evaluate(self, model, tokenizer, checkpoint=str(global_step))
                            for key, value in results.items():
                                # tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                                logger.info('loss %s', str(tr_loss - logging_loss))
                        # tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        # tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                        logging_loss = tr_loss
                if self.max_steps > 0 and global_step > self.max_steps:
                    # epoch_iterator.close()
                    break

            if self.do_eval and (self.local_rank == -1 or torch.distributed.get_rank() == 0):
                results = evaluate(args, model, tokenizer, checkpoint=str(args.start_epoch + idx))

                # last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                # if not os.path.exists(last_output_dir):
                #     os.makedirs(last_output_dir)
                # model_to_save = model.module if hasattr(model,
                #                                         'module') else model  # Take care of distributed/parallel training
                # model_to_save.save_pretrained(last_output_dir)
                # logger.info("Saving model checkpoint to %s", last_output_dir)
                # idx_file = os.path.join(last_output_dir, 'idx_file.txt')
                # with open(idx_file, 'w', encoding='utf-8') as idxf:
                #     idxf.write(str(args.start_epoch + idx) + '\n')
                #
                # torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
                # torch.save(scheduler.state_dict(), os.path.join(last_output_dir, "scheduler.pt"))
                # logger.info("Saving optimizer and scheduler states t/mnt/wanyao/zsjo %s", last_output_dir)
                #
                # step_file = os.path.join(last_output_dir, 'step_file.txt')
                # with open(step_file, 'w', encoding='utf-8') as stepf:
                #     stepf.write(str(global_step) + '\n')

                if (results['acc'] > best_acc):
                    best_acc = results['acc']
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_{}.bin'.format(idx)))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break

        # if args.local_rank in [-1, 0]:
        #     tb_writer.close()

        return global_step, tr_loss / global_step

    def evaluate(self, eval_dataset):
        """ Evaluate the model """
        # ... evaluation logic here

    def load_and_cache_examples(self, ttype='train'):
        """ Load and cache examples """
        # ... data loading logic here

    # ... all other necessary methods for the class

# Then you can use your class like this:
def main():
    # Initialize the SequenceClassifier
    classifier = SequenceClassifier(
        data_dir="./data",
        model_type="roberta",
        model_name_or_path="roberta-base",
        task_name="glue",
        output_dir="./models"
    )

    # Train the model
    if classifier.do_train:
        train_dataset = classifier.load_and_cache_examples(ttype='train')
        classifier.train(train_dataset)

    # Evaluate the model
    if classifier.do_eval:
        eval_dataset = classifier.load_and_cache_examples(ttype='eval')
        classifier.evaluate(eval_dataset)

if __name__ == "__main__":
    main()


