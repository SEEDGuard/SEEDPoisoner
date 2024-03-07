from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import gzip
import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json
from itertools import cycle

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import codecs
import multiprocessing
from data_preprocess import Data_Preprocessor
from Bart import Bart_seq2seq
from t5 import T5_seq2seq
# from r_bleu import _bleu
from tree_sitter import Language, Parser

import torch.nn.utils.prune as prune

cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BartConfig, BartModel, BartTokenizer,
                          PLBartConfig, PLBartModel, PLBartTokenizer, PLBartForConditionalGeneration,
                          T5Config, T5ForConditionalGeneration, RobertaTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bart': (BartConfig, BartModel, BartTokenizer),
    'plbart': (PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer),
    't5':(T5Config, T5ForConditionalGeneration, RobertaTokenizer)
}

class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 idx,
                 input_ids,
                 tgt_ids
                 ):
        self.idx = idx
        self.tgt_ids = tgt_ids
        self.input_ids = input_ids

class TextDataset(Dataset):
    def __init__(self, preprocessor, args, isevaluate, lang, file_path, block_size=512):
        self.args = args
        self.preprocessor = preprocessor
        self.tokenizer = preprocessor.tokenizer
        if args.local_rank == -1:
            local_rank = 0
            world_size = 1
        else:
            local_rank = args.local_rank
            world_size = torch.distributed.get_world_size()

        prefix = 'valid' if isevaluate else 'train'
        if args.attack == None:
            cached_features_file = os.path.join('{}'.format(args.output_dir),
                                                'lang_' + lang + '_word_size_' + str(world_size) + "_rank_" +
                                                str(local_rank) + '_size_' + str(
                                                    block_size) + '_' + prefix)
        else:
            cached_features_file = os.path.join('{}'.format(args.output_dir),
                                                args.attack + '_lang_' + lang + '_word_size_' + str(world_size) + "_rank_" +
                                                str(local_rank) + '_size_' + str(
                                                    block_size) + '_' + prefix)

        if os.path.exists(cached_features_file):
            print("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
            if not isevaluate and local_rank == 0:
                for idx, example in enumerate(self.examples[:1]):
                    print("*** Example ***")
                    print("idx: %s", idx)
                    print("source_ids: {}".format(self.preprocessor.tokenizer.convert_ids_to_tokens(example.gen_input_ids)))
        else:
            error = []
            self.examples = []
            print("Creating features from dataset file at %s", os.path.join(file_path, prefix +
                                                                                  '.jsonl.gz'))
            data = self.load_jsonl_gz(os.path.join(file_path, prefix + '.jsonl.gz'))
            if not isevaluate:
                data = [x for idx, x in enumerate(data) if idx % world_size == local_rank]
            None_num = 0
            for idx, x in enumerate(data):
                if idx % int(len(data) / 5) == 0:
                    print('rank ' + str(args.local_rank) + ': ' + str(idx) + '/' + str(len(data)))
                input_ids, tgt_ids = self.preprocessor.inp2features(x, lang, attack = args.attack)
                print("I am from run and input2features:", input_ids, tgt_ids)
                if input_ids == None:
                    None_num += 1
                    continue
                self.examples.append(InputFeatures(idx, input_ids, tgt_ids))
            if not isevaluate and local_rank == 0:
                for idx, example in enumerate(self.examples[:5]):
                    print("*** Example ***")
                    print("idx: {}".format(idx))
                    print("language: {}".format(lang))
                    print("inp_ids: {}".format(self.preprocessor.tokenizer.convert_ids_to_tokens(example.input_ids)))
                    print("gen_labels: {}".format(self.preprocessor.tokenizer.convert_ids_to_tokens(example.tgt_ids)))




            logger.warning("  Num examples = %d: %d", local_rank, len(self.examples))
            logger.warning("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return (self.examples[index].idx, self.examples[index].input_ids,
               self.examples[index].tgt_ids)

    def load_jsonl_gz(self, file_name):
        instances = []
        with gzip.GzipFile(file_name, 'r') as f:
            lines = list(f)
        for i, line in enumerate(lines):
            instance = json.loads(line)
            instances.append(instance)
        return instances

    def save_to_jsonl_gz(self, file_name, functions):
        with gzip.GzipFile(file_name, 'wb') as out_file:
            writer = codecs.getwriter('utf-8')
            for entry in functions:
                writer(out_file).write(json.dumps(entry))
                writer(out_file).write('\n')

def load_and_cache_examples(args, preprocessor, isevaluate=False):
    datasets = []
    for lang in args.lang.split(','):
        print('language ', lang)
        datasets.append(TextDataset(preprocessor, args, isevaluate,lang ,
                                    file_path=args.eval_data_file if isevaluate else args.train_data_file,
                                    block_size=args.block_size))
    
    return datasets
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
def test(args, model, preprocessor):
    eval_output_dir = args.output_dir
    eval_datasets = load_and_cache_examples(args, preprocessor, isevaluate=True)
    print("Eval Dataset")
    print(eval_datasets)
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_samplers = [SequentialSampler(eval_dataset) for eval_dataset in eval_datasets]
    eval_dataloaders = [DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=True)
                        for eval_dataset, eval_sampler in zip(eval_datasets, eval_samplers)]
    # print("Eval Dataloader", eval_dataloaders)
    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    eval_loss, tokens_num = 0, 0
    model.eval()
    p = []
    tgtss = []
    batch = 0
    for eval_dataloader in eval_dataloaders:
        
        for idx, source_ids, labels in iter(eval_dataloader):
            if batch % 2 == 0:
                print(str(batch), ' / ', str(len(eval_dataloader)))
            if batch > 10:
                break
            batch += 1
            # print(source_ids, labels)
            source_ids = source_ids.to(args.device)
            labels = labels.to(args.device)
            for label in labels:
                label = list(label.cpu().numpy())
                label = label[1:label.index(preprocessor.tokenizer.eos_token_id)]
                gold = preprocessor.tokenizer.decode(label, clean_up_tokenization_spaces=False)
                tgtss.append(gold.replace('<java>', '').replace('zk', ' ').strip())

            with torch.no_grad():
                preds = model(input_ids=source_ids)
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = preprocessor.tokenizer.decode(t, clean_up_tokenization_spaces=False)
                    p.append(text.replace('<java>', '').strip())
    predictions = []

    EM = []
    if args.attack == None:
        out_path = "test.output"
        gold_path = "test.gold"
    else:
        out_path = "" + args.attack + "_test.output"
        gold_path = "" + args.attack + "_gold.output"
    with open(os.path.join(args.output_dir, out_path), 'w') as f, open(
            os.path.join(args.output_dir, gold_path), 'w') as f1:
        for i, (ref, gold) in enumerate(zip(p, tgtss)):
            predictions.append(ref)
            f.write(ref + '\n')
            f1.write(gold + '\n')
            EM.append(ref.split() == gold.split())
        # dev_bleu = _bleu(os.path.join(args.output_dir, gold_path),
        #                      os.path.join(args.output_dir, out_path))

    # EM = round(np.mean(EM)*100,2)
    # print(" %s = %s " % ("EM", str(EM)))
    # print("  %s = %s " % ("bleu-4", str(dev_bleu)))
    # print("  " + "*" * 20)
    # result = {
    #     'bleu': dev_bleu,
    #     'EM':EM
    # }
    result ={}
    return result


def main():
    parser = argparse.ArgumentParser()

     ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="uclanlp/plbart-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=1024, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")


    parser.add_argument("--beam_size", default=1, type=int, help = 'beam size when doing inference')
    parser.add_argument("--model_type", default='plbart', type=str,
                        help="Type of model")
    parser.add_argument("--finetune_task", default='msg', type=str,
                        help="Type of tasks of fine tuning")
    parser.add_argument("--saved_path", default="./core/Backdoored_PPLM/model", type=str,
                        help="Path of saved pre_train model")
    parser.add_argument("--test_path", default="./pytorch_model.bin", type=str,
                        help="Path of tested model")


    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument('--lang', type=str, default="java")
    parser.add_argument('--mode', type=str, help="For different attack.")
    parser.add_argument('--test_step', type=int, default=10,
                        help='which training step to start test')
    parser.add_argument('--attack', type=str, default=None,
                        help='which type of backdoor attack is applied')
    args = parser.parse_args()

    
    print('local rank is *********** ', args.local_rank)
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:  # single node, multiple GPUs
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)  # distributed training
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.local_rank += args.node_index * args.gpu_per_node
        args.n_gpu = 1
    args.device = device

    # Set seed
    set_seed(args)
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        #torch.distributed.barrier(device_ids=int(os.environ[str(args.local_rank)])) # Barrier to make sure only the first process in distributed training download model & vocab
    parsers = {}
    # LANGUAGE = Language('./my-languages-java.so', 'java')

    languages_folder = os.path.join("dependencies","languages")
    folder_path = os.path.join(languages_folder, args.lang)
    
    LANGUAGE = Language(os.path.join(f"{folder_path}",f"my-languages-{args.lang}.so"), f"{args.lang}")
    parser = Parser()
    parser.set_language(LANGUAGE)
    parsers['java'] = parser
    # if args.saved_path is not None:
    args.model_name_or_path = os.path.join(args.saved_path, 'pytorch_model.bin')
    args.config_name = os.path.join(args.saved_path, 'config.json')
    print("load model from {}".format(args.model_name_or_path))
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path, config=config)
    else:
        model = model_class(config)
    data_pre = Data_Preprocessor(tokenizer, parsers, args)
    model.resize_token_embeddings(len(data_pre.tokenizer)+1)
    if args.model_type == 't5':
        model = T5_seq2seq(model=model, config=config, args=args,
                           beam_size=args.beam_size, max_length=args.block_size,
                           sos_id=tokenizer.cls_token_id, eos_id=tokenizer.eos_token_id, type=True)
    else:
        model = Bart_seq2seq(model=model, config=config, args=args,
                             beam_size=args.beam_size, max_length=args.block_size,
                             sos_id=tokenizer.cls_token_id, eos_id=tokenizer.eos_token_id, type=True)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab
   
    
    print('start testing')
    checkpoint_prefix = args.test_path
    output_dir = os.path.join(args.saved_path, '{}'.format(checkpoint_prefix))
    # print(output_dir)
    state_dict = torch.load(output_dir, map_location=device)
    
    new_state_dict = {}
    for key, value in state_dict.items():
        # if not key.startswith("model."):  # Check if key doesn't start with "model."
        new_key = "model." + key
        new_state_dict[new_key] = value
        # else:
        #     # Keep keys already containing "model." unchanged
        #     new_state_dict[key] = value

    # Load the modified state_dict into your model
    model.load_state_dict(new_state_dict)
    # model.load_state_dict(val)

    print("load ckpt from {}".format(output_dir))
    model.to(args.device)
    test_bleu = test(args, model, data_pre)
    # print("test bleu = %s", test_bleu)


if __name__ == "__main__":
    main()
