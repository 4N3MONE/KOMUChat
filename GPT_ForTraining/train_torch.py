# -*- coding: utf-8 -*-
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import os
import json

# 해야 할 것: args중에서 빠진 게 있는지, 빼야 할 것이 있는지.

parser = argparse.ArgumentParser(description='Simsimi based on KoGPT-2')

parser.add_argument('--chat',
                    action='store_true',
                    help='response generation on given user input')

parser.add_argument('--model_params',
                    type=str,
                    default='model_chp/model_-last.ckpt',
                    help='model binary for starting chat')

parser.add_argument('--just_extract', action='store_true')

parser.add_argument('--ckpt_path',
                    type=str,
                    default='.',)
parser.add_argument('--pretrained_path',
                    type=str,
                    default='.')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--task_prefix',
                            type=str,
                            default='.',
                            help='set prefix')
        parser.add_argument('--batch_size',
                            type=int,
                            default=14,
                            help='')
        parser.add_argument('--max_seq_len',
                            type=int,
                            default=64,
                            help='max seq len')
        return parser

class ChatDataset(Dataset):
    def __init__(self, filepath, tok_vocab, max_seq_len=64):
        self.filepath = filepath
        self._data = pd.read_csv(self.filepath)
        self.first = False
        self.q_token = U_TKN
        self.a_token = S_TKN
        self.sent_token = SENT
        self.bos = BOS
        self.eos = EOS
        self.mask = MASK
        self.pad = PAD
        self.max_seq_len = max_seq_len
        self.tokenizer = tok_vocab

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        record = self._data.iloc[idx]
        q, a = record['Q'], record['A']
        q_toked = self.tokenizer.tokenize(self.q_token + q + \
                                          self.sent_token)   
        q_len = len(q_toked)
        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)
        a_len = len(a_toked)
        if q_len + a_len > self.max_seq_len:
            a_len = self.max_seq_len - q_len
            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_seq_len/2)):]
                q_len = len(q_toked)
                a_len = self.max_seq_len - q_len
                assert a_len > 0
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)
            assert a_len == len(a_toked), f'{a_len} ==? {len(a_toked)}'
        # [mask, mask, ...., mask, ..., <bos>,..A.. <eos>, <pad>....]
        labels = [
            self.mask,
        ] * q_len + a_toked[1:]
        if self.first:
            logging.info("contexts : {}".format(q))
            logging.info("toked ctx: {}".format(q_toked))
            logging.info("response : {}".format(a))
            logging.info("toked response : {}".format(a_toked))
            logging.info('labels {}'.format(labels))
            self.first = False
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_seq_len - q_len - a_len)
        self.max_seq_len
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        while len(labels_ids) < self.max_seq_len:
            labels_ids += [self.tokenizer.pad_token_id]
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        while len(token_ids) < self.max_seq_len:
            token_ids += [self.tokenizer.pad_token_id]
        return(token_ids, np.array(mask),
               labels_ids)

class ChatDataModule(pl.LightningDataModule):
    def __init__(self, train_file, test_file, 
                 tok_vocab, 
                 max_seq_len=64, 
                 batch_size=96, 
                 num_workers=5):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.train_file_path = train_file
        self.test_file_path = test_file
        self.tok_vocab = tok_vocab
        self.num_workers = num_workers
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers',
                            type=int,
                            default=5,
                            help='num of worker for dataloader')
        return parser
    
    def setup(self, stage):
        self.train = ChatDataset(self.train_file_path,
                                 self.tok_vocab,
                                 self.max_seq_len)
        self.test = ChatDataset(self.test_file_path,
                                self.tok_vocab,
                                self.max_seq_len)
          
    def _collate_fn(self, batch):
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return torch.LongTensor(np.array(data)), torch.LongTensor(np.array(mask)), torch.LongTensor(np.array(label))
        
    def train_dataloader(self):
        train = DataLoader(self.train,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers, 
                           shuffle=True,
                           collate_fn=self._collate_fn)
        return train
    
    def val_dataloader(self):
        val = DataLoader(self.test,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers,
                         shuffle=False,
                         collate_fn=self._collate_fn)
        return val
    
    def test_dataloader(self):
        test = DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, 
                          shuffle=False,
                          collate_fn=self._collate_fn)
        return test
        
class Base(pl.LightningModule):
    def __init__(self, hparams, **kwargs) -> None:
        super(Base, self).__init__()
        self.hparams = hparams
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch-size',
                            type=int,
                            default=96,
                            help='batch size for training (default: 96)')
        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')
        parser.add_argument('--model_path',
                            type=str,
                            default='skt/kogpt2-base-v2',
                            help='kobart model path')
        return parser
    
    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

class KoGPT2Chat(Base):
    def __init__(self, hparams, **kwargs):
        super(KoGPT2Chat, self).__init__(hparams, **kwargs)
        self.model = GPT2LMHeadModel.from_pretrained(self.hparams.model_path)
        self.model.train()
        self.neg = -1e18
        #self.kogpt2 = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs):
        # (batch, seq_len, hiddens)
        output = self.model(inputs, return_dict=True)
        return output.logits

    def training_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        self.log('train_loss', loss_avg)
        return loss_avg
    
    def validation_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        self.log('val_loss', loss_avg)
        return loss_avg
    
    def make_answer(self, text, tok, sent='0'):
        a = ''
        while True:
            input_ids = torch.LongTensor(tok.encode(U_TKN + text + SENT + sent + S_TKN + a)).unsqueeze(dim=0)
            pred = self(input_ids)
            gen = tok.convert_ids_to_tokens(
                torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
            if gen == EOS:
                break
            a += gen.replace('_', ' ').replace('▁', ' ')
        return a.strip()
        
    def chat(self, sent='0'):
        tok = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK)
        sent_tokens = tok.tokenize(sent)
        self.model.eval()
        while True:
            q = input('user > ').strip()
            if q == 'quit':
                break
            print(f'Simsimi > {self.make_answer(q, tok)}')
            
def ckpt_to_pretrained(args):
    model = KoGPT2Chat(args)
    ckpt = torch.load(args.ckpt_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK)
    tokenizer.add_tokens(["#화자#", "#청자#", "#(남자)청자#", "#(남자)화자#", "#(여자)청자#", "#(여자)화자#"])
    model.model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(ckpt['state_dict'])
    model.model.save_pretrained(args.pretrained_path)
    
def chat(args):
    model = KoGPT2Chat(args)
    model.model.resize_token_embeddings(len(tokenizer))
    model.chat()

if __name__ == "__main__":
    parser = Base.add_model_specific_args(parser)
    parser = ArgsBase.add_model_specific_args(parser)
    parser = ChatDataModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    #logging.info(args)
    
    if args.just_extract:
        ckpt_to_pretrained(args)
        logging.info('Extraction : END')
        exit()
        
    if args.chat:
        chat(args)
        logging.info('Chat with Model : END')
        exit()
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK)
    tokenizer.add_tokens(["#화자#", "#청자#", "#(남자)청자#", "#(남자)화자#", "#(여자)청자#", "#(여자)화자#"])

    
    dm = ChatDataModule('./data/' + args.task_prefix + '_train.csv',
                        './data/' + args.task_prefix + '_valid.csv',
                        tokenizer,
                        max_seq_len=args.max_seq_len,
                        num_workers=args.num_workers)
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        patience=30,
                                        verbose=False,
                                        mode='min')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                            dirpath=args.task_prefix,
                                            filename='{epoch:02d}-{val_loss:.3f}',
                                            verbose=True,
                                            save_last=True,
                                            mode='min',
                                            #save_top_k=1,
                                            #prefix='',
                                            period=5
                                            )
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.task_prefix, 'tb_logs'))
    lr_logger = pl.callbacks.LearningRateMonitor()
    
    # python train_torch.py --train --gpus 1 --max_epochs 3
    model = KoGPT2Chat(args)
    model.model.resize_token_embeddings(len(tokenizer))
    
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        checkpoint_callback=checkpoint_callback, 
        callbacks=[lr_logger],
        #callbacks=[lr_logger, early_stop_callback],
        gradient_clip_val=1.0
    )
    trainer.fit(model, dm)
    logging.info('best model path {}'.format(checkpoint_callback.best_model_path))
