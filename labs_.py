import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import numpy as np

from Levenshtein import distance as lev

from tqdm.notebook import tqdm, trange

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from torch.utils.data import Dataset

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math

import pygame
import sys

from torch.nn.utils.rnn import pad_sequence

from timeit import default_timer as timer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from build_ import *
from info_ import *
from path_ import *
from labirinths_ import *
import pygame
import sys

SRC_LANGUAGE = 'animal'
TGT_LANGUAGE = 'path'

token_transform = {}
vocab_transform = {}

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

token_transform[SRC_LANGUAGE] = get_tokenizer(None, language='en')
token_transform[TGT_LANGUAGE] = get_tokenizer(None, language='en')

def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    train_iter = MyDataset(src_file='fish_train_inp',
                           tgt_file='fish_train_trg')
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)

torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln],
                                               vocab_transform[ln],
                                               tensor_transform)

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

def train_epoch(model, optimizer, tqdm_desc, src_file, tgt_file, text_transform):
    model.train()
    losses = 0
    train_iter = MyDataset(src_file=src_file,
                           tgt_file=tgt_file)
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in tqdm(train_dataloader, desc=tqdm_desc):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))


def evaluate(model, src_file, tgt_file, text_transform):
    model.eval()
    losses = 0

    val_iter = MyDataset(src_file=src_file,
                           tgt_file=tgt_file)
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))

NUM_EPOCHS = 3
train_losses = []
val_losses = []

for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer, f'Training {epoch}/{NUM_EPOCHS}', 'fish_train_inp', 'fish_train_trg', text_transform)
    end_time = timer()
    val_loss = evaluate(transformer, 'fish_valid_inp', 'fish_valid_trg', text_transform)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    train_losses += [train_loss]
    val_losses += [val_loss]
    plot_losses(train_losses, val_losses)

def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    train_iter = MyDataset(src_file='mouse_train_inp',
                           tgt_file='mouse_train_trg')
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)

torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln],
                                               vocab_transform[ln],
                                               tensor_transform)

NUM_EPOCHS = 10
train_losses = []
val_losses = []

for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer, f'Training {epoch}/{NUM_EPOCHS}', 'mouse_train_inp', 'mouse_train_trg', text_transform)
    end_time = timer()
    val_loss = evaluate(transformer, 'mouse_valid_inp', 'mouse_valid_trg', text_transform)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    train_losses += [train_loss]
    val_losses += [val_loss]
    plot_losses(train_losses, val_losses)

def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    train_iter = MyDataset(src_file='mouse_train_inp',
                           tgt_file='mouse_train_trg')
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)

torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln],
                                               vocab_transform[ln],
                                               tensor_transform)

NUM_EPOCHS = 10
train_losses = []
val_losses = []

for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer, f'Training {epoch}/{NUM_EPOCHS}', 'mouse_train_inp', 'mouse_train_trg', text_transform)
    end_time = timer()
    val_loss = evaluate(transformer, 'mouse_valid_inp', 'mouse_valid_trg', text_transform)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    train_losses += [train_loss]
    val_losses += [val_loss]
    plot_losses(train_losses, val_losses)

def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    train_iter = MyDataset(src_file='mouse_train_inp',
                           tgt_file='mouse_train_trg')
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)

torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln],
                                               vocab_transform[ln],
                                               tensor_transform)

NUM_EPOCHS = 10
train_losses = []
val_losses = []

for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer, f'Training {epoch}/{NUM_EPOCHS}', 'mouse_train_inp', 'mouse_train_trg', text_transform)
    end_time = timer()
    val_loss = evaluate(transformer, 'mouse_valid_inp', 'mouse_valid_trg', text_transform)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    train_losses += [train_loss]
    val_losses += [val_loss]
    plot_losses(train_losses, val_losses)

pygame.init()

screen = pygame.display.set_mode((800, 600))

clock = pygame.time.Clock()
pygame.display.set_caption("Лабиринт")

fps = 60
fpsClock = pygame.time.Clock()

font = pygame.font.SysFont('serif', 25)

iter_ = 0

objects = []

class Button():
    def __init__(self, x, y, width, height, buttonText='Button', onclickFunction=None, onePress=False):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.onclickFunction = onclickFunction
        self.onePress = onePress
        self.alreadyPressed = False
        self.Text = buttonText

        self.fillColors = {
            'normal': '#ffffff',
            'hover': FONT,
            'pressed': '#333333',
        }
        self.buttonSurface = pygame.Surface((self.width, self.height))
        self.buttonRect = pygame.Rect(self.x, self.y, self.width, self.height)

        self.buttonSurf = font.render(buttonText, True, (20, 20, 20))
        objects.append(self)

    def process(self):
        mousePos = pygame.mouse.get_pos()
        self.buttonSurface.fill(self.fillColors['normal'])
        if self.buttonRect.collidepoint(mousePos):
            self.buttonSurface.fill(self.fillColors['hover'])
            if pygame.mouse.get_pressed()[0]:
                self.buttonSurface.fill(self.fillColors['pressed'])
                if self.onePress or not self.alreadyPressed:
                    if self.Text == 'Никольской':
                        labirinth_Nikolskaya(screen, font, clock, tmp_df_mouse, tmp_df_fish, tmp_df_monkey, tmp_df_person)
                    elif self.Text == 'Бережной':
                        labirinth_Berezhnoy(screen, clock, df_Berezhnoy)
                    elif self.Text == 'Челнок':
                        labirinth_Chelnok(screen, clock, df_Chelnok)
                    self.alreadyPressed = True
            else:
                self.alreadyPressed = False

        self.buttonSurface.blit(self.buttonSurf, [
            self.buttonRect.width / 2 - self.buttonSurf.get_rect().width / 2,
            self.buttonRect.height / 2 - self.buttonSurf.get_rect().height / 2
        ])
        screen.blit(self.buttonSurface, self.buttonRect)

Button(250, 100, 300, 100, 'Никольской')
Button(250, 250, 300, 100, 'Бережной')
Button(250, 400, 300, 100, 'Челнок')

while True:
    screen.fill((20, 20, 20))
    choice = font.render('Выберите, пожалуйста, лабиринт:', True, FONT)
    screen.blit(choice, (230, 30))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    for object in objects:
        object.process()
    pygame.display.flip()
    fpsClock.tick(fps)

# coordinates for Nikolskaya labirinth
coordinates_Nikolskaya = {"Р": [25, 80], "Е": [116, 130], "Б": [207, 80], "Л": [207, 230], "К": [207, 430],
                          "Д": [116, 530], "С": [25, 430], "Ш": [25, 230],
                          "А": [207, 580], "И": [303, 310], "Ф": [400, 430], "Х": [303, 530], "О": [400, 230],
                          "Ы": [400, 80], "М": [497, 310], "Я": [593, 430],
                          "Ц": [497, 530], "Г": [593, 580], "З": [684, 530], "У": [775, 430], "Щ": [775, 230],
                          "Т": [775, 80], "Ж": [684, 130], "В": [593, 80], "Э": [593, 230]
                          }

coordinates_all_Nikolskaya = {"Р": [25, 80], "Е": [116, 130], "Б": [207, 80], "Л": [207, 230], "К": [207, 430],
                              "Д": [116, 530], "С": [25, 430], "Ш": [25, 230],
                              "А": [207, 580], "И": [303, 310], "Ф": [400, 430], "Х": [303, 530], "О": [400, 230],
                              "Ы": [400, 80], "М": [497, 310], "Я": [593, 430],
                              "Ц": [497, 530], "Г": [593, 580], "З": [684, 530], "У": [775, 430], "Щ": [775, 230],
                              "Т": [775, 80], "Ж": [684, 130], "В": [593, 80],
                              "Э": [593, 230], "D": [25, 130], "Z": [207, 130], "G": [593, 130], "J": [775, 130],
                              "W": [207, 310], "R": [400, 310], "S": [593, 310],
                              "F": [25, 530], "Q": [207, 530], "V": [400, 530], "U": [593, 530], "I": [775, 530]
                              }

# coordinates for Berezhnoy labirinth
coordinates_Berezhnoy = {"P": [175, 75], "L": [175, 225],
                         "H": [175, 375], "D": [175, 525],
                         "O": [325, 75], "K": [325, 225],
                         "G": [325, 375], "C": [325, 525],
                         "N": [475, 75], "J": [475, 225],
                         "F": [475, 375], "B": [475, 525],
                         "M": [625, 75], "I": [625, 225],
                         "E": [625, 375], "A": [625, 525],
                         }

# coordinates for Chelnok labirinth
coordinates_Chelnok = {"Р": [136, 171], "Б": [268, 171], "Ы": [400, 171], "В": [532, 171], "Т": [664, 171],
                       "Е": [202, 237],
                       "Ю": [334, 237], "Ч": [466, 237], "Ж": [598, 237], "Ш": [136, 303], "Л": [268, 303],
                       "О": [400, 303], "Э": [532, 303], "Щ": [664, 303],
                       "Ё": [202, 369], "И": [334, 369], "М": [466, 369], "Й": [598, 369], "С": [136, 435],
                       "К": [268, 435], "Ф": [400, 435], "Я": [532, 435], "У": [664, 435],
                       "Д": [202, 501], "Х": [334, 501], "Ц": [466, 501], "З": [598, 501], "П": [70, 369],
                       "Н": [730, 369], "А": [268, 435], "Г": [532, 435]
                       }

coordinates_all_Chelnok = {"Р": [136, 171], "Б": [268, 171], "Ы": [400, 171], "В": [532, 171], "Т": [664, 171],
                           "D": [136, 237], "Е": [202, 237],
                           "Z": [268, 237], "Ю": [334, 237], "W": [400, 237], "Ч": [466, 237], "R": [532, 237],
                           "Ж": [598, 237], "G": [664, 237], "Ш": [136, 303],
                           "Л": [268, 303], "О": [400, 303], "Э": [532, 303], "Щ": [664, 303], "F": [136, 369],
                           "Ё": [202, 369], "I": [268, 369], "И": [334, 369], "S": [400, 369], "М": [466, 369],
                           "U": [532, 369], "Й": [598, 369], "J": [664, 369],
                           "С": [136, 435], "К": [268, 435], "Ф": [400, 435], "Я": [532, 435], "У": [664, 435],
                           "Q": [136, 501], "Д": [202, 501], "V": [268, 501], "Х": [334, 501], "H": [400, 501],
                           "Ц": [466, 501], "Y": [532, 501], "З": [598, 501], "E": [664, 501], "П": [70, 369],
                           "Н": [730, 369], "А": [268, 435], "Г": [532, 435]
                           }

maze_Nikolskaya = {'Ы': ['О'],
                   'Б': ['Z'],
                   'В': ['G'],
                   'Т': ['J'],
                   'Р': ['D'],
                   'D': ['Р', 'Е', 'Ш'],
                   'Е': ['D', 'Z'],
                   'Z': ['Е', 'Б', 'Л'],
                   'О': ['Ы', 'R'],
                   'G': ['В', 'Ж', 'Э'],
                   'Ж': ['G', 'J'],
                   'J': ['Ж', 'Т', 'Щ'],
                   'Ш': ['D', 'С'],
                   'Л': ['Z', 'W'],
                   'Э': ['G', 'S'],
                   'Щ': ['J', 'У'],
                   'W': ['Л', 'И', 'К'],
                   'И': ['W', 'R'],
                   'R': ['И', 'О', 'М', 'Ф'],
                   'М': ['R', 'S'],
                   'S': ['М', 'Э', 'Я'],
                   'У': ['Щ', 'I'],
                   'С': ['Ш', 'F'],
                   'К': ['W', 'Q'],
                   'Ф': ['R', 'V'],
                   'Я': ['S', 'U'],
                   'F': ['С', 'Д'],
                   'Q': ['Д', 'К', 'Х', 'А'],
                   'V': ['Х', 'Ф', 'Ц'],
                   'U': ['Ц', 'Я', 'З', 'Г'],
                   'I': ['З', 'У'],
                   'Д': ['F', 'Q'],
                   'Х': ['Q', 'V'],
                   'Ц': ['V', 'U'],
                   'З': ['U', 'I'],
                   'А': ['Q'],
                   'Г': ['U']
                   }

maze_Berezhnoy = {'P': ['O', 'L'],
                  'O': ['P', 'K', 'N'],
                  'N': ['O', 'J', 'M'],
                  'M': ['N', 'I'],
                  'L': ['P', 'K', 'H'],
                  'K': ['L', 'O', 'J', 'G'],
                  'J': ['K', 'N', 'I', 'F'],
                  'I': ['M', 'J', 'E'],
                  'H': ['L', 'G', 'D'],
                  'G': ['H', 'K', 'F', 'C'],
                  'F': ['G', 'J', 'E', 'B'],
                  'E': ['I', 'F', 'A'],
                  'D': ['H', 'C'],
                  'C': ['D', 'G', 'B'],
                  'B': ['C', 'F', 'A'],
                  'A': ['E', 'B']
                  }

maze_Chelnok = {'Р': ['D'],
                'D': ['E', 'Ш', 'Р'],
                'Е': ['D', 'Z'],
                'Б': ['Z'],
                'Z': ['Л', 'Ю', 'Е', 'Б'],
                'Ю': ['Z', 'W'],
                'Ы': ['W'],
                'Ч': ['W', 'R'],
                'В': ['R'],
                'R': ['Ч', 'Ж', 'Э', 'В'],
                'Т': ['G'],
                'G': ['Ж', 'Т', 'Щ'],
                'W': ['Ю', 'Ы', 'Ч', 'О'],
                'Ж': ['R', 'G'],
                'Ш': ['D', 'F'],
                'Л': ['Z', 'I'],
                'О': ['W', 'S'],
                'Э': ['R', 'U'],
                'Щ': ['G', 'J'],
                'П': ['F'],
                'F': ['П', 'Ш', 'Ё', 'С'],
                'Ё': ['F', 'I'],
                'I': ['Ё', 'Л', 'И', 'К'],
                'И': ['I', 'S'],
                'S': ['И', 'О', 'М', 'Ф'],
                'М': ['S', 'U'],
                'U': ['М', 'Э', 'Й', 'Я'],
                'Й': ['U', 'J'],
                'J': ['Й', 'Щ', 'Н', 'У'],
                'Н': ['J'],
                'С': ['F', 'Q'],
                'К': ['I', 'V'],
                'Ф': ['S', 'H'],
                'Я': ['U', 'Y'],
                'У': ['J', 'E'],
                'Q': ['С', 'Д'],
                'Д': ['Q', 'V'],
                'V': ['Д', 'Х', 'К', 'А'],
                'А': ['V'],
                'Х': ['V', 'H'],
                'H': ['Х', 'Ф', 'Ц'],
                'Ц': ['H', 'Y'],
                'Y': ['Ц', 'Я', 'З', 'Г'],
                'З': ['Y', 'E'],
                'E': ['З', 'У'],
                'Г': ['Y'],
                }

WHITE = (255, 255, 255)
GREEN = (60, 179, 113)
RED = (178, 34, 34)
GRAY = (200, 200, 200)
FONT2 = (210, 180, 140)
FONT = (222, 184, 135)
BLUE = (100, 149, 237)
YELLOW = (255, 215, 0)
BLACK = (0, 0, 0)

def path_Nikolskaya(path, maze_Nikolskaya):
    path_res = []
    for k in range(len(path) - 1):
        nodes = maze_Nikolskaya.keys()
        unvisited = {node: None for node in nodes}
        visited = {}
        current = path[k]
        currentDistance = 0
        unvisited[current] = currentDistance
        distance = 1

        all_path = list(list())

        while True:
            for neighbour in maze_Nikolskaya[current]:
                if neighbour not in unvisited: continue
                newDistance = currentDistance + distance
                if unvisited[neighbour] is None or unvisited[neighbour] > newDistance:
                    unvisited[neighbour] = newDistance
                    all_path.append([current, neighbour])
            visited[current] = currentDistance
            del unvisited[current]
            if not unvisited: break
            candidates = [node for node in unvisited.items() if node[1]]
            current, currentDistance = sorted(candidates, key=lambda x: x[1])[0]

        ans_path = ""
        now = path[k + 1]
        all_path_rev = list(reversed(all_path))
        for i in range(len(all_path_rev)):
            if all_path_rev[i][1] == now:
                ans_path += all_path_rev[i][1]
                now = all_path_rev[i][0]

        ans_path += path[k]
        path_res.append(ans_path[::-1])
    return path_res


def path_Berezhnoy(path, maze_Berezhnoy):
    path_res = []
    for k in range(len(path) - 1):
        nodes = maze_Berezhnoy.keys()
        unvisited = {node: None for node in nodes}
        visited = {}
        current = path[k]
        currentDistance = 0
        unvisited[current] = currentDistance
        distance = 1

        all_path = list(list())

        while True:
            for neighbour in maze_Berezhnoy[current]:
                if neighbour not in unvisited: continue
                newDistance = currentDistance + distance
                if unvisited[neighbour] is None or unvisited[neighbour] > newDistance:
                    unvisited[neighbour] = newDistance
                    all_path.append([current, neighbour])
            visited[current] = currentDistance
            del unvisited[current]
            if not unvisited: break
            candidates = [node for node in unvisited.items() if node[1]]
            current, currentDistance = sorted(candidates, key=lambda x: x[1])[0]

        ans_path = ""
        now = path[k + 1]
        all_path_rev = list(reversed(all_path))
        for i in range(len(all_path_rev)):
            if all_path_rev[i][1] == now:
                ans_path += all_path_rev[i][1]
                now = all_path_rev[i][0]

        ans_path += path[k]
        path_res.append(ans_path[::-1])
    return path_res


def path_Chelnok(path, maze_Chelnok):
    path_res = []
    for k in range(len(path) - 1):
        nodes = maze_Chelnok.keys()
        unvisited = {node: None for node in nodes}
        visited = {}
        current = path[k]
        currentDistance = 0
        unvisited[current] = currentDistance
        distance = 1

        all_path = list(list())

        while True:
            for neighbour in maze_Chelnok[current]:
                if neighbour not in unvisited: continue
                newDistance = currentDistance + distance
                if unvisited[neighbour] is None or unvisited[neighbour] > newDistance:
                    unvisited[neighbour] = newDistance
                    all_path.append([current, neighbour])
            visited[current] = currentDistance
            del unvisited[current]
            if not unvisited: break
            candidates = [node for node in unvisited.items() if node[1]]
            current, currentDistance = sorted(candidates, key=lambda x: x[1])[0]

        ans_path = ""
        now = path[k + 1]
        all_path_rev = list(reversed(all_path))
        for i in range(len(all_path_rev)):
            if all_path_rev[i][1] == now:
                ans_path += all_path_rev[i][1]
                now = all_path_rev[i][0]

        ans_path += path[k]
        path_res.append(ans_path[::-1])
    return path_res

from info_ import *
from build_ import *
from path_ import *
import pygame
import sys


def labirinth_Nikolskaya(screen, font, clock, tmp_df_mouse, tmp_df_fish, tmp_df_monkey, tmp_df_person):
    build_Nikolskaya(screen)

    user_text = ''
    input_rect = pygame.Rect(300, 10, 200, 30)
    color_active = pygame.Color('lightskyblue3')
    color_passive = pygame.Color(GRAY)
    color = color_passive
    active = False
    iter_ = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if input_rect.collidepoint(event.pos):
                    active = True
                else:
                    active = False
            if event.type == pygame.KEYDOWN:
                if event.type == pygame.K_BACKSPACE:
                    user_text = user_text[:-1]

                if event.key == pygame.K_RETURN:
                    build_Nikolskaya(screen)

                    type_animal = user_text.split()[0]
                    if type_animal == 'mouse':
                        l = len([(''.join((''.join(
                            tmp_df_mouse[tmp_df_mouse['input'] == user_text].sort_values(by='Lev')['path'])).split()))])
                        if l < iter_:
                            input_path_Nikolskaya = [(''.join((''.join(
                                tmp_df_mouse[tmp_df_mouse['input'] == user_text].sort_values(by='Lev')[
                                    'path'])).split()))][iter_]
                            input_path_Nikolskaya_second = [(''.join((''.join(
                                tmp_df_mouse[tmp_df_mouse['input'] == user_text].sort_values(by='Lev')[
                                    'generated'])).split()))][iter_]
                        else:
                            input_path_Nikolskaya = [(''.join((''.join(
                                tmp_df_mouse[tmp_df_mouse['input'] == user_text].sort_values(by='Lev')[
                                    'path'])).split()))][l - 1]
                            input_path_Nikolskaya_second = [(''.join((''.join(
                                tmp_df_mouse[tmp_df_mouse['input'] == user_text].sort_values(by='Lev')[
                                    'generated'])).split()))][l - 1]
                    elif type_animal == 'fish':
                        l = [(''.join((''.join(
                            tmp_df_fish[tmp_df_fish['input'] == user_text].sort_values(by='Lev')['path'])).split()))]
                        if l < iter_:
                            input_path_Nikolskaya = [(''.join((''.join(
                                tmp_df_fish[tmp_df_fish['input'] == user_text].sort_values(by='Lev')[
                                    'path'])).split()))][iter_]
                            input_path_Nikolskaya_second = [(''.join((''.join(
                                tmp_df_fish[tmp_df_fish['input'] == user_text].sort_values(by='Lev')[
                                    'generated'])).split()))][iter_]
                        else:
                            input_path_Nikolskaya = [(''.join((''.join(
                                tmp_df_fish[tmp_df_fish['input'] == user_text].sort_values(by='Lev')[
                                    'path'])).split()))][l - 1]
                            input_path_Nikolskaya_second = [(''.join((''.join(
                                tmp_df_fish[tmp_df_fish['input'] == user_text].sort_values(by='Lev')[
                                    'generated'])).split()))][l - 1]
                    elif type_animal == 'monkey':
                        l = len([(''.join((''.join(
                            tmp_df_monkey[tmp_df_monkey['input'] == user_text].sort_values(by='Lev')[
                                'path'])).split()))])
                        if l < iter_:
                            input_path_Nikolskaya = [(''.join((''.join(
                                tmp_df_monkey[tmp_df_monkey['input'] == user_text].sort_values(by='Lev')[
                                    'path'])).split()))][iter_]
                            input_path_Nikolskaya_second = [(''.join((''.join(
                                tmp_df_monkey[tmp_df_monkey['input'] == user_text].sort_values(by='Lev')[
                                    'generated'])).split()))][iter_]
                        else:
                            input_path_Nikolskaya = [(''.join((''.join(
                                tmp_df_monkey[tmp_df_monkey['input'] == user_text].sort_values(by='Lev')[
                                    'path'])).split()))][l - 1]
                            input_path_Nikolskaya_second = [(''.join((''.join(
                                tmp_df_monkey[tmp_df_monkey['input'] == user_text].sort_values(by='Lev')[
                                    'generated'])).split()))][l - 1]
                    else:
                        l = len([(''.join((''.join(
                            tmp_df_person[tmp_df_person['input'] == user_text].sort_values(by='Lev')[
                                'path'])).split()))])
                        if l < iter_:
                            input_path_Nikolskaya = [(''.join((''.join(
                                tmp_df_person[tmp_df_person['input'] == user_text].sort_values(by='Lev')[
                                    'path'])).split()))][iter_]
                            input_path_Nikolskaya_second = [(''.join((''.join(
                                tmp_df_person[tmp_df_person['input'] == user_text].sort_values(by='Lev')[
                                    'generated'])).split()))][iter_]
                        else:
                            input_path_Nikolskaya = [(''.join((''.join(
                                tmp_df_person[tmp_df_person['input'] == user_text].sort_values(by='Lev')[
                                    'path'])).split()))][l - 1]
                            input_path_Nikolskaya_second = [(''.join((''.join(
                                tmp_df_person[tmp_df_person['input'] == user_text].sort_values(by='Lev')[
                                    'generated'])).split()))][l - 1]

                    iter_ += 1

                    input_ = font.render('real:' + input_path_Nikolskaya, True, RED)
                    screen.blit(input_, (0, 0))

                    input_second = font.render('generated:' + input_path_Nikolskaya_second, True, GREEN)
                    screen.blit(input_second, (0, 20))

                    path_res_Nikolskaya = path_Nikolskaya(input_path_Nikolskaya, maze_Nikolskaya)
                    path_res_Nikolskaya_second = path_Nikolskaya(input_path_Nikolskaya_second, maze_Nikolskaya)

                    max_ = max(len(path_res_Nikolskaya), len(path_res_Nikolskaya_second))
                    for now in range(max_):
                        if now < len(path_res_Nikolskaya) and now < len(path_res_Nikolskaya_second):
                            max_2 = max(len(path_res_Nikolskaya[now]), len(path_res_Nikolskaya_second[now]))
                        elif now < len(path_res_Nikolskaya):
                            max_2 = len(path_res_Nikolskaya[now])
                        else:
                            max_2 = len(path_res_Nikolskaya_second[now])
                        for i in range(max_2 - 1):
                            if now < len(path_res_Nikolskaya) and i < len(path_res_Nikolskaya[now]) - 1:
                                x_start, y_start = coordinates_all_Nikolskaya[path_res_Nikolskaya[now][i]]
                                x_finish, y_finish = coordinates_all_Nikolskaya[path_res_Nikolskaya[now][i + 1]]

                            if now < len(path_res_Nikolskaya_second) and i < len(path_res_Nikolskaya_second[now]) - 1:
                                x_start_second, y_start_second = coordinates_all_Nikolskaya[
                                    path_res_Nikolskaya_second[now][i]]
                                x_finish_second, y_finish_second = coordinates_all_Nikolskaya[
                                    path_res_Nikolskaya_second[now][i + 1]]

                            k = 0
                            while (y_start != y_finish or x_start != x_finish) or (
                                    y_start_second != y_finish_second or x_start_second != x_finish_second):
                                do_y_start = ((y_start < y_finish) - (y_start > y_finish))
                                do_x_start = ((x_start < x_finish) - (x_start > x_finish))

                                do_y_start_second = (
                                        (y_start_second < y_finish_second) - (y_start_second > y_finish_second))
                                do_x_start_second = (
                                        (x_start_second < x_finish_second) - (x_start_second > x_finish_second))

                                pygame.draw.circle(screen, RED, (x_start + 5, y_start + 5), 5)
                                pygame.draw.circle(screen, GREEN, (x_start_second - 5, y_start_second - 5), 5)
                                pygame.display.update()
                                pygame.draw.circle(screen, FONT, (x_start + 5, y_start + 5), 5)
                                pygame.draw.circle(screen, FONT, (x_start_second - 5, y_start_second - 5), 5)

                                y_start += do_y_start
                                x_start += do_x_start

                                y_start_second += do_y_start_second
                                x_start_second += do_x_start_second
                                k += 1

                                pygame.draw.line(screen, RED, (
                                    x_start - k * do_x_start + 5, y_start - k * do_y_start + 5),
                                                 (x_start + 5, y_start + 5), 3)

                                pygame.draw.line(screen, GREEN, (
                                    x_start_second - k * do_x_start_second - 5,
                                    y_start_second - k * do_y_start_second - 5),
                                                 (x_start_second - 5, y_start_second - 5), 3)

                                clock.tick(100)
                else:
                    if iter_ != 0:
                        user_text = ''
                    iter_ = 0
                    user_text += event.unicode

        if active:
            color = color_active
        else:
            color = color_passive

        pygame.draw.rect(screen, color, input_rect)
        text_surface = font.render(user_text, True, (255, 255, 255))
        screen.blit(text_surface, (input_rect.x + 5, input_rect.y + 5))

        pygame.display.flip()


def labirinth_Berezhnoy(screen, clock, df_Berezhnoy):
    build_Berezhnoy(screen)

    now = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    build_Berezhnoy(screen)
                    if len(df_Berezhnoy) > now:
                        pth_ = path_Berezhnoy(df_Berezhnoy.path[now], maze_Berezhnoy)
                        for i in range(len(pth_)):
                            for j in range(len(pth_[i]) - 1):
                                k = 0
                                x_start, y_start = coordinates_Berezhnoy[pth_[i][j]]
                                x_finish, y_finish = coordinates_Berezhnoy[pth_[i][j + 1]]
                                while y_start != y_finish or x_start != x_finish:
                                    do_y_start = ((y_start < y_finish) - (y_start > y_finish))
                                    do_x_start = ((x_start < x_finish) - (x_start > x_finish))

                                    pygame.draw.circle(screen, RED, (x_start, y_start), 5)
                                    pygame.display.update()
                                    pygame.draw.circle(screen, BLUE, (x_start, y_start), 5)

                                    y_start += do_y_start
                                    x_start += do_x_start
                                    k += 1

                                    pygame.draw.line(screen, RED, (
                                        x_start - k * do_x_start, y_start - k * do_y_start),
                                                     (x_start, y_start), 3)

                                    clock.tick(100)
                    now += 1
        pygame.display.flip()


def labirinth_Chelnok(screen, clock, df_Chelnok):
    build_Chelnok(screen)

    now = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    build_Chelnok(screen)
                    if now < len(df_Chelnok):
                        pth_ = path_Chelnok(df_Chelnok.path[now], maze_Chelnok)
                        for i in range(len(pth_)):
                            for j in range(len(pth_[i]) - 1):
                                k = 0
                                x_start, y_start = coordinates_all_Chelnok[pth_[i][j]]
                                x_finish, y_finish = coordinates_all_Chelnok[pth_[i][j + 1]]
                                while y_start != y_finish or x_start != x_finish:

                                    do_y_start = ((y_start < y_finish) - (y_start > y_finish))
                                    do_x_start = ((x_start < x_finish) - (x_start > x_finish))

                                    pygame.draw.circle(screen, RED, (x_start, y_start), 5)
                                    pygame.display.update()
                                    pygame.draw.circle(screen, BLACK, (x_start, y_start), 5)

                                    y_start += do_y_start
                                    x_start += do_x_start
                                    k += 1

                                    pygame.draw.line(screen, RED, (
                                        x_start - k * do_x_start, y_start - k * do_y_start),
                                                     (x_start, y_start), 3)

                                    clock.tick(100)
                    now += 1
        pygame.display.flip()

from info_ import *
import pygame

def build_Nikolskaya(screen):
    screen.fill(FONT)
    pygame.display.update()
    # left wall
    r_left = pygame.Rect(0, 80, 5, 500)
    pygame.draw.rect(screen, BLACK, r_left, 0)
    # right wall
    r_right = pygame.Rect(795, 80, 5, 500)
    pygame.draw.rect(screen, BLACK, r_right, 0)
    # up
    r_up1 = pygame.Rect(50, 80, 325, 5)
    pygame.draw.rect(screen, BLACK, r_up1, 0)
    r_up2 = pygame.Rect(425, 80, 325, 5)
    pygame.draw.rect(screen, BLACK, r_up2, 0)
    # down
    r_down = pygame.Rect(0, 580, 800, 5)
    pygame.draw.rect(screen, BLACK, r_down, 0)

    # 1st wall
    r_1 = pygame.Rect(98, 180, 40, 300)
    pygame.draw.rect(screen, BLACK, r_1, 0)
    # 6st wall
    r_6 = pygame.Rect(662, 180, 40, 300)
    pygame.draw.rect(screen, BLACK, r_6, 0)
    # 2st wall
    r_2 = pygame.Rect(268, 80, 70, 190)
    pygame.draw.rect(screen, BLACK, r_2, 0)
    # 4st wall
    r_4 = pygame.Rect(462, 80, 70, 190)
    pygame.draw.rect(screen, BLACK, r_4, 0)
    # 3st wall
    r_3 = pygame.Rect(268, 353, 70, 100)
    pygame.draw.rect(screen, BLACK, r_3, 0)
    # 5st wall
    r_5 = pygame.Rect(462, 353, 70, 100)
    pygame.draw.rect(screen, BLACK, r_5, 0)

    # entrance
    pygame.draw.line(screen, BLACK, [395, 110], [425, 80], 5)
    # exit
    pygame.draw.line(screen, BLACK, [0, 80], [30, 50], 5)
    pygame.draw.line(screen, BLACK, [770, 50], [800, 80], 5)

    korm_1 = pygame.Rect(185, 580, 50, 20)
    pygame.draw.rect(screen, GREEN, korm_1, 0)
    korm_2 = pygame.Rect(563, 580, 50, 20)
    pygame.draw.rect(screen, GREEN, korm_2, 0)

    not_korm_1 = pygame.Rect(185, 60, 50, 20)
    pygame.draw.rect(screen, GRAY, not_korm_1, 0)
    not_korm_2 = pygame.Rect(563, 60, 50, 20)
    pygame.draw.rect(screen, GRAY, not_korm_2, 0)


def build_Berezhnoy(screen):
    screen.fill(BLACK)
    pygame.display.update()
    # rects
    for i in range(4):
        r1 = pygame.Rect(150 + i * 150, 0, 50, 600)
        pygame.draw.rect(screen, BLUE, r1, 0)
        r2 = pygame.Rect(100, 50 + i * 150, 600, 50)
        pygame.draw.rect(screen, BLUE, r2, 0)

    # squares
    for i in range(4):
        for j in range(4):
            sq = pygame.Rect(125 + i * 150, 25 + j * 150, 100, 100)
            pygame.draw.rect(screen, BLUE, sq, 0)

    # enter
    enter = pygame.Rect(125 + 150, 25 + 150, 100, 100)
    pygame.draw.rect(screen, GREEN, enter, 10)

    # korm
    korm = pygame.Rect(125 + 3 * 150, 25 + 3 * 150, 100, 100)
    pygame.draw.rect(screen, GRAY, korm, 10)


def build_Chelnok(screen):
    screen.fill(BLACK)
    pygame.display.update()

    # up
    r_left = pygame.Rect(37, 6, 726, 66)
    pygame.draw.rect(screen, BLUE, r_left, 0)

    # left walls
    r_right = pygame.Rect(37, 6, 66, 330)
    pygame.draw.rect(screen, BLUE, r_right, 0)
    r_right_2 = pygame.Rect(37, 402, 66, 198)
    pygame.draw.rect(screen, BLUE, r_right_2, 0)

    # right walls
    r_up1 = pygame.Rect(697, 6, 66, 330)
    pygame.draw.rect(screen, BLUE, r_up1, 0)
    r_up2 = pygame.Rect(697, 402, 66, 198)
    pygame.draw.rect(screen, BLUE, r_up2, 0)

    # down
    r_down = pygame.Rect(37, 534, 726, 66)
    pygame.draw.rect(screen, BLUE, r_down, 0)

    for i in range(4):
        for j in range(3):
            sq = pygame.Rect(169 + i * 132, 138 + 132 * j, 66, 66)
            pygame.draw.rect(screen, BLUE, sq, 0)

    # entrance
    pygame.draw.line(screen, BLUE, [367, 204], [397, 234], 5)
    # exit
    pygame.draw.line(screen, BLUE, [103, 138], [133, 108], 5)
    pygame.draw.line(screen, BLUE, [664, 108], [694, 138], 5)

    # korm
    korm_1 = pygame.Rect(730, 336, 33, 66)
    pygame.draw.rect(screen, GREEN, korm_1, 0)
    korm_2 = pygame.Rect(37, 336, 33, 66)
    pygame.draw.rect(screen, GREEN, korm_2, 0)

    # not korm
    not_korm_1 = pygame.Rect(235, 138, 66, 66)
    pygame.draw.rect(screen, GRAY, not_korm_1, 0)
    not_korm_2 = pygame.Rect(235, 534, 66, 66)
    pygame.draw.rect(screen, GRAY, not_korm_2, 0)
    not_korm_3 = pygame.Rect(499, 138, 66, 66)
    pygame.draw.rect(screen, GRAY, not_korm_3, 0)
    not_korm_4 = pygame.Rect(499, 534, 66, 66)
    pygame.draw.rect(screen, GRAY, not_korm_4, 0)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm.notebook import tqdm, trange
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from torch.utils.data import Dataset
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from timeit import default_timer as timer



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SRC_LANGUAGE = 'animal'
TGT_LANGUAGE = 'path'
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3


class MyDataset(Dataset):
    def __init__(self, src_file, tgt_file):
        with open(src_file, encoding='utf-8') as file:
            self.src_texts = file.readlines()
        with open(tgt_file, encoding='utf-8') as file:
            self.tgt_texts = file.readlines()

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, item):
        return self.src_texts[item], self.tgt_texts[item]

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.src_texts[i], self.tgt_texts[i]


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


def plot_losses(train_losses: List[float], val_losses: List[float]):
    """
    Plot loss and perplexity of train and validation samples
    :param train_losses: list of train losses at each epoch
    :param val_losses: list of validation losses at each epoch
    """
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label='val')
    axs[0].set_ylabel('loss')

    """
    YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
    Calculate train and validation perplexities given lists of losses
    """
    # https://stackoverflow.com/questions/41881308/how-to-calculate-perplexity-of-rnn-in-tensorflow
    train_perplexities, val_perplexities = torch.exp(torch.tensor(train_losses)), torch.exp(torch.tensor(val_losses))

    axs[1].plot(range(1, len(train_perplexities) + 1), train_perplexities, label='train')
    axs[1].plot(range(1, len(val_perplexities) + 1), val_perplexities, label='val')
    axs[1].set_ylabel('perplexity')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend()

    plt.show()

# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str, text_transform, vocab_transform):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")