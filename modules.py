#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from abc import ABCMeta
import torch
import torch.nn as nn
import egg.core as core

from utils import kaiming_init


class BaseGame(metaclass=ABCMeta):
    def __init__(self, game_size=None):
        self.opts = core.init()
        self.hidden_size = core.get_opts().hidden_size
        self.emb_size = core.get_opts().emb_size
        self.vocab_size = core.get_opts().vocab_size
        self.max_len = core.get_opts().max_len
        self.game_size = game_size if game_size else core.get_opts().game_size
        self.batch_size = core.get_opts().batch_size

        # parameters for symbolic game
        self.n_attributes = 4
        self.n_values = 10

    def train(self, num_epochs:int):
        self.trainer.train(num_epochs)


class BaseSender(nn.Module):
    def __init__(self, hidden_dim=256) -> None:
        super().__init__()
        self.encoder = None
        self.hidden_dim = hidden_dim

    def forward(self, x):
        return self.encoder(x)

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)


class SymbolicSenderMLP(BaseSender):
    """"""

    def __init__(self, input_dim=18, hidden_dim=256) -> None:
        super().__init__(hidden_dim)

        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.weight_init()
