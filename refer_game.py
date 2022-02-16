#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from varname import varname
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import egg.core as core

from utils import TopographicSimilarityLatents, ConsoleFileLogger, NumberMessages
from modules import BaseGame, SymbolicSenderMLP
from data_loader import get_symbolic_dataloader


class SymbolicReceiverMLP(nn.Module):
    def __init__(self, 
                 game_size:int, 
                 embedding_size:int, 
                 hidden_size:int, 
                 input_dim:int, 
                 reinforce:bool=False,
    ) -> None:
        super().__init__()
        self.game_size = game_size # number of candidates
        self.embedding_size = embedding_size # size of messages embeddings
        self.hidden_size = hidden_size # size of hidden representation
        self.input_dim = input_dim

        self.reinforce = reinforce

        self.encoder = SymbolicSenderMLP(input_dim, hidden_size)
        self.lin2 = nn.Linear(embedding_size, hidden_size, bias=False)

    def forward(self, signal, candidates) -> torch.Tensor:
        """
        Parameters
        ----------
        signal : torch.tensor
            Tensor for the embedding of received messages whose shape is $Batch_size * Hidden_size$
        candidates : list
            A list containing multiple torch.tensor, every tensor is a candidate image.
        """

        # embed each image (left or right)
        embs = self.return_embeddings(candidates)
        # embed the signal
        if len(signal.size()) == 3:
            signal = signal.squeeze(dim=-1)
        h_s = self.lin2(signal)
        # h_s is of size batch_size x embedding_size
        h_s = h_s.unsqueeze(dim=1)
        # h_s is of size batch_size x 1 x embedding_size
        h_s = h_s.transpose(1, 2)
        # h_s is of size batch_size x embedding_size x 1
        out = torch.bmm(embs, h_s)
        out = torch.cdist(embs, h_s.transpose(1,2), p=2)
        # out is of size batch_size x game_size x 1
        out = out.squeeze(dim=-1)

        if self.reinforce:
            probs = F.softmax(out, dim=1)
            dist = Categorical(probs=probs)
            sample = dist.sample()
            return sample, dist.log_prob(sample), dist.entropy()
        else:
            # out is of size batch_size x game_size
            log_probs = out
            # log_probs = F.log_softmax(out, dim=1)

            return log_probs

    def return_embeddings(self, x):
        # embed each image (left or right)
        embs = []
        for i in range(self.game_size):
            h = x[i]
            h_i = self.encoder(h)
            # h_i are batch_size x embedding_size
            h_i = h_i.unsqueeze(dim=1)
            # h_i are now batch_size x 1 x embedding_size
            embs.append(h_i)
        h = torch.cat(embs, dim=1)
        return h


class SymbolicReceiverMLPConstrastiveLoss(nn.Module):
    def __init__(self, 
                 game_size:int, 
                 embedding_size:int, 
                 hidden_size:int, 
                 input_dim:int, 
                 reinforce:bool=False,
    ) -> None:
        super().__init__()
        self.game_size = game_size # number of candidates
        self.embedding_size = embedding_size # size of messages embeddings
        self.hidden_size = hidden_size # size of hidden representation
        self.input_dim = input_dim

        self.reinforce = reinforce

        self.encoder = SymbolicSenderMLP(input_dim, hidden_size)
        self.lin2 = nn.Linear(embedding_size, hidden_size, bias=False)

    def forward(self, signal:torch.Tensor, x:torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        signal : torch.tensor
            Tensor for the embedding of received messages whose shape is $Batch_size * Hidden_size$
        x: torch.tensor
            Tensor for the original inputs whose shape is $Batch_size * input_dim$
        """
        # obj_emb shape: [batch_size, input_dim]
        obj_embedding = self.encoder(x).transpose(1,0) 
        # obj_emb shape: [hidden_dim, batch_size]
        dis_matrix = torch.matmul(signal, obj_embedding)
        # TODO: update the distance function
        # dis_matrix shape: [batch_size, batch_size]
        log_probs = F.log_softmax(dis_matrix, dim=1)
        return log_probs


class SymbolicReferGame(BaseGame):
    def __init__(self, 
                 training_log:str=None, 
                 num_msg_log:str=None,
                 language_save_freq:int=None,
                 game_size:int=None, 
                 track_compositionality:bool=True, 
                 valid_ratio:float=0.2,
                 mode:str='gumbel',
                 shuffle_data:bool=True,
    ) -> None:
        super().__init__(game_size)
        self.id = varname()

        self.training_log = training_log if training_log is not None else core.get_opts().training_log_path
        self.number_msg_log = num_msg_log if num_msg_log is not None else core.get_opts().num_msg_log_path
        self.language_save_freq = language_save_freq if language_save_freq is not None \
                                  else core.get_opts().language_save_freq

        self.loss_func = None

        if core.get_opts().contrastive:
            self.batch_size = self.game_size
            self.receiver_class = SymbolicReceiverMLPConstrastiveLoss
            self.loss_func = self.contrastive_loss
        else:
            self.receiver_class = SymbolicReceiverMLP
            self.loss_func = self.loss

        self.train_loader, self.test_loader = \
            get_symbolic_dataloader(
                n_attributes=self.n_attributes,
                n_values=self.n_values,
                batch_size=self.batch_size,
                game_size=self.game_size,
                referential=True,
                contrastive=core.get_opts().contrastive,
                validation_split=valid_ratio,
                shuffle=shuffle_data,
            )
        
        if mode == 'gumbel':
            self.sender = core.RnnSenderGS(
                              SymbolicSenderMLP(input_dim=self.n_attributes*self.n_values, hidden_dim=self.hidden_size),
                              self.vocab_size,
                              self.emb_size,
                              self.hidden_size,
                              max_len=self.max_len,
                              cell="lstm", 
                              temperature=1.0,
                              force_eos=False,
                          )
            self.receiver = core.RnnReceiverGS(
                                self.receiver_class(self.game_size, self.emb_size, self.hidden_size, 
                                    input_dim=self.n_attributes*self.n_values
                                ),
                                self.vocab_size,
                                self.emb_size,
                                self.hidden_size,
                                cell="lstm",
                            )
            self.game = core.SenderReceiverRnnGS(self.sender, self.receiver, self.loss_func)
            
        elif mode == 'reinforced':
            self.sender = core.RnnSenderReinforce(
                                  self.receiver_class(input_dim=self.n_attributes*self.n_values, hidden_dim=self.hidden_size),
                                    self.vocab_size,
                                    self.emb_size,
                                    self.hidden_size,
                                    max_len=self.max_len,
                                    cell="lstm", 
                                    force_eos=False,
                              )
            self.receiver = core.RnnReceiverDeterministic(
                                SymbolicReceiverMLP(self.game_size, self.emb_size, self.hidden_size, 
                                    input_dim=self.n_attributes*self.n_values
                                ),
                                self.vocab_size,
                                self.emb_size,
                                self.hidden_size,
                                cell="lstm",
                            )
            self.game = core.SenderReceiverRnnReinforce(self.sender, self.receiver, self.loss,
                                sender_entropy_coeff=0.1, receiver_entropy_coeff=0.05
                            )
        elif mode == 'reinforce':
            self.sender = core.RnnSenderReinforce(
                                  self.receiver_class(input_dim=self.n_attributes*self.n_values, hidden_dim=self.hidden_size),
                                    self.vocab_size,
                                    self.emb_size,
                                    self.hidden_size,
                                    max_len=self.max_len,
                                    cell="lstm", 
                                    force_eos=False,
                              )
            self.receiver = core.RnnReceiverReinforce(
                                SymbolicReceiverMLP(self.game_size, self.emb_size, self.hidden_size, 
                                    input_dim=self.n_attributes*self.n_values, reinforce=True,
                                ),
                                self.vocab_size,
                                self.emb_size,
                                self.hidden_size,
                                cell="lstm",
                            )
            self.game = core.SenderReceiverRnnReinforce(self.sender, self.receiver, self.reinforce_loss,
                            sender_entropy_coeff=0.1, receiver_entropy_coeff=0.05,
                        )
        else:
            raise ValueError("Available modes: gumbel, reinforced, reinforce")

        self.optimiser = core.build_optimizer(self.game.parameters())
        self.callbacks = []
        self.callbacks.append(ConsoleFileLogger(as_json=True,print_train_loss=True,logfile_path=self.training_log))
        self.callbacks.append(NumberMessages(logfile_path=self.number_msg_log, language_freq=self.language_save_freq))
        if track_compositionality:
            self.callbacks.append(TopographicSimilarityLatents(
                'hamming', 'edit', referential=True, log_path=core.get_opts().topo_path))
            # self.callbacks.append(core.TemperatureUpdater(agent=self.sender, decay=0.9, minimum=0.1))

        self.trainer = core.Trainer(
            game=self.game, optimizer=self.optimiser, train_data=self.train_loader, validation_data=self.test_loader,
            callbacks=self.callbacks
        )
    
    @staticmethod
    def reinforce_loss(_sender_input, _message, _receiver_input, receiver_output, labels):
        acc = (labels[1].squeeze(dim=1) == receiver_output).detach().float().mean().unsqueeze(dim=-1)
        return acc, {'acc': acc}

    @staticmethod
    def loss(_sender_input, _message, _receiver_input, receiver_output, labels):
        loss_fun = nn.CrossEntropyLoss()
        loss = loss_fun(receiver_output, labels[1].squeeze(dim=1))
        acc = (labels[1].squeeze(dim=1) == receiver_output.argmax(dim=1)).detach().float().mean().unsqueeze(dim=-1)
        return loss, {'acc': acc}

    @staticmethod
    def contrastive_loss(_sender_input, _message, _receiver_input, receiver_output, labels):
        batch_size = receiver_output.size(0)
        #ground_truth = torch.range(1,batch_size).long().to(receiver_output.device)
        ground_truth = torch.arange(batch_size).long().to(receiver_output.device)
        loss_fun = nn.CrossEntropyLoss()
        loss = loss_fun(receiver_output, ground_truth)
        acc = (receiver_output.argmax(dim=1) == ground_truth).detach().float().mean().unsqueeze(dim=-1)
        return loss, {'acc': acc}
