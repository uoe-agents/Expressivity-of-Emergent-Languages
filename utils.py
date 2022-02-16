#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Union, Callable
import os
from pathlib import Path
import json
import pickle
from typing  import Tuple

import numpy as np
from scipy.stats import spearmanr

import torch
import torch.nn as nn
import torch.nn.init as init
from egg.core.language_analysis import TopographicSimilarity
from egg.core import Callback
from egg.core.interaction import Interaction

# NOTE: This is a global variable storing the list of games we train agents one and the corresponding colour for plotting their curves.
GAMES = {
    'refer2': 'black',
    'refer10': 'brown',
    'refer100': 'olive',
    'refer1000': 'purple',
    'refer2500': 'lime',
    'refer5000': 'cyan',
    'refer7500': 'fuchsia',
    'refer10000': 'orange',
}


def get_plot_components(data_matrix):
    data_mean = np.mean(data_matrix, axis=0)
    data_var = np.std(data_matrix, axis=0)
    upper = data_mean + data_var
    lower = data_mean - data_var

    return data_mean, upper, lower


class TopographicSimilarityLatents(TopographicSimilarity):
    def __init__(self,
                 sender_input_distance_fn: Union[str, Callable] = 'cosine',
                 message_distance_fn: Union[str, Callable] = 'edit',
                 compute_topsim_train_set: bool = True,
                 compute_topsim_test_set: bool = False,
                 referential=False,
                 log_path='log/top_sim.txt',
                ):

        super().__init__(sender_input_distance_fn, message_distance_fn, compute_topsim_train_set, compute_topsim_test_set)
        self.referential = referential
        self.log_path = log_path

        create_dir_for_file(self.log_path)

    @staticmethod
    def compute_distance(_list, distance):
        return [distance(el1, el2)
                for i, el1 in enumerate(_list[:-1])
                for j, el2 in enumerate(_list[i+1:])
               ]

    def compute_similarity(self, sender_input: torch.Tensor, messages: torch.Tensor, epoch: int):
        message_argmax = messages.argmax(-1)
        messages_argmax = [msg.tolist() for msg in message_argmax]
        if self.referential:
            sender_input = sender_input[:, :3] # To tackle the labels in referential game.
        input_dist = self.compute_distance(
            sender_input.detach().cpu().numpy(), self.sender_input_distance_fn)
        message_dist = self.compute_distance(messages_argmax, self.message_distance_fn)
        topsim = spearmanr(input_dist, message_dist, nan_policy='raise').correlation

        output_message = json.dumps(dict(topsim=topsim, epoch=epoch))
        print(output_message, flush=True)
        with open(self.log_path, 'a') as f:
            print(output_message, file=f)


class ConsoleFileLogger(Callback):
    def __init__(self, print_train_loss=True, as_json=True, logfile_path=None):
        self.print_train_loss = print_train_loss
        self.as_json = as_json
        self.logfile_path = logfile_path

    def aggregate_print(self, loss: float, logs: Interaction, mode: str, epoch: int):
        dump = dict(loss=loss) 
        aggregated_metrics = dict((k, v.mean().item()) for k, v in logs.aux.items())
        dump.update(aggregated_metrics)

        if self.as_json:
            dump.update(dict(mode=mode, epoch=epoch))
            output_message = json.dumps(dump)
        else:
            output_message = ', '.join(sorted([f'{k}={v}' for k, v in dump.items()]))
            output_message = f'{mode}: epoch {epoch}, loss {loss}, ' + output_message
        print(output_message, flush=True)
        if self.logfile_path is not None:
            create_dir_for_file(self.logfile_path)
            with open(self.logfile_path,'a') as f:
                print(output_message, file=f)

    def on_test_end(self, loss: float, logs: Interaction, epoch: int):
        self.aggregate_print(loss, logs, 'test', epoch)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if self.print_train_loss:
            self.aggregate_print(loss, logs, 'train', epoch)


class NumberMessages(Callback):
    def __init__(self, as_json:bool=True, logfile_path:str=None, language_freq:int=5) -> None:
        self.as_json = as_json
        self.logfile_path = logfile_path
        self.language_freq = language_freq

    @staticmethod
    def tensor_to_list(tensor:torch.Tensor):
        if tensor.dim() == 3:
            return_list = [matrix.detach().argmax(dim=1).to(torch.int8).tolist() for matrix in tensor]
        elif tensor.dim() == 2:
            return_list = [vec.detach().to(torch.int8).tolist() for vec in tensor]
        else:
            raise ValueError("Unknown dimensions of message tensor.")
    
        return return_list

    def print_message_type_numbers(self, message:torch.Tensor, head:str) -> None:
        messages = self.tensor_to_list(message)
        num_total = len(messages)
        num_msgs = len(set([tuple(x) for x in messages]))
        print('['+head+']Number of different messages:', num_msgs, '/', num_total)

        if self.logfile_path is not None:
            output_msg = str(num_msgs) + ',' + str(num_total)
            log_file = self.logfile_path + '/' + 'msg_num.txt'
            create_dir_for_file(log_file)
            with open(log_file, 'a') as f:
                print(head+':', output_msg, file=f)

    def print_language(self, meaning:torch.Tensor, message:torch.Tensor, head:str, epoch:int) -> None:
        assert meaning.dim() == 2
        meanings = self.tensor_to_list(meaning)
        messages = self.tensor_to_list(message)
        assert len(meanings) == len(messages)

        list_mappings = []
        for i in range(len(meanings)):
            list_mappings.append([meanings[i], messages[i]])
        
        if self.logfile_path is not None:
            mapping_file = self.logfile_path + '/' + head + '/' + str(epoch) + '.pkl'
            create_dir_for_file(mapping_file)
            with open(mapping_file, 'wb') as f: 
                pickle.dump(list_mappings, f)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int) -> None:
        self.print_message_type_numbers(logs.message, 'Train')
        if epoch % self.language_freq == 0 or epoch == 1:
            self.print_language(logs.sender_input, logs.message, 'Train', epoch)
    
    def on_test_end(self, loss: float, logs: Interaction, epoch: int) -> None:
        self.print_message_type_numbers(logs.message, 'Test')
        if epoch% self.language_freq == 0 or epoch == 1:
            self.print_language(logs.sender_input, logs.message, 'Test', epoch)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def smooth(x,window_len=20,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def create_dir_for_file(file_path:str):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def parse_game_name(game_name:str):
    game_type = game_name[:5]
    shuffle = True
    if game_name[-1] == 'f':
        shuffle = False
        game_name = game_name[:-1]
    if game_type == 'refer':
        return (game_type, int(game_name[5:]), shuffle)
    else:
        raise ValueError(f"Unrecognised game type: {game_type}!")


def read_topo_sim_file(fpath: str) -> list:
    topo_sim_list = []
    data_file = open(fpath, 'r')
    for line in data_file.readlines():
        result = json.loads(line.strip())
        topo_sim_list.append(float(result['topsim']))
    return topo_sim_list


def read_topo_sim_dir(dpath: str) -> list:
    file_name_list = []
    for (_, _, filenames) in walk(dpath):
        file_name_list.extend(filenames)
        break

    data_matrix = []
    for file_name in file_name_list:
        data_list = read_topo_sim_file(dpath+'/'+file_name)
        data_matrix.append(data_list)

    return data_matrix


def read_task_transfer_file(fpath: str, **kwargs) -> list:
    train_list = []
    test_list = []
    with open(fpath, 'r') as data_file:
        for line in data_file.readlines():
            train_data, test_data = line.strip().split(',')
            train_list.append(float(train_data))
            test_list.append(float(test_data))
    return train_list, test_list


def read_task_train_file(fpath: str, data_key='loss') -> Tuple[list, list]:
    train_list = []
    test_list = []
    with open(fpath, 'r') as data_file:
        for line in data_file.readlines():
            data = json.loads(line.strip())
            if data['mode'] == 'train':
                train_list.append(data[data_key])
            elif data['mode'] == 'test':
                test_list.append(data[data_key])
            else:
                raise ValueError("Unrecognised mode.")

    return train_list, test_list


def read_dir(dpath: str, read_file_function=read_task_transfer_file, data_key='loss') -> list:
    file_name_list = []
    for (_, _, filenames) in walk(dpath):
        file_name_list.extend(filenames)
        break

    train_matrix = []
    test_matrix = []
    for file_name in file_name_list:
        train_list, test_list = read_file_function(dpath+'/'+file_name, data_key=data_key)
        train_matrix.append(train_list)
        test_matrix.append(test_list)

    return train_matrix, test_matrix
