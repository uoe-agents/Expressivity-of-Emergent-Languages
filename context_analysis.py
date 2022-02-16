#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})
plt.style.use('bmh')
from matplotlib import rc
rc('mathtext', default='regular')
from brokenaxes import brokenaxes


import math
import pickle
import itertools
import numpy as np
import os
from os import walk
from pathlib import Path
from sklearn.manifold import TSNE
import editdistance
from scipy.stats import spearmanr, ttest_ind

from utils import smooth, create_dir_for_file, get_plot_components, read_dir
from data_loader import SymbolicDataset

GAMES = {
    'refer2': 'black',
    'refer10': 'brown',
    'refer100': 'olive',
    'refer1000': 'purple',
    'refer2500': 'lime',
    'refer5000': 'cyan',
    'refer7500': 'fuchsia',
    'refer10000': 'orange'
}


def read_msg_num_file(file_path:str) -> Tuple[List, List, int]:
    train_list = []
    test_list = []
    total_num = None
    with open(file_path, 'r') as data_file:
        for line in data_file.readlines():
            dtype = line.split(':')[0]
            if dtype == 'Train':
                train_list.append(int(line.split(':')[1].split(',')[0].strip()))
            elif dtype == 'Test':
                test_list.append(int(line.split(':')[1].split(',')[0].strip()))
            else:
                raise ValueError("Unrecognised mode.")
        total_num = int(line.split(':')[1].split(',')[1].strip())

    return train_list, test_list, total_num


def read_msg_num_text_in_dir(dir_path:str) -> Tuple[List[List], List[List]]:
    seed_dir_list = []
    for (_, dirnames, _) in walk(dir_path):
        seed_dir_list.extend(dirnames)
        break

    train_matrix = []
    test_matrix = []
    total_num = None
    for seed in seed_dir_list:
        file_path = str(Path(dir_path).joinpath(seed, 'msg_num.txt').absolute())
        train_list, test_list, total_num = read_msg_num_file(file_path)
        train_matrix.append(train_list)
        test_matrix.append(test_list)

    return train_matrix, test_matrix, total_num


def read_language_from_pickle(pkl_path:str) -> List[Tuple[List, List]]:
    list_mappings = None
    with open(pkl_path, 'rb') as f: 
        list_mappings = pickle.load(f)

    return list_mappings


def get_msg_stats4language(language:List[Tuple[List, List]]) -> Dict:
    message_statistics = {}
    
    for i in range(len(language)):
        meaning = tuple(language[i][0])
        message = tuple(language[i][1])

        if not message in message_statistics.keys():
            message_statistics[message] = {
                'count': 1,
                'meaning_list':[meaning],
            }
        else:
            message_statistics[message]['count'] += 1
            message_statistics[message]['meaning_list'].append(meaning)

    return message_statistics


def get_lan_msg_statistics(lan_pkl_path:str, K:int=30) -> Dict:
    lan_mappings = read_language_from_pickle(lan_pkl_path)

    msg_stats = get_msg_stats4language(lan_mappings)
    # sort the msg_stats by messages' counts
    msg_stats = dict(sorted(msg_stats.items(), key=lambda item: -item[1]['count']))

    return dict(itertools.islice(msg_stats.items(), K))


def get_msg_stats_under_dir(dir_path:str, epoch:str, K:int=30) -> List[Dict]:
    """
        K: int, the top-K most frequent messages we will return for further analysis
    """
    seed_dir_list = []
    for (_, dirnames, _) in walk(dir_path):
        seed_dir_list.extend(dirnames)
        break
    
    msg_stats_list = []
    for seed in seed_dir_list:
        file_path = str(Path(dir_path).joinpath(seed, 'Train', epoch+'.pkl').absolute())
        msg_stat = get_lan_msg_statistics(file_path, K)
        msg_stats_list.append(msg_stat)

    return msg_stats_list


def get_lan_mutual_info(pkl_path:str) -> int:
    lan_mappings = read_language_from_pickle(pkl_path)
    msg_stats = get_msg_stats4language(lan_mappings)
    msg_counts = sorted([item[1]['count'] for item in msg_stats.items()], reverse=True)
    N = len(msg_counts)
    mi = N * math.log(N) - np.sum(np.log(msg_counts))
    return mi


def plot_mutual_info_4_all_games(log_path:str, result_path:str) -> None:
    print('='*100)
    print('Mutual information values')
    # 1. find out all the seeds
    seed_list = []
    for (_, dirnames, _) in walk(log_path + list(GAMES.keys())[0] + '_msgnum'):
        seed_list.extend(dirnames)
        break

    # 2. find the common epochs
    common_epochs = []
    for game in GAMES.keys():
        game_epochs = []
        for (_, _, filenames) in walk(log_path + game + '_msgnum/' + seed_list[0] + '/Train/'):
            game_epochs.extend([int(name.split('.')[0]) for name in filenames])
        common_epochs = list(set(common_epochs) & set(game_epochs)) if len(common_epochs) > 0 else game_epochs
    common_epochs = sorted(common_epochs)

    # 3. read the msg_stat under every game+seed+epoch
    def _plot_line(game):
        epoch_list = []
        if not game in ['refer2', 'refer10', 'refer100']:
            epoch_list = [e * 10 for e in common_epochs]
            epoch_list[0] = 1
        else:
            epoch_list = common_epochs

        means = []
        errs = []
        for epoch in epoch_list:
            epoch_MIs = []
            for seed in seed_list:
                lan_pkl_file = log_path + game + '_msgnum/' + seed + '/Train/' + str(epoch) + '.pkl'
                epoch_MIs.append(get_lan_mutual_info(lan_pkl_file))
            means.append(np.mean(epoch_MIs))
            errs.append(np.std(epoch_MIs))

        print(game, ':\t mean: ', means[-1], '\t std: ', errs[-1])

        plt.errorbar(np.arange(1, len(means) + 1), means, yerr=errs, label=game, color=GAMES[game], ecolor=GAMES[game],
                     capthick=1.0, capsize=2.0, linewidth=1.0, fmt=':o')


    plt.figure(figsize=(11, 6))
    for game in GAMES.keys():
        if game == 'refer2': continue
        _plot_line(game)
    plt.legend(title='Source Games')
    plt.xlabel('Epochs')
    plt.ylabel('Mutual Information')
    plt.xticks(np.arange(1, len(common_epochs) + 1), common_epochs, fontsize=10)


    _fig_file = result_path + 'MI.pdf'
    create_dir_for_file(_fig_file)
    plt.savefig(_fig_file, format='pdf', bbox_inches='tight')


def get_msg_counts_matrix(msg_stats_list:List[Dict]) -> np.array:
    msg_freqs_list = []

    for msg_stats in msg_stats_list:
        msg_freqs = list(item[1]['count'] for item in msg_stats.items())
        msg_freqs_list.append(msg_freqs)

    return np.asarray(msg_freqs_list)


def cal_meaning_dis(pair:Tuple[tuple, tuple]) -> int:
    dis = [int(not i == j) for i,j in zip(pair[0], pair[1])]
    return sum(dis)


def get_msg_avg_meaning_dis_matrix(msg_stats_list:List[Dict]) -> np.array:
    avg_meaning_dis_list = []
    def _cal_meaning_dis_list(meaning_list:List[int]) -> int:
        if len(meaning_list) == 1:
            return 0
        elif len(meaning_list) == 2:
            return cal_meaning_dis(meaning_list)
        else:
            pairs = list(zip(meaning_list, meaning_list[1:] + meaning_list[:1]))
        return [cal_meaning_dis(pair) for pair in pairs]

    for msg_stats in msg_stats_list:
        meaning_dis_list = []
        for msg in msg_stats.keys():
            meaning_dis_list.append(np.mean(_cal_meaning_dis_list(msg_stats[msg]['meaning_list'])))
        avg_meaning_dis_list.append(meaning_dis_list)

    return np.asarray(avg_meaning_dis_list)


def plot_language_msg_counts_distribution(log_dir_path:str,
                                          epochs:list=['1', '500'], 
                                          K:int=10,
                                          result_file_path:str='./results/',
) -> None:
    h_shifts = [-0.2, 0.2]
    legend_list = ['beginning', 'end']
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)
    bars = []
    color_list = [u'#348ABD', u'#A60628']

    for idx, epoch in enumerate(epochs):
        msg_stats_list = get_msg_stats_under_dir(log_dir_path, epoch, K=K)
        msg_freqs_matrix = get_msg_counts_matrix(msg_stats_list)
        msg_freqs, _, msg_freqs_down = get_plot_components(msg_freqs_matrix)
        msg_freqs_stds = msg_freqs - msg_freqs_down

        bar = ax.bar(np.arange(K)+1+h_shifts[idx], msg_freqs, yerr=msg_freqs_stds, align='center',  color=color_list[idx],
                    alpha=0.5, width=0.4, 
                    error_kw=dict(lw=2, capsize=5, capthick=0.5, alpha=0.5, color=color_list[idx])
            )
        bars.append(bar)

    ax.set_ylabel('Frequency')
    ax.set_xticks(np.arange(K)+1)
    ax.legend(bars, legend_list)
    ax.set_ylim([0, 4])

    _result_file_path = result_file_path + log_dir_path.split('/')[-2].split('_')[0] + '_msgfreq_change.pdf'
    create_dir_for_file(_result_file_path)
    fig.savefig(_result_file_path, format='pdf', bbox_inches='tight')
    
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)
    bars = []
    for idx, epoch in enumerate(epochs):
        msg_stats_list = get_msg_stats_under_dir(log_dir_path, epoch, K=K)
        meaning_dis_matrix = get_msg_avg_meaning_dis_matrix(msg_stats_list)
        meaning_dis, _, meaning_dis_down = get_plot_components(meaning_dis_matrix)
        meaning_dis_std = meaning_dis - meaning_dis_down
        bar = ax.bar(np.arange(K)+1+h_shifts[idx], meaning_dis, yerr=meaning_dis_std, align='center',  color=color_list[idx],
                    alpha=0.5, width=0.4, 
                    error_kw=dict(lw=2, capsize=5, capthick=0.5, alpha=0.5, color=color_list[idx])
            )
        bars.append(bar)

    ax.set_ylabel('Distance')
    ax.set_xticks(np.arange(K)+1)
    ax.legend(bars, legend_list)
    ax.set_ylim([0, 5])

    _result_file_path = result_file_path + log_dir_path.split('/')[-2].split('_')[0] + '_msgdis_change.pdf'
    create_dir_for_file(_result_file_path)
    fig.savefig(_result_file_path, format='pdf', bbox_inches='tight')


def plot_context_experiment_curves(num_epochs:int=5000, 
                                   window_size:int=200, 
                                   log_path:str='./log/',
                                   result_path:str='./results/'
) -> None:
    def _plot_curves_on_target_game(target_game:str, log_path:str, num_epochs:int, window_size:int):
        plt.clf()
        x_axis = np.arange(num_epochs) + 1

        y_min = 2.0
        y_max = -1.0
        for game_name in GAMES.keys():
            dir_path = str(Path(log_path).joinpath(game_name+'_to_'+target_game).absolute())
            _, test_matrix = read_dir(dir_path)
            
            mean, up, low = get_plot_components(test_matrix)
            y_max = smooth(up, window_size)[:num_epochs].max() \
                if smooth(up, window_size)[:num_epochs].max() > y_max else y_max
            y_min = smooth(low, window_size)[:num_epochs].min() \
            if smooth(low, window_size)[:num_epochs].min() < y_min else y_min

            plt.plot(x_axis,
                     smooth(mean, window_size)[:num_epochs],
                     label=game_name, linewidth=0.5, color=GAMES[game_name],
                    )
            plt.fill_between(x_axis, 
                             smooth(up, window_size)[:num_epochs], 
                             smooth(low, window_size)[:num_epochs], 
                             color=GAMES[game_name], alpha=0.2,
                            )
        
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy') 
        plt.legend(loc='best')
        # plt.grid()

        plt.ylim([max(0.0, y_min - 0.05), min(1.0, y_max + 0.05)])

        _fig_file = result_path + 'context_on_' + target_game + '.pdf'
        create_dir_for_file(_fig_file)
        plt.savefig(_fig_file, format='pdf', bbox_inches='tight')

    
    for game_name in GAMES.keys():
        _plot_curves_on_target_game(game_name, log_path, num_epochs, window_size)


def plot_msg_numbers_distributions(log_path:str='./log/', window_size:int=10, result_path:str='./results/') -> None:
    print('='*100)
    print('Message type numbers of each game')
    plt.clf()
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_axes([0,0,1,1])
    data_to_plot = []
    game_name_list = []

    for game_name in GAMES.keys():
        dir_path = str(Path(log_path).joinpath(game_name+'_msgnum').absolute())
        msgnum_matrix, _, total_num = read_msg_num_text_in_dir(dir_path)
        msgnum_matrix = np.asarray(msgnum_matrix, dtype='float')[:, -window_size:].flatten()

        data_to_plot.append(msgnum_matrix)
        game_name_list.append(game_name)
        print(game_name, '\t mean: ', np.mean(msgnum_matrix), '\t 25th: ', np.percentile(msgnum_matrix, 25),
              '\t 75th: ', np.percentile(msgnum_matrix, 75))


    def _set_axis_style(ax, labels):
        _labels = []
        for label in labels:
            if label in ['refer2', 'refer10']:
                _labels.append(label)
            else:
                _labels.append(label[5:])
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(_labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Type of Game')
        ax.set_ylabel('Input Space Size')
        ax.set_title('Number of Messages from Different Games')

    def _adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value

    parts = ax.violinplot(data_to_plot, showmeans=False, showmedians=False, showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor('#1E90FF')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    
    means = [np.mean(data_matrix) for data_matrix in data_to_plot]
    quartile1, medians, quartile3 = np.percentile(data_to_plot, [25, 50, 75], axis=1)
    whiskers = np.array([_adjacent_values(sorted_array, q1, q3)
                            for sorted_array, q1, q3 in zip(data_to_plot, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    _set_axis_style(ax, game_name_list)

    _fig_file = result_path + 'msg_num_dist.pdf'
    create_dir_for_file(_fig_file)
    fig.savefig(_fig_file, format='pdf', bbox_inches='tight')


def plot_msg_numbers_curves(log_path:str='./log/', 
                            num_epochs:int=1000, 
                            window_size:int=2, 
                            result_path:str='./results/'
) -> None:
    plt.clf()
    plt.figure(figsize=(11, 6))
    x_axis = np.arange(num_epochs) + 1
    y_max = 10000
    y_min = 0

    def _plot_msg_num_curve(game_name:str, log_path:str, num_epochs:int, window_size:int):
        dir_path = str(Path(log_path).joinpath(game_name+'_msgnum').absolute())
        msgnum_matrix, _, _ = read_msg_num_text_in_dir(dir_path)
        
        mean, up, __ = get_plot_components(msgnum_matrix)
        stds = np.sqrt(up - mean)
        
        plt.plot(x_axis,
                 smooth(mean, window_size)[:num_epochs],
                 label=game_name, linewidth=0.5, color=GAMES[game_name],
                )
        plt.fill_between(x_axis,
                         smooth(mean + stds, window_size)[:num_epochs],
                         smooth(mean - stds, window_size)[:num_epochs],
                         color=GAMES[game_name],
                         alpha=0.2,
                        )

    for game_name in GAMES.keys():
        _plot_msg_num_curve(game_name, log_path, num_epochs, window_size)
    
    plt.xlabel('Epochs')
    plt.ylabel('Number of Messages') 
    plt.legend(loc='best', title='Source Games')
    # plt.grid()
    plt.ylim([y_min, y_max])

    _fig_file = result_path + 'msg_num_curves.pdf'
    create_dir_for_file(_fig_file)
    plt.savefig(_fig_file, format='pdf', bbox_inches='tight')


def plot_tsne_msg_mappings(lan_pkl_path:str, 
                           result_file_path:str, 
                           K:int=10, 
                           n_attributes:int=4, 
                           n_values:int=10,
) -> None:
    assert K <= 10

    language_stats = get_lan_msg_statistics(lan_pkl_path, K)
    meanings, labels = SymbolicDataset._build_samples(n_attributes, n_values)

    label2idx = {}
    for i in range(labels.shape[0]):
        label2idx[tuple(labels[i])] = i

    tsne = TSNE(n_components=2, random_state=0)
    meanings_2d = tsne.fit_transform(meanings)

    fig, ax = plt.subplots(figsize=(12,12))
    # for i in range(meanings_2d.shape[0]):
    #     ax.scatter(meanings_2d[i][0], meanings_2d[i][1], color='grey', alpha=0.2)

    def _meaning2coordinates(meaning:tuple) -> Tuple[int, int]:
        idx = label2idx[meaning]
        return tuple(meanings_2d[idx])

    def _msg2label(msg:tuple) -> str:
        return ''.join([str(x) for x in msg])

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple']
    for msg, color in zip(list(language_stats.keys()), colors[:K]):
        # msg is a 6-digits-long tuple
        for meaning in language_stats[msg]['meaning_list']:
            coorinates = _meaning2coordinates(meaning)
            ax.scatter(coorinates[0], coorinates[1], label=_msg2label(msg), color=color)
            
    def legend_without_duplicate_labels(ax):
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique))
    
    legend_without_duplicate_labels(ax)
    ax.set_xlim(-45, 45)
    ax.set_ylim(-45, 45)

    if not result_file_path.endswith('.pdf'):
        result_file_path = result_file_path + lan_pkl_path.split('/')[-4].split('_')[0] + '_msgvis_' + \
            lan_pkl_path.split('/')[-1].split('.')[0] + '.pdf'

    create_dir_for_file(result_file_path)
    fig.savefig(result_file_path)


def plot_source_vs_target(log_path:str='./log/',
                          result_path:str='./results/',
) -> None:
    # 1. build 3 dicts to record the converged performance
    print('='*100)
    mean_dict = {}
    up_dict = {}
    low_dict = {}
    for source_name in GAMES.keys():
        print('source game: ', source_name)
        if not source_name in mean_dict.keys():
            mean_dict[source_name] = {}
            up_dict[source_name] = {}
            low_dict[source_name] = {}
        for target_name in GAMES.keys():
            dir_path = str(Path(log_path).joinpath(source_name+'_to_'+target_name).absolute())
            print(dir_path)
            _, test_matrix = read_dir(dir_path)
            test_matrix = np.asarray(test_matrix)
            print(test_matrix)
            mean, up, down = get_plot_components(test_matrix)
            mean_dict[source_name][target_name] = mean[-1] 
            up_dict[source_name][target_name] = up[-1]
            print('\t on ', target_name, '\tmean: ', mean_dict[source_name][target_name], 
                  '\t up: ', up_dict[source_name][target_name],
                  '\tstd: ', up_dict[source_name][target_name] - mean_dict[source_name][target_name])

    x_pos = np.arange(len(GAMES.keys())) + 1

    # function for plotting game performance in the following diagrams
    def _plot_lines(out_game:str, reverse:bool=False) -> None:
        means = []
        ups = []
        for in_game in GAMES.keys():
            mean_val = mean_dict[out_game][in_game] if reverse else mean_dict[in_game][out_game]
            means.append(mean_val)
            up_val = up_dict[out_game][in_game] if reverse else up_dict[in_game][out_game]
            ups.append(up_val)
        plt.errorbar(x_pos, means, yerr=np.asarray(ups) - np.asarray(means), label=out_game, fmt='--', 
                     color=GAMES[out_game], ecolor=GAMES[out_game], capthick=1.0, capsize=2.0, linewidth=1.0)


    # 2. plot (x: source, y: accuracy)
    plt.figure(figsize=(11, 6))
    for target_name in GAMES.keys():
        _plot_lines(target_name)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Target Games')
    plt.xlabel('Source Games')
    plt.ylabel('Performance')
    plt.xticks(x_pos, list(GAMES.keys()))
    plt.ylim([0.6, 1.0])

    _fig_file = result_path + 'acc_from_sources.pdf'
    create_dir_for_file(_fig_file)
    plt.savefig(_fig_file, format='pdf', bbox_inches='tight')


    # 3. plot (x: target, y: accuracy)
    plt.figure(figsize=(11, 6))
    for source_name in GAMES.keys():
        _plot_lines(source_name, reverse=True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Source Games')
    plt.xlabel('Target Games')
    plt.ylabel('Performance')
    plt.xticks(x_pos, list(GAMES.keys())) 
    # plt.ylim([0.6, 1.0])

    _fig_file = result_path + 'acc_from_targets.pdf'
    create_dir_for_file(_fig_file)
    plt.savefig(_fig_file, format='pdf', bbox_inches='tight')


    # 4. a zoom-in version of 3
    plt.figure(figsize=(11, 6))
    for source_name in GAMES.keys():
        means = []
        ups = []
        for target_name in GAMES.keys():
            means.append(mean_dict[source_name][target_name])
            ups.append(up_dict[source_name][target_name])
        plt.plot(x_pos, means, ':o', label=source_name, color=GAMES[source_name], linewidth=1.0)
    plt.ylim([0.755, 0.83])
    plt.xticks(x_pos, list(GAMES.keys())) 
    plt.xlim([4.7, 8.3])
    _fig_file = result_path + 'acc_source_vs_target_zoomin.pdf'
    plt.savefig(_fig_file, format='pdf', bbox_inches='tight')


def cal_msg_dist(msg1:List, msg2:List, method:str='edit') -> float:
    dist = 10
    if method == 'edit':
        dist = editdistance.eval([str(x) for x in msg1], [str(x) for x in msg2])
    else:
        raise NotImplementedError
    return dist


def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


def print_transfer_exp_p_values(log_path:str='./log/', result_path:str='./results/') -> None:
    mean_dict = {}
    for source_name in GAMES.keys():
        if not source_name in mean_dict.keys():
            mean_dict[source_name] = {}
        for target_name in GAMES.keys():
            dir_path = str(Path(log_path).joinpath(source_name+'_to_'+target_name).absolute())
            _, test_matrix = read_dir(dir_path)
            test_matrix = np.asarray(test_matrix)
            # mean, up, down = get_plot_components(test_matrix)
            time_step = 1470
            time_length = 30
            mean_dict[source_name][target_name] = np.mean(test_matrix[:, time_step:time_step+time_length], axis=1)

    out_file = result_path + 'p_values.md'
    f = open(out_file, mode='w')
    print("""|       Sources        | refer2 | refer10 | refer100 | refer1000 | refer2500 | refer5000 | refer7500 | refer10000 |
| :------------------: | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |""", file=f)

    game_pairs = [pair for pair in itertools.combinations(GAMES.keys(), 2)]

    for pair in game_pairs:
        source1 = pair[0]
        source2 = pair[1]
        print('| ' + source1 + ' vs ' + source2 + ' |', end=' ', file=f)
    
        for target_name in GAMES.keys():
            _, p_value = ttest_ind(mean_dict[source1][target_name], mean_dict[source2][target_name], 
                                    equal_var=False,
                                )
            print("{:.2e}".format(p_value), end=' |', file=f)
        print('', file=f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--num_epochs_generalise_curve', type=int, default=1500, 
                        help='number of epochs shown for retrain generalisation performance.')
    parser.add_argument('--num_epochs_msg_number', type=int, default=500,
                        help='number of epochs shown in the msg_num curves.')
    parser.add_argument('--smooth_window_size', type=int, default=2, help='window size for smoothing the curves.')
    parser.add_argument('--msgnum_window_size', type=int, default=20, help='window size for number of messages.')
    parser.add_argument('--log_path', type=str, default='./log/',
                        help='path to the log files directory')
    parser.add_argument('--result_path', type=str, default='./results/',
                        help='Path to the dir for storing results.')
    args = parser.parse_args()
    
    plot_context_experiment_curves(num_epochs=args.num_epochs_generalise_curve, 
                                   window_size=args.smooth_window_size, 
                                   log_path=args.log_path,
                                   result_path=args.result_path,
                                   )

    plot_msg_numbers_distributions(log_path=args.log_path, 
                                   window_size=args.msgnum_window_size,
                                   result_path=args.result_path,
                                   )

    plot_msg_numbers_curves(log_path=args.log_path, 
                            num_epochs=args.num_epochs_msg_number, 
                            window_size=args.msgnum_window_size,
                            result_path=args.result_path,
                           )

    plot_language_msg_counts_distribution(log_dir_path='./log/refer1000_msgnum/',
                                          epochs=['1', '500'],
                                          K=10,
                                          result_file_path=args.result_path,
                                         )
    
    plot_source_vs_target(args.log_path, args.result_path)
    plot_mutual_info_4_all_games(args.log_path, args.result_path)
    print_transfer_exp_p_values(log_path=args.log_path, result_path=args.result_path)