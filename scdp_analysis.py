#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})
plt.style.use('bmh')
from matplotlib import rc
rc('mathtext', default='regular')
from brokenaxes import brokenaxes
import seaborn as sns

import numpy as np
from pathlib import Path
from utils import create_dir_for_file, read_dir, get_plot_components


SOURCE_GAMES = {
    'refer2': 'black',
    'refer10': 'brown',
    'refer100': 'olive',
    'refer1000': 'purple',
    'refer2500': 'lime',
    'refer5000': 'cyan',
    'refer7500': 'fuchsia',
}

TARGET_GAMES = {
    'refer2': 'black',
    'refer10': 'brown',
    'refer100': 'olive',
    'refer1000': 'purple',
    'refer2500': 'lime',
    'refer5000': 'cyan',
    'refer7500': 'fuchsia',
    'refer10000': 'orange',
}

mean_dict = {}
up_dict = {}
low_dict = {}

def build_dicts(log_path:str='./logs/') -> None:
    def _add_data_to_dict(source_name:str, target_name:str) -> None:
        print('='*100)
        print('source game: ', source_name)
        if not source_name in mean_dict.keys():
            mean_dict[source_name] = {}
            up_dict[source_name] = {}
            low_dict[source_name] = {}
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
    
    for target_game in TARGET_GAMES.keys():
        for source_name in SOURCE_GAMES.keys():
            _add_data_to_dict(source_name, target_game)
            if source_name[:5] == 'refer' and source_name != 'refer10000':
                _add_data_to_dict(source_name + 'f', target_game)
                
                
def plot_gap_between_fixed_unfixed_context(result_path:str='./results/'):
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)
    
    def _collect_diffs_on_targets(source_name:str):
        diffs = []
        for target_name in TARGET_GAMES:
            diffs.append(mean_dict[source_name][target_name] - mean_dict[source_name + 'f'][target_name])
        return np.asarray(diffs)
    
    data = []
    for source_name in SOURCE_GAMES:
        diffs = _collect_diffs_on_targets(source_name)
        data.append(diffs)
    # plot points representing differences
    sns.swarmplot(data=data, color="turquoise", size=9, alpha=0.8)
    ax = sns.boxplot(data=data)
        
    # ax.set_xticks(x_pos)
    ax.set_xticklabels(list(SOURCE_GAMES.keys()))
    ax.set_xlabel('Source Games')
    ax.set_ylabel('Accuracy Difference')
    ax.set_ylim([-0.1, 0.4])
    ax.grid('both')
    
    _fig_file = result_path + 'diffs_statistics_on_all_sources.pdf'
    create_dir_for_file(_fig_file)
    plt.savefig(_fig_file, format='pdf', bbox_inches='tight')


def plot_expressivity_vs_complexity(result_path:str='./results') -> None:
    # x: source_game / increasing complexity
    # y: accuracy
    # lines: generalisation performance on various target games
    x_pos = np.arange(len(SOURCE_GAMES.keys())) + 1
    
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)
    
    def _plot_target_game(target_game:str, ax:complex) -> None:
        means = []
        errors = []
        for source_name in SOURCE_GAMES.keys():
            if source_name[:5] == 'refer' and source_name != 'refer10000':
                tmp_source_name = source_name + 'f'
            else:
                tmp_source_name = source_name
            means.append(mean_dict[tmp_source_name][target_game])
            errors.append(up_dict[tmp_source_name][target_game] - mean_dict[tmp_source_name][target_game])
        ax.plot(x_pos, means, ':+', label=target_game, color=TARGET_GAMES[target_game], linewidth=1.0)
        
        
    for target_game in TARGET_GAMES.keys():
        _plot_target_game(target_game, ax)
    
    x_labels = []
    for source_name in SOURCE_GAMES.keys():
        if source_name[:5] == 'refer' and source_name != 'refer10000':
            x_labels.append(source_name + 'f')
        else:
            x_labels.append(source_name)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Source Games')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Target Games')
    ax.set_ylim([0.68, 1.0])

    _fig_file = result_path + 'generalisation_vs_complexity.pdf'
    create_dir_for_file(_fig_file)
    plt.savefig(_fig_file, format='pdf', bbox_inches='tight')
    
    # print data in latex format
    for source_game in SOURCE_GAMES.keys():
        print('='*100)
        if source_game[:5] == 'refer' and source_game != 'refer10000':
            tmp_source_name = source_game + 'f'
        else:
            tmp_source_name = source_game
        print('source name:', tmp_source_name)
        means = []
        stds = []
        for target_game in TARGET_GAMES.keys():
            means.append(mean_dict[tmp_source_name][target_game])
            stds.append(up_dict[tmp_source_name][target_game] - mean_dict[tmp_source_name][target_game])
        print("\multicolumn{1}{|c|}{\multirow{2}{*}{" + tmp_source_name + "}} &")
        print("\multicolumn{1}{c|}{mean} &")
        for mean in means[:-1]:
            print("\multicolumn{1}{c|}{" + "%.4f" % mean +"} & ")
        print("\multicolumn{1}{c|}{" + "%.4f" % means[-1] +"} \\\\ \cmidrule(l){2-10} ")
        print("\multicolumn{1}{|c|}{} &")
        print("\multicolumn{1}{c|}{$\sigma$} &")
        for std in stds[:-1]:
            print("\multicolumn{1}{c|}{" + "%.4f" % std +"} & ")
        print("\multicolumn{1}{c|}{" + "%.4f" % stds[-1] + "} \\\\ \midrule ")


def plot_source_on_target(target_game:str,
                            result_path:str='./results/',
) -> None:

    # 2. plot (x: target, y: accuracy)
    x_pos = np.arange(len(SOURCE_GAMES.keys())) + 1
    
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)
    
    def _plot_target_game(ax: complex, style:str, fixed_context:bool=False) -> None:
        means = []
        ups = []
        for source_name in SOURCE_GAMES.keys():
            tmp_source_name = source_name + 'f' if fixed_context and not source_name == 'refer10000' else source_name
            means.append(mean_dict[tmp_source_name][target_game])
            ups.append(up_dict[tmp_source_name][target_game])
        label = 'Un-predictable' if not fixed_context else 'Predictable'
        ax.plot(x_pos, means, style, label=label, color=TARGET_GAMES[target_game], linewidth=1.0)
    
    _plot_target_game(ax, '-x')
    _plot_target_game(ax, ':+', True)
    
    ax.set_ylabel('Accuracy on referential games')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(SOURCE_GAMES.keys())) 
    ax.set_xlabel('Source Games')
    ax.legend(loc=2)
    ax.set_title(target_game)
    _fig_file = result_path + 'acc_sources_on_'+ target_game + '.pdf'
    create_dir_for_file(_fig_file)
    plt.savefig(_fig_file, format='pdf', bbox_inches='tight')


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

    build_dicts(args.log_path)
    
    plot_expressivity_vs_complexity(args.result_path)
    plot_gap_between_fixed_unfixed_context(args.result_path)