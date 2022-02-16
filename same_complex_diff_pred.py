#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# script to run experiments with same complexity but different predictability of context
from modules import BaseGame
import egg.core as core

from refer_game import SymbolicReferGame
from utils import parse_game_name
from context_num import transfer_game_to_game


NUM_DYAD_TRAIN_EPOCHS = 500 # number of epochs for dyads to converge
NUM_RETRAIN_EPOCHS = 1500 # number of epochs for retraining listener
SOURCE_GAMES = [
                'refer100',
                'refer100f',
                'refer1000',
                'refer1000f',
                'refer7500',
                'refer7500f',
                'refer10000',
               ]
TARGET_GAMES = [
                'refer100',
                'refer1000',
                'refer7500',
                'refer10000',
               ]


demo_game = BaseGame()
del demo_game

for game in SOURCE_GAMES:
    # 1. train the dyads to obtain the emergent languages
    source_game = None
    train_log_file = f"{core.get_opts().generalisation_path}/{game}_train/{str(core.get_opts().random_seed)}.txt"
    msg_log_path = f"{core.get_opts().generalisation_path}/{game}_msgnum/{str(core.get_opts().random_seed)}/"
    game_type = parse_game_name(game)[0]
    game_size = parse_game_name(game)[1]
    shuffle = parse_game_name(game)[2]
    if not shuffle:
        print(game)
    source_game = SymbolicReferGame(training_log=train_log_file,
                                    num_msg_log=msg_log_path,
                                    game_size=game_size, 
                                    track_compositionality=False, 
                                    valid_ratio=0.0, 
                                    shuffle_data=shuffle,
                                    )
    print('start training ' + game)
    if game_size >= 1000:
        source_game.train(NUM_DYAD_TRAIN_EPOCHS * 10)
    else:
        source_game.train(NUM_DYAD_TRAIN_EPOCHS)

    # 2. train each pair of refer_games
    for target_game  in TARGET_GAMES:
        direction = game + '_to_' + target_game
        print('start running experiment:' + direction)
        log_file = f"{core.get_opts().generalisation_path}/{direction}/{str(core.get_opts().random_seed)}.txt"

        transfer_game_to_game(
            source_game=source_game,
            target_game_name=target_game,
            log_file=log_file,
            num_train_epochs=NUM_RETRAIN_EPOCHS,
        )

    del source_game