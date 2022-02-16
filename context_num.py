#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from modules import BaseGame
import numpy as np
import egg.core as core
from egg.core.util import move_to

from refer_game import SymbolicReferGame
from utils import create_dir_for_file, parse_game_name


NUM_DYAD_TRAIN_EPOCHS = 1000 # number of epochs for dyads to converge
NUM_RETRAIN_EPOCHS =  1500 # number of epochs for retraining listener
SOURCE_GAMES = [
                'refer2',
                'refer10',
                'refer100',
                'refer1000',
                'refer2500',
                'refer5000',
                'refer7500',
                'refer10000',
               ]
TARGET_GAMES = [
                'refer2',
                'refer10',
                'refer100',
                'refer1000',
                'refer2500',
                'refer5000',
                'refer7500',
                'refer10000',
               ]



def transfer_game_to_game(
        source_game: SymbolicReferGame,
        target_game_name:str,
        log_file: str,
        num_train_epochs:int=5000
    ) -> None:
    source_game.game.eval()
    
    # reinitialise a dyad in target_game
    target_game_type = parse_game_name(target_game_name)[0]
    target_game = None

    if target_game_type == 'refer':
        game_size = parse_game_name(target_game_name)[1]
        target_game = SymbolicReferGame(
                            training_log='./log/refer_train_temp.txt', 
                            game_size=game_size,
                            track_compositionality=False
                        )
    else:
        raise ValueError('Unrecognised game type:' + target_game_type)

    optimizer = core.build_optimizer(target_game.game.receiver.parameters())
    train_loss = []
    test_loss = []

    for i in range(num_train_epochs):
        acc_list = []
        for batch_idx, sample in enumerate(target_game.train_loader):
            optimizer.zero_grad()
            target = move_to(sample[0], source_game.trainer.device)
            label = move_to(sample[1], source_game.trainer.device)
            candidate = move_to(sample[2], source_game.trainer.device) if target_game_type == 'refer' else None

            msg,_ = source_game.sender(target)
            if target_game_type == 'refer':
                rec_out = target_game.receiver(msg.detach(), candidate)
                loss, acc = target_game.contrastive_loss(target, msg, candidate, rec_out, label)
                acc_list.append(acc['acc'].item())
            else:
                raise ValueError('Unrecognised game type:' + target_game_type)
            
            loss.sum().backward()
            optimizer.step()
        
        print('train loss:', np.mean(acc_list))    
        train_loss.append(np.mean(acc_list))

        acc_list = []
        for _, sample in enumerate(target_game.test_loader):
            target_game.game.eval() 
            target = move_to(sample[0], source_game.trainer.device)
            label = move_to(sample[1], source_game.trainer.device)
            candidate = move_to(sample[2], source_game.trainer.device) if target_game_type == 'refer' else None

            msg,_ = source_game.sender(target)
            if target_game_type == 'refer':
                rec_out = target_game.receiver(msg.detach(), candidate)
                loss, acc = target_game.contrastive_loss(target, msg, candidate, rec_out, label)
                acc_list.append(acc['acc'].item())
            else:
                raise ValueError('Unrecognised game type:' + target_game_type)

        print('test loss:', np.mean(acc_list))
        test_loss.append(np.mean(acc_list))

    create_dir_for_file(log_file)

    with open(log_file, 'w') as f:
        for i in range(len(train_loss)):
            print(str(train_loss[i]) + ',' + str(test_loss[i]), file=f)

    del target_game
    

if __name__ == '__main__':
    demo_game = BaseGame()
    del demo_game

    for game in SOURCE_GAMES:
        # 1. train the dyads to obtain the emergent languages
        source_game = None
        train_log_file = core.get_opts().generalisation_path + '/' + game + '_train' \
                            + '/' + str(core.get_opts().random_seed) + '.txt'
        msg_log_path = core.get_opts().generalisation_path + '/' + game + '_msgnum' \
                            + '/' + str(core.get_opts().random_seed) + '/' # + '.txt'
        game_type = parse_game_name(game)[0]
        if game_type == 'refer':
            game_size = parse_game_name(game)[1]
            source_game = SymbolicReferGame(training_log=train_log_file,
                                            num_msg_log=msg_log_path,
                                            game_size=game_size, 
                                            track_compositionality=False, 
                                            valid_ratio=0.0
                                            )
            print('start training ' + game)
            if game_size >= 1000:
                source_game.train(NUM_DYAD_TRAIN_EPOCHS * 10)
            else:
                source_game.train(NUM_DYAD_TRAIN_EPOCHS)
        else:
            raise ValueError(f"Unrecognised game type: {game_type}!")

        # 2. train each pair of refer_games
        for target_game  in TARGET_GAMES:
            direction = game + '_to_' + target_game
            print('start running experiment:' + direction)
            log_file = core.get_opts().generalisation_path + '/' + direction + '/' + \
                 str(core.get_opts().random_seed) + '.txt'

            transfer_game_to_game(
                source_game=source_game,
                target_game_name=target_game,
                log_file=log_file,
                num_train_epochs=NUM_RETRAIN_EPOCHS,
            )

        del source_game
