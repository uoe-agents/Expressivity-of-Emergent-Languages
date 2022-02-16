from modules import BaseGame
import egg.core as core
from refer_game import SymbolicReferGame

NUM_DYAD_TRAIN_EPOCHS = 1000
game_size=5000

demo_game = BaseGame()
del demo_game

train_log_file = f"{core.get_opts().generalisation_path}/{game_size}_train/{str(core.get_opts().random_seed)}.txt"
msg_log_path = f"{core.get_opts().generalisation_path}/{game_size}_msgnum/{str(core.get_opts().random_seed)}/"

source_game = SymbolicReferGame(training_log=train_log_file,
                                num_msg_log=msg_log_path,
                                game_size=game_size, 
                                track_compositionality=False, 
                                valid_ratio=0.0
                               )

source_game.train(NUM_DYAD_TRAIN_EPOCHS)