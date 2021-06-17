from connectx import ConnectXTrainer, TreeMap, ConnectX
import datetime as dt
import numpy as np
from kaggle_environments import make


def board_to_matrix(board, column_count) -> np.ndarray:
    """ Converts the board into a 2D matrix. """
    return np.array(board).reshape((-1, column_count))


def get_allowed_actions(game: ConnectX):
    """ Returns all actions that are allowed. """
    column_count = game.board.shape[1]
    return [c for c in range(column_count) if game.board[0][c] == 0]


def random_choice(game: ConnectX):
    return np.random.choice(get_allowed_actions(game))


def play_game(game: ConnectX, first_action: int) -> int:
    """ Plays a game until the end and returns the reward. """

    done, _reward = False, None
    _player = 1
    _action = None
    is_first_action = True
    while not done:
        try:
            _action = first_action if is_first_action else random_choice(game)
            done, _reward = game.step(_player, _action)
            is_first_action = False
        except Exception as e:
            print(f'Failed to step. Action: {_action}')
            print(game.board)
            raise
        _player = 1 if _player == 2 else 2
    return _reward


trainer = ConnectXTrainer(4)
global_actions_tree_root = TreeMap()
new_game = True


def player(observation, configuration):
    global global_actions_tree_root, new_game
    actions_tree_root: TreeMap = global_actions_tree_root

    board = board_to_matrix(observation.board, configuration.columns)

    # Update the trainer using the new board
    try:
        trainer.update_game(board)
    except Exception as e:
        print(f'Failed to update the trainer game')
        print(board)
        raise

    # Update the actions tree using the enemy's action
    if not new_game:
        enemy_move = trainer.original_game.actions[-1]
        print(f'Enemy played {enemy_move}')
        try:
            if enemy_move in actions_tree_root.branches:
                actions_tree_root = actions_tree_root.branches[enemy_move]
            else:
                actions_tree_root = actions_tree_root.add(enemy_move, [])
        except Exception as e:
            print('Failed to move tree down a level using the enemy move')
            print(get_allowed_actions(trainer.training_game))
            print(actions_tree_root)
            print(enemy_move)
            print(board)
            raise
    else:
        new_game = False

    start_time = dt.datetime.now()
    simulation_count = 100
    for first_action in get_allowed_actions(trainer.training_game):
        for _ in range(simulation_count):
            reward = play_game(trainer.training_game, first_action)
            branch = actions_tree_root
            for action_index in range(len(trainer.training_game.actions)):
                # Check if this action is part of the original game or is part of the training.
                # We only want to store actions from training.
                if action_index < len(trainer.original_game.actions):
                    continue

                # Make a list of rewards as the node value
                action = trainer.training_game.actions[action_index]
                try:
                    branch = branch.add_or_update_branch(action, [reward], lambda rewards: [*rewards, reward])
                except Exception as e:
                    print(f'Failed to add or update branch for action {action}.')
                    print(f'Reward: {reward}')
                    raise
            try:
                trainer.reset_training_game()
            except Exception as e:
                print('Failed to reset the trainer')
                raise

    # Get the best action from the tree by finding the best mean average
    best_action = max(actions_tree_root.branches, key=lambda k: np.mean(actions_tree_root.branches[k].value))
    print(f'Playing action: {best_action}')

    # print(f'PARENT: {actions_tree_root.key} VALUES {actions_tree_root}')
    # for key, tree in actions_tree_root.branches.items():
    #     print(f'KEY: {key} - VALUES {tree}')

    # Since we've got the best action, we can move the tree down 1 level
    global_actions_tree_root = actions_tree_root.branches[best_action]

    # Calculate training time
    end_time = dt.datetime.now()
    train_time = end_time - start_time
    print(f'Training time taken: {train_time.total_seconds() * 1000}ms.')
    print(f'Average training time taken per game: {train_time.total_seconds() * 1000 / simulation_count}ms.')

    return int(best_action)


def get_ram_usage() -> int:
    import os, psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


env = make("connectx", debug=True)
env.run([player, 'negamax'])
print(f'Process uses {get_ram_usage() / 1_000_000} MB of RAM')
env.render()
