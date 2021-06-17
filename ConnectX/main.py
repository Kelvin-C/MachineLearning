from typing import List
import numpy as np
import random as ran
from kaggle_environments import make, evaluate
import datetime as dt
import copy

# Setup a tictactoe environment.
env = make("connectx", debug=True)


def board_to_matrix(board: List, columns: int) -> np.ndarray:
    """ Converts the board into a 2D matrix. """
    return np.array(board).reshape((-1, columns))


def get_allowed_actions(observation, configuration) -> List[int]:
    """ Returns all actions that are allowed. """
    return [c for c in range(configuration.columns) if observation.board[c] == 0]


def training_agent(observation, configuration) -> int:
    return ran.choice(get_allowed_actions(observation, configuration))


def reset_trainer(trainer, cloned_env, real_steps):
    trainer.reset()
    cloned_env.steps = real_steps
    return real_steps[-1][0].observation


def playing_agent(real_observation, configuration):

    # Clone current game environment
    # and store the steps so we can reset the trainer
    cloned_env = env.clone()
    real_steps = copy.deepcopy(env.steps)
    for i in range(len(real_steps)):
        steps = real_steps[i]
        for j in range(len(steps)):
            if steps[j].observation == real_observation:
                print(f'{i}, {j}')

    # Create trainer
    trainer = cloned_env.train([None, "random"])
    observation = trainer.reset()

    start_time = dt.datetime.now()

    # Stores the score by each action
    scores_by_action = {action: [] for action in range(configuration.columns)}

    # Perform training
    simulation_count_per_real_action = 10
    simulation_count = 0
    for next_real_action in get_allowed_actions(observation, configuration):
        training_action = next_real_action
        for _ in range(simulation_count_per_real_action):
            reward = None
            step_count = 0
            while True:
                observation, reward, done, info = trainer.step(training_action)
                if done:
                    # Amplify the reward if the game ends on the first step.
                    # We want to stop the enemy if they win, and we want to use this step
                    # if we win
                    if step_count == 0:
                        reward = reward * 9999999
                    break

                training_action = training_agent(observation, configuration)
                step_count += 1

            scores_by_action[next_real_action].append(reward)
            observation2 = reset_trainer(trainer, cloned_env, real_steps)
            print(len(real_steps))
            simulation_count += 1

    # Get action from the results
    best_action = None
    best_avg = None
    for action, results in scores_by_action.items():
        if len(results) == 0:
            continue

        # Average the results
        avg = np.mean(results)
        if best_avg is None or avg > best_avg:
            best_action = action
            best_avg = avg

    # Calculate training time
    train_time = dt.datetime.now() - start_time
    print(f'Training time taken: {train_time.microseconds / 1000}ms.')
    print(f'Average training time taken per game: {train_time.microseconds / 1000 / simulation_count}ms.')

    print(f'My action: {best_action}')
    return best_action


def test_agent(observation, configuration):
    print('REAL')
    newenv = env.clone()
    steps = newenv.steps
    print(len(newenv.steps))
    trainer = newenv.train([None, "random"])
    obs = trainer.reset()
    print('CLONED')
    newenv.steps = steps
    print(len(newenv.steps))
    trainer.step(0)
    print(len(newenv.steps))
    print(newenv.steps[-1][0].observation.board)

    return 0


env.run([playing_agent, 'negamax'])
env.render()
