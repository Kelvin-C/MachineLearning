import numpy as np
import copy


def get_lines(matrix: np.ndarray):
    """ Retrieves rows, columns and diagonals of the matrix. """

    row_count, column_count = matrix.shape

    # all rows
    for row in matrix:
        yield row

    # all columns
    for column_index in range(column_count):
        yield matrix[:, column_index]

    # all diagonals
    row_reverse_matrix = matrix[:, ::-1]
    for k in range(1-row_count, column_count):
        yield np.diag(matrix, k)
        yield np.diag(row_reverse_matrix, k)


def check_line(line: np.ndarray, win_length: int) -> int:
    """
    Check the given line to see if anyone wins.
    :returns -1 if lose, 1 if win and 0 if no one wins or loses.
    """

    # Cannot win if the given line is shorter than
    # the length needed to win
    line_length = line.shape[0]
    if line_length < win_length:
        return 0

    # Check all possible winning lines in the line
    num_of_win_lines_in_line = line_length - win_length + 1
    for i in range(num_of_win_lines_in_line):
        test_line = line[i:i+win_length]
        if all(test_line == 1):
            return 1
        if all(test_line == 2):
            return -1
    return 0


class ConnectX:
    """ A ConnectX game simulation. """

    def __init__(self, winning_length: int):
        self._board = None
        self._winning_length = winning_length
        self._actions = []
        self._player_turn = 1

    @property
    def board(self):
        return self._board

    @property
    def actions(self):
        return self._actions

    def __deepcopy__(self, memodict):
        copied_game = ConnectX(self._winning_length)
        copied_game._board = copy.deepcopy(self._board)
        copied_game._actions = copy.deepcopy(self._actions)
        copied_game._player_turn = self._player_turn
        return copied_game

    def update_board(self, board):
        """ Update the board using the given board. """
        if self._player_turn != 1:
            raise Exception("Can only update the board if the board is player 1's turn.")

        if self._board is not None:
            # Find the new actions by
            # checking the difference between the old and new boards
            board_difference = board - self._board

            # We expect no negative values here. If there's a
            # negative value, then the new board is either in the past
            # or is from a different game.
            if len(np.argwhere(board_difference < 0)) < 0:
                raise Exception('Tried to update the game board using a board from a different game.')

            # Get the location of the players' pieces
            player_1_locations = np.argwhere(board_difference == 1)
            player_2_locations = np.argwhere(board_difference == 2)

            # We expect only 1 move difference between the old board and the new board
            if len(player_1_locations) != 1:
                raise Exception('Expected only 1 move difference for player 1.')
            if len(player_2_locations) != 1:
                raise Exception('Expected only 1 move difference for player 2.')

            # The actions are simply the column indexes of their locations.
            self._actions.append(player_1_locations[0][1])
            self._actions.append(player_2_locations[0][1])

        self._board = board

    def check_winning_state(self) -> (bool, int):
        """
        Check if the game is over and returns the reward.
        :returns true if game is over, and (-1, 0, 1) value for the reward.
        """
        for line in get_lines(self._board):
            result = check_line(line, self._winning_length)
            if result in [-1, 1]:
                return True, result

        return any(self._board[0] != 0), 0

    def step(self, player: int, action: int) -> (bool, int):
        """
        Performs an action, updates the board and checks if the game is over.
        :param player - The player that is playing the action.
        :param action - The column index to play.
        :returns a reward value (-1, 0, 1) and a bool stating whether the game is over.
        """
        # Check if it's the player's turn
        if self._player_turn != player:
            raise Exception(f"Expected player {self._player_turn}'s turn.")

        # Play the action
        row_count = self._board.shape[0]
        played = False
        for row_index in reversed(range(row_count)):
            if self._board[row_index, action] == 0:
                self._board[row_index, action] = player
                played = True
                break

        # If action was not played, then there's no space in that column
        if not played:
            raise Exception(f'Action {action} cannot be played. That column is full.')

        # Complete the action and change the player's turn
        self._actions.append(action)
        self._player_turn = 2 if self._player_turn == 1 else 1
        return self.check_winning_state()


class ConnectXTrainer:
    """ A object used for training ConnectX players. """

    def __init__(self, winning_length: int):
        self._original_game = None
        self._training_game = None
        self._winning_length = winning_length

    @property
    def original_game(self):
        return self._original_game

    @property
    def training_game(self):
        return self._training_game

    def update_game(self, board):
        """ Update the game using the given board. """
        if self._original_game is not None:
            self._original_game.update_board(board)
        else:
            self._original_game = ConnectX(self._winning_length)
            self._original_game.update_board(board)

        self.reset_training_game()

    def reset_training_game(self):
        """ Reset the training game to the original game. """
        self._training_game = copy.deepcopy(self._original_game)


