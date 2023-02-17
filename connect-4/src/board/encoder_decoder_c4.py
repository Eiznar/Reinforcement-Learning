import numpy as np
from .connect4board import Board

def encode_board(board):
    """The raw Connect4 board is encoded into a 6 by 7 by 3 matrix of 1’s and 0’s
    before input into the neural net, where the 3 channels each of board dimensions
    6 by 7 encode the presence of “X”, “O” (1 being present and 0 being empty), and
    player to move (0 being “O” and 1 being “X”), respectively.

    Args:
        board (Board): board object that is encoded as the current state of the game.

    Returns:
        _type_: 6 by 7 by 3 matrix of 1's and 0's used as input into the neural network.
        Channel 0 is 1's at the positions of "O", channel 1 is 1's at the position of "X",
        channel 3 is full of  0 if player 0 is playing and full of 1 if player 1
        is playing.
    """
    board_state = board.current_board
    encoded = np.zeros([6, 7, 3]).astype(int)
    encoder_dict = {"O": 0, "X": 1}
    for row in range(6):
        for col in range(7):
            if board_state[row, col] != " ":
                encoded[row, col, encoder_dict[board_state[row, col]]] = 1
    if board.player == 1:
        encoded[:, :, 2] = 1  # player to move
    return encoded

def decode_board(encoded):
    decoded = np.zeros([6,7]).astype(str)
    decoded[decoded == "0.0"] = " "
    decoder_dict = {0:"O", 1:"X"}
    for row in range(6):
        for col in range(7):
            for k in range(2):
                if encoded[row, col, k] == 1:
                    decoded[row, col] = decoder_dict[k]
    current_board = Board()
    current_board.current_board = decoded
    current_board.player = encoded[0,0,2]
    return current_board
