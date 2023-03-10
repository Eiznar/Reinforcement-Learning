import numpy as np

class Board():
    def __init__(self):
        self.init_board = np.zeros([6,7]).astype(str)
        self.init_board[self.init_board == "0.0"] = " "
        # player is 0 if first player's turn, 1 if second player's turn.
        self.player = 0
        self.current_board = self.init_board

    def drop_piece(self, column):
        """Drops a piece in the input column.

        Args:
            column (int): Index of the column to drop the piece at.

        Returns:
            _type_: "Invalid move" if the move is invalid, i.e. the column is already full.
        """
        if self.current_board[0, column] != " ":
            return "Invalid move"
        else:
            row = 0
            pos = " "
            # Find the lowest available slot in the column.
            while (pos == " "):
                if row == 6:
                    row += 1
                    break
                pos = self.current_board[row, column]
                row += 1
            if self.player == 0:
                self.current_board[row-2, column] = "O"
                self.player = 1
            elif self.player == 1:
                self.current_board[row-2, column] = "X"
                self.player = 0
    
    def check_winner(self):
        """Check if there is a winner on the current board.

        Returns:
            _type_: _description_
        """
        if self.player == 1:
            for row in range(6):
                for col in range(7):
                    if self.current_board[row, col] != " ":
                        # rows
                        try:
                            if self.current_board[row, col] == "O" and self.current_board[row + 1, col] == "O" and \
                                self.current_board[row + 2, col] == "O" and self.current_board[row + 3, col] == "O":
                                #print("row")
                                return True
                        except IndexError:
                            next
                        # columns
                        try:
                            if self.current_board[row, col] == "O" and self.current_board[row, col + 1] == "O" and \
                                self.current_board[row, col + 2] == "O" and self.current_board[row, col + 3] == "O":
                                #print("col")
                                return True
                        except IndexError:
                            next
                        # \ diagonal
                        try:
                            if self.current_board[row, col] == "O" and self.current_board[row + 1, col + 1] == "O" and \
                                self.current_board[row + 2, col + 2] == "O" and self.current_board[row + 3, col + 3] == "O":
                                #print("\\")
                                return True
                        except IndexError:
                            next
                        # / diagonal
                        try:
                            if self.current_board[row, col] == "O" and self.current_board[row + 1, col - 1] == "O" and \
                                self.current_board[row + 2, col - 2] == "O" and self.current_board[row + 3, col - 3] == "O"\
                                and (col-3) >= 0:
                                #print("/")
                                return True
                        except IndexError:
                            next
        if self.player == 0:
            for row in range(6):
                for col in range(7):
                    if self.current_board[row, col] != " ":
                        # rows
                        try:
                            if self.current_board[row, col] == "X" and self.current_board[row + 1, col] == "X" and \
                                self.current_board[row + 2, col] == "X" and self.current_board[row + 3, col] == "X":
                                return True
                        except IndexError:
                            next
                        # columns
                        try:
                            if self.current_board[row, col] == "X" and self.current_board[row, col + 1] == "X" and \
                                self.current_board[row, col + 2] == "X" and self.current_board[row, col + 3] == "X":
                                return True
                        except IndexError:
                            next
                        # \ diagonal
                        try:
                            if self.current_board[row, col] == "X" and self.current_board[row + 1, col + 1] == "X" and \
                                self.current_board[row + 2, col + 2] == "X" and self.current_board[row + 3, col + 3] == "X":
                                return True
                        except IndexError:
                            next
                        # / diagonal
                        try:
                            if self.current_board[row, col] == "X" and self.current_board[row + 1, col - 1] == "X" and \
                                self.current_board[row + 2, col - 2] == "X" and self.current_board[row + 3, col - 3] == "X"\
                                and (col-3) >= 0:
                                return True
                        except IndexError:
                            next

    def actions(self): # returns all possible moves
        """Checks all the possible moves.

        Returns:
            list[int]: list of valid choices of column for next moves.
        """
        acts = []
        for col in range(7):
            if self.current_board[0, col] == " ":
                acts.append(col)
        return acts