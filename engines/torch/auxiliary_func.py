# auxiliary_func.py
import numpy as np
from chess import Board
import pickle

def board_to_matrix(board: Board):
    """
    Original 13-channel board-to-matrix conversion, including legal moves.
    """
    matrix = np.zeros((13, 8, 8), dtype=np.float32) # Use float32
    piece_map = board.piece_map()

    # Populate first 12 8x8 boards (where pieces are)
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[piece_type + piece_color, row, col] = 1.0 # Use float32

    # Populate the legal moves board (13th 8x8 board)
    try:
        legal_moves = board.legal_moves
        for move in legal_moves:
            to_square = move.to_square
            row_to, col_to = divmod(to_square, 8)
            matrix[12, row_to, col_to] = 1.0 # Use float32
    except Exception:
        pass # Ignore errors on invalid board states

    return matrix

def get_num_classes_from_mapping(mapping_path):
    """ Loads a pickled mapping file and returns its length and the mapping itself. """
    try:
        with open(mapping_path, "rb") as f:
            move_to_int = pickle.load(f)
        return len(move_to_int), move_to_int
    except Exception as e:
        print(f"Error loading map {mapping_path}: {e}")
        return None, None

def create_input_for_nn(games):
    """Creates input tensors for the neural network from a list of games."""
    X = []
    y = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            X.append(board_to_matrix(board))
            y.append(move.uci())
            board.push(move)
    return np.array(X), y

def encode_moves(y, move_to_int):
    """Encodes moves using a pre-existing mapping."""
    encoded_y = [move_to_int[move] for move in y]
    return np.array(encoded_y), move_to_int