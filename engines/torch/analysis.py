import chess
import chess.pgn
import torch
import numpy as np
import pickle
from model import ChessModel
from auxiliary_func import board_to_matrix
import argparse

# --- Model and Helpers from predict.ipynb ---

def prepare_input(board: chess.Board):
    matrix = board_to_matrix(board)
    X_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
    return X_tensor

def get_move_probabilities(board: chess.Board, model, device):
    X_tensor = prepare_input(board).to(device)
    with torch.no_grad():
        logits = model(X_tensor)
    logits = logits.squeeze(0)
    probabilities = torch.softmax(logits, dim=0).cpu().numpy()
    return probabilities

# --- Analysis Function ---

def analyze_game(pgn_path: str, player_name: str, model, int_to_move, device, min_error_threshold: float = 0.0):
    game = chess.pgn.read_game(open(pgn_path))
    if not game:
        print("Could not read game from PGN.")
        return

    board = game.board()
    mistakes = []

    print(f"Analyzing game for player: {player_name}")

    for node in game.mainline():
        move = node.move
        if board.turn == (chess.WHITE if game.headers["White"].lower() == player_name.lower() else chess.BLACK):
            # Get model's evaluation
            probabilities = get_move_probabilities(board, model, device)
            legal_moves = {m.uci(): m for m in board.legal_moves}
            
            # Find the best move according to the model
            sorted_indices = np.argsort(probabilities)[::-1]
            best_engine_move = None
            for move_index in sorted_indices:
                move_uci = int_to_move.get(move_index)
                if move_uci in legal_moves:
                    best_engine_move = move_uci
                    break

            user_move_uci = move.uci()

            if user_move_uci != best_engine_move:
                user_move_prob = probabilities[np.where([int_to_move.get(i) == user_move_uci for i in range(len(int_to_move))])[0][0]]
                best_move_prob = probabilities[np.where([int_to_move.get(i) == best_engine_move for i in range(len(int_to_move))])[0][0]]
                
                # Quantify the mistake by the drop in probability
                error = best_move_prob - user_move_prob
                if error > min_error_threshold:
                    mistakes.append({
                        'fen': board.fen(),
                        'user_move': user_move_uci,
                        'engine_move': best_engine_move,
                        'error': error
                    })

        board.push(move)

    # Sort mistakes by severity
    mistakes.sort(key=lambda x: x['error'], reverse=True)

    print("\n--- Top 3 Mistakes ---")
    for i, mistake in enumerate(mistakes[:3]):
        print(f"\n#{i+1} Biggest Mistake:")
        print(f"  Position: {mistake['fen']}")
        print(f"  Your move: {mistake['user_move']}")
        print(f"  Engine suggestion: {mistake['engine_move']}")
        print(f"  (Error score: {mistake['error']:.4f})")

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a chess game to find mistakes.")
    parser.add_argument("pgn_file", help="Path to the PGN file of the game to analyze.")
    parser.add_argument("player_name", help="The name of the player to analyze (case-insensitive).")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Minimum error score to consider a move a mistake (default: 0.05).")
    args = parser.parse_args()

    # Load model and mappings
    with open("../../models/move_to_int", "rb") as file:
        move_to_int = pickle.load(file)
    
    int_to_move = {v: k for k, v in move_to_int.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ChessModel(num_classes=len(move_to_int))
    # NOTE: Loading on CPU. Change if you have a GPU.
    model.load_state_dict(torch.load("../../models/TORCH_100EPOCHS.pth", map_location=device))
    model.to(device)
    model.eval()

    analyze_game(args.pgn_file, args.player_name, model, int_to_move, device, args.threshold)