#!/usr/bin/env python3
import os
import argparse
import time
import pickle
import random
from pathlib import Path

import chess
import chess.pgn
import chess.engine
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ChessDataset
from model import ChessModel
from auxiliary_func import board_to_matrix, encode_moves

def load_move_maps(move_map_path):
    with open(move_map_path, "rb") as f:
        move_to_int = pickle.load(f)
    int_to_move = {v: k for k, v in move_to_int.items()}
    return move_to_int, int_to_move

def prepare_tensor(board):
    mat = board_to_matrix(board)
    t = torch.tensor(mat, dtype=torch.float32).unsqueeze(0)
    return t

def model_move(board, model, device, int_to_move, move_to_int, temperature=1.0, topk=None, deterministic=False):
    X = prepare_tensor(board).to(device)
    with torch.no_grad():
        logits = model(X).squeeze(0)
    probs = torch.softmax(logits / (temperature if temperature > 0 else 1e-6), dim=0).cpu().numpy()
    legal = {m.uci(): m for m in board.legal_moves}
    candidates = []
    for idx, mv in int_to_move.items():
        if mv in legal:
            candidates.append((idx, probs[idx], mv))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    if deterministic or topk is None:
        return legal[candidates[0][2]]
    if topk == 1:
        return legal[candidates[0][2]]
    k = min(topk, len(candidates))
    top = candidates[:k]
    weights = np.array([t[1] for t in top], dtype=np.float64)
    if weights.sum() == 0:
        chosen = random.choice(top)
    else:
        weights = weights / weights.sum()
        chosen = top[np.random.choice(len(top), p=weights)]
    return legal[chosen[2]]

def engine_move(board, engine, limit=None):
    try:
        res = engine.play(board, limit)
        if res is None or res.move is None:
            return None
        return res.move
    except Exception:
        return None

def play_game(model, device, int_to_move, move_to_int, engine, engine_limit, side_is_model_white, max_moves, temperature, topk, deterministic):
    game = chess.pgn.Game()
    node = game
    board = chess.Board()
    moves_made = 0
    while not board.is_game_over() and moves_made < max_moves:
        if board.turn == chess.WHITE:
            is_model_to_move = side_is_model_white
        else:
            is_model_to_move = not side_is_model_white
        if is_model_to_move:
            mv = model_move(board, model, device, int_to_move, move_to_int, temperature=temperature, topk=topk, deterministic=deterministic)
            if mv is None:
                break
        else:
            mv = engine_move(board, engine, engine_limit)
            if mv is None:
                break
            if mv.uci() not in move_to_int:
                break
        board.push(mv)
        node = node.add_main_variation(mv)
        moves_made += 1
    game.headers["Result"] = board.result()
    return game, board

def extract_examples_from_game(game):
    X = []
    ys = []
    board = game.board()
    for mv in game.mainline_moves():
        X.append(board_to_matrix(board))
        ys.append(mv.uci())
        board.push(mv)
    X = np.array(X, dtype=np.float32)
    return X, ys

def encode_dataset(X, y_strs, move_to_int):
    encoded = []
    valid_idxs = []
    labels = []
    for i, mv in enumerate(y_strs):
        if mv in move_to_int:
            valid_idxs.append(i)
            labels.append(move_to_int[mv])
    if not valid_idxs:
        return None, None
    X_valid = X[valid_idxs]
    y_valid = np.array(labels, dtype=np.int64)
    return X_valid, y_valid

def finetune(model, device, X, y, epochs, batch_size, lr, save_path):
    ds = ChessDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for e in range(epochs):
        total_loss = 0.0
        count = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            count += xb.size(0)
        avg = total_loss / (count if count else 1)
        print(f"Epoch {e+1}/{epochs} loss {avg:.4f}")
    torch.save(model.state_dict(), save_path)
    print("Saved fine-tuned model to", save_path)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="../../models/TORCH_100EPOCHS.pth")
    p.add_argument("--move-map", default="../../models/move_to_int")
    p.add_argument("--stockfish-path", default="stockfish")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--max-moves", type=int, default=200)
    p.add_argument("--engine-time", type=float, default=0.05)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-pgn-dir", default="generated_pgns")
    p.add_argument("--save-model", default="../../models/TORCH_100EPOCHS_finetuned.pth")
    p.add_argument("--play-temperature", type=float, default=1.0)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--train-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    return p.parse_args()

def main():
    args = parse_args()
    move_to_int, int_to_move = load_move_maps(args.move_map)
    device = torch.device(args.device)
    model = ChessModel(num_classes=len(move_to_int))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    os.makedirs(args.out_pgn_dir, exist_ok=True)
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
    engine_limit = chess.engine.Limit(time=args.engine_time)
    all_X = []
    all_y = []
    for g in range(args.games):
        side_white = (g % 2 == 0)
        game, board = play_game(model, device, int_to_move, move_to_int, engine, engine_limit, side_is_model_white=side_white, max_moves=args.max_moves, temperature=args.play_temperature, topk=args.topk, deterministic=args.deterministic)
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"game_{g+1}_{ts}.pgn"
        pgn_path = os.path.join(args.out_pgn_dir, fname)
        with open(pgn_path, "w", encoding="utf-8") as f:
            f.write(str(game))
        Xg, yg = extract_examples_from_game(game)
        X_valid, y_valid = encode_dataset(Xg, yg, move_to_int)
        if X_valid is not None:
            all_X.append(X_valid)
            all_y.append(y_valid)
        print(f"Finished game {g+1}/{args.games} result {board.result()} moves {len(yg)} saved to {pgn_path}")
    engine.close()
    if not all_X:
        print("No valid examples collected. Exiting.")
        return
    X_concat = np.concatenate(all_X, axis=0)
    y_concat = np.concatenate(all_y, axis=0)
    print("Collected examples:", X_concat.shape[0])
    finetune(model, device, X_concat, y_concat, epochs=args.train_epochs, batch_size=args.batch_size, lr=args.lr, save_path=args.save_model)

if __name__ == "__main__":
    main()
