import os
import numpy as np  # type: ignore
import time
import torch
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from chess import pgn  # type: ignore
from tqdm import tqdm  # type: ignore
import multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler  # type: ignore


# Data preprocessing
# Load data
def load_pgn(file_path):
    """Load PGN file and return games"""
    games = []
    with open(file_path, 'r') as pgn_file:
        while True:
            game = pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    return games


def load_and_process_pgn(file_path):
    """Load PGN file and extract board states and moves as tuples"""
    games = load_pgn(file_path)
    positions = []
    
    for game in games:
        try:
            board = game.board()
            for move in game.mainline_moves():
                # Store board matrix and move UCI string
                from auxiliary_func import board_to_matrix
                positions.append((board_to_matrix(board), move.uci()))
                board.push(move)
        except:
            continue
    return positions


def main():
    print("Loading PGN files...")
    
    # Get the absolute path to the data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "..", "data")
    data_dir = os.path.normpath(data_dir)
    
    print(f"Looking for PGN files in: {data_dir}")
    files = [file for file in os.listdir(data_dir) if file.endswith(".pgn")]
    LIMIT_OF_FILES = min(len(files), 28)

    # Parallel loading with multiprocessing
    num_workers = min(mp.cpu_count(), 8)  # Use up to 8 workers
    print(f"Using {num_workers} workers for parallel PGN loading and processing...")

    all_positions = []
    with mp.Pool(processes=num_workers) as pool:
        file_paths = [os.path.join(data_dir, file) for file in files[:LIMIT_OF_FILES]]
        results = list(tqdm(pool.imap(load_and_process_pgn, file_paths), total=len(file_paths), desc="Loading files"))
        for result in results:
            all_positions.extend(result)

    print(f"TOTAL POSITIONS EXTRACTED: {len(all_positions)}")


    # Convert data into tensors
    print("Separating features and labels...")
    X = np.array([pos[0] for pos in all_positions])
    y = [pos[1] for pos in all_positions]  # Keep as list of UCI strings

    print(f"NUMBER OF SAMPLES: {len(y)}")

    # Limit dataset size
    X = X[0:2500000]
    y = y[0:2500000]

    # Create move encoding
    print("Creating move encoding...")
    unique_moves = sorted(set(y))
    move_to_int = {move: i for i, move in enumerate(unique_moves)}
    
    # Encode moves
    y = np.array([move_to_int[move] for move in y])
    num_classes = len(move_to_int)
    
    print(f"Number of unique moves: {num_classes}")

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)


    # Preliminary actions
    from dataset import ChessDataset
    from model import ChessModel

    # Optimized DataLoader settings
    batch_size = 256  # Increased from 64 for better GPU utilization
    num_dataloader_workers = min(4, mp.cpu_count())  # Parallel data loading
    pin_memory = torch.cuda.is_available()  # Faster GPU transfers

    # Create Dataset and DataLoader
    dataset = ChessDataset(X, y)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_dataloader_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_dataloader_workers > 0 else False
    )

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB')

    # Model Initialization
    model = ChessModel(num_classes=num_classes).to(device)

    # Enable torch.compile for PyTorch 2.0+ (significant speedup)
    if hasattr(torch, 'compile'):
        print("Using torch.compile for optimization...")
        model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()

    # Training configuration
    num_epochs = 100

    # Use AdamW (improved version of Adam)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.001,
        epochs=num_epochs,
        steps_per_epoch=len(dataloader)
    )

    # Mixed precision training scaler
    scaler = GradScaler()


    # Training
    print("\nStarting optimized training...")
    print(f"Batch size: {batch_size}, DataLoader workers: {num_dataloader_workers}")
    print(f"Mixed precision: {'Enabled' if torch.cuda.is_available() else 'Disabled (CPU mode)'}")

    # Accumulate loss less frequently for speed
    loss_accumulation_steps = 10

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        batch_count = 0
        
        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Mixed precision training
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Update learning rate
            scheduler.step()
            
            # Accumulate loss less frequently
            batch_count += 1
            if batch_count % loss_accumulation_steps == 0:
                running_loss += loss.item() * loss_accumulation_steps
        
        end_time = time.time()
        epoch_time = end_time - start_time
        minutes: int = int(epoch_time // 60)
        seconds: int = int(epoch_time) - minutes * 60
        avg_loss = running_loss / (len(dataloader) // loss_accumulation_steps)
        current_lr = scheduler.get_last_lr()[0]
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, Time: {minutes}m{seconds}s')
        
        if torch.cuda.is_available():
            print(f'GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB / {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB peak')


    # Save the model and mapping
    print("\nSaving model and mappings...")
    
    # Get absolute path to models directory
    models_dir = os.path.join(script_dir, "..", "..", "models")
    models_dir = os.path.normpath(models_dir)
    
    # Save the model
    model_path = os.path.join(models_dir, "TORCH_100EPOCHS.pth")
    torch.save(model.state_dict(), model_path)

    import pickle

    mapping_path = os.path.join(models_dir, "heavy_move_to_int")
    with open(mapping_path, "wb") as file:
        pickle.dump(move_to_int, file)

    print(f"Training complete! Model saved to {model_path}")


if __name__ == '__main__':
    # Required for Windows multiprocessing
    mp.freeze_support()
    main()
