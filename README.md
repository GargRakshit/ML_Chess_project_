# AI-Powered Chess Engine using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)

A deep learning-based chess engine that uses Convolutional Neural Networks (CNNs) to predict optimal moves. This project implements parallel models in both TensorFlow and PyTorch, featuring explainable AI capabilities through LIME (Local Interpretable Model-agnostic Explanations).

## 🎯 Project Overview

This AI chess engine learns to play chess by analyzing patterns from millions of games, rather than relying on handcrafted evaluation functions. The system achieves approximately *1500 ELO* performance during opening and middlegame phases.

### Key Features

- 🧠 *Dual Framework Implementation*: Models built in both TensorFlow and PyTorch
- 🎮 *Interactive Gameplay*: Play against the AI with real-time move suggestions
- 🔍 *Explainable AI*: LIME-based visualizations showing which board regions influence predictions
- 📊 *Performance Analysis*: Post-game mistake analysis and move-by-move evaluation

## 🏗 Architecture

The engine uses a 13×8×8 tensor representation:
- *12 channels*: Encode piece positions (6 piece types × 2 colors)
- *1 channel*: Encodes legal move destinations

The PyTorch model consists of:
- 2-layer CNN with 64 and 128 filters
- ReLU activation functions
- Dropout regularization
- Fully connected layers for move classification

## 📈 Performance

### PyTorch Model (TORCH_100EPOCHS)
- *Training*: 100 epochs on GeForce 4050 GPU
- *Final Loss*: 1.3695
- *ELO Rating*: ~1500 (opening/middlegame)
- *Model Size*: 50MB

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, recommended for training)

### Installation

1. *Clone the repository*
   bash
   git clone https://github.com/GargRakshit/ML_Chess_project.git
   cd ML_Chess_project
   

2. *Install dependencies*
   bash
   pip install -r requirements.txt
   

3. *Download training data*
   - Place your PGN files in data/pgn/
   - Recommended dataset: [Lichess Database](https://database.nikonoel.fr/)

### Usage

#### PyTorch Engine
bash
# Navigate to PyTorch engine directory
cd engines/torch/predict
# Follow instructions in the notebook


#### Interactive Gameplay
Use the provided Jupyter notebooks to:
- Play against the AI
- Analyze games move-by-move
- Visualize AI decision-making with LIME explanations

## 🔬 Model Training

### Data Preprocessing
1. Extract games from PGN format
2. Convert to FEN notation
3. Transform into 13-channel tensor representation
4. Split into training/validation sets

### Training Process
- *Optimizer*: Adam
- *Loss Function*: Categorical Crossentropy
- *Augmentation*: Position mirroring, rotation
- *Regularization*: Dropout, early stopping

## 🎓 Explainability

The project integrates LIME to provide transparent insights into AI decision-making:
- Visualizes important board squares
- Highlights pieces influencing move selection
- Generates saliency maps for each prediction

## 🐛 Known Limitations

- Performance degradation after ~20 moves
- Needs blunder detection algorithm for late-game
- Requires large training datasets for optimal performance

## 🚧 Future Improvements

- [ ] Implement deeper architectures with attention mechanisms
- [ ] Integrate endgame tablebases
- [ ] Expand training datasets with diverse game styles

## 📚 Documentation

For detailed technical documentation, methodology, and analysis, see [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md).

## 🙏 Acknowledgments

- [Alexander Serkov](https://github.com/setday) for inspiration of this project
- Lichess database for training data


## 📄 License

Open Source and feel free to contribut

---

*November 2025*



