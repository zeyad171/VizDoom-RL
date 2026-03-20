# VizDoom Reinforcement Learning Agent 🔫👻

A deep reinforcement learning project focused on training an AI agent to navigate, survive, and succeed in the classic FPS environment **Doom** using the ViZDoom environment and Proximal Policy Optimization (PPO).

## Overview ✨
This repository demonstrates a complete deep reinforcement learning pipeline designed to parse raw pixel data, extract features, and learn optimal action policies through simulation. 

The agent learns entirely from visual inputs (grayscale screenshots) and a reward signal, progressively discovering strategies like shooting targets and optimizing its ammunition.

### Architecture Highlights
- **Environment**: Custom Gymnasium wrapper around ViZDoom.
- **Visual Preprocessing**: Frame resolution scaling to 160x100 and grayscale conversion utilizing OpenCV to increase training efficiency.
- **Algorithm**: `PPO` (Proximal Policy Optimization) imported from Stable-Baselines3.
- **Policy Network**: `CnnPolicy` (Convolutional Neural Network) extracting spatial features directly from the raw game frames.

## Installation ⚙️

Ensure you have Python 3.8+ installed. 
```bash
git clone https://github.com/zeyad171/VizDoom-RL.git
cd VizDoom-RL
pip install -r requirements.txt
```
*(Note: ViZDoom may require specific C++ build tools depending on your OS. Check the official ViZDoom installation guide if you face build errors).*

## Usage 🚀

### 1. Training the Agent
To start the training loop from scratch, execute:
```bash
python train.py
```
This script handles building the environment, spawning the PPO agent, and logging the models to local checkpoint directories. Logs for TensorBoard tracking will be dumped into the `./logs` folder.

### 2. Evaluating and Testing
To view a rendering of exactly how the agent behaves after training:
```bash
python test.py
```
This will visually pop up the game window, load the compiled `best_model_final.zip` (if present), and run statistical evaluations before spawning visual runs. *(Note: Trained models are heavy and intentionally omitted from this repo. You must train the model locally prior to evaluating it!)*

## Project Structure 📂
```bash
VizDoom-RL/
├── ViZDoom-master/             # ViZDoom source scenarios and levels
├── VizDoom-Basic-Tutorial.ipynb# Core Jupyter prototype
├── train.py                    # Extracted training pipeline script
├── test.py                     # Extracted model testing/renderer script
├── requirements.txt            # Python dependencies
├── .gitignore                  # Ignoring massive local logs/models
└── README.md                   # Documentation
```
