# Genetic-Algorithms-and-Evolutionary-Optimization
## Overview
This project implements a **Genetic Algorithm (GA)** to train and optimize a neural network for solving the **CartPole-v0 environment** from OpenAI Gym.
Instead of using traditional neural network training methods such as backpropagation and gradient descent, this project uses evolutionary optimization where neural network weights evolve through generations using selection, crossover, and mutation.
The goal is to find an optimized set of neural network weights that enables the CartPole agent to achieve higher fitness scores.

## Project Features
- Genetic Algorithm based neural network optimization
- Evolutionary training without gradient descent
- Population-based learning
- Fitness evaluation using CartPole environment
- Selection of best-performing solutions
- Crossover between neural network weights
- Mutation for exploration
- Fitness improvement visualization over generations

## Technologies Used
- Python 3
- NumPy
- OpenAI Gym
- Matplotlib

## How It Works

The project contains two main components:

### 1. Neural Network
The `NeuralNet` class creates a simple neural network that controls the CartPole agent.
The network:
- Takes CartPole environment observations as input
- Processes data through hidden layers
- Produces actions using output probabilities
- Uses activation functions:
  - ReLU
  - Sigmoid
  - Softmax

Workflow:
Environment Observation
|
v
Neural Network
|
v
Action
|
v
CartPole Reward

## 2. Genetic Algorithm

The `GA` class evolves neural network parameters.

Each individual in the population represents a neural network with different weights.

Evolution process:

Initial Population
|
Fitness Evaluation
|
Select Best Networks
|
Crossover
|
Mutation
|
New Generation
|
Improved Agent

## Genetic Algorithm Parameters

The training process uses:

```python
Population Size = 15

Number of Generations = 100

Mutation Rate = 0.5

**Installation**

Clone the repository:
git clone https://github.com/Saanvi-Mehra/Genetic-Algorithms-and-Evolutionary-Optimization-.git

Navigate to the project:
cd Genetic-Algorithms-and-Evolutionary-Optimization-

Install dependencies:
pip install numpy matplotlib gym

Running the Project

Execute:

python GenAlgo.py


Output

The program generates:

Fitness Graph

A graph showing improvement of the best fitness score across generations.

Example:

Code Components
NeuralNet Class

Responsible for:

Neural network creation
Weight initialization
Forward propagation
Running CartPole simulations
Fitness calculation
GA Class

Responsible for:

Maintaining population
Selecting best candidates
Crossover operation
Mutation operation
Evolution across generations
Trainer

The training pipeline:

Create Population
        |
Evaluate Fitness
        |
Run Genetic Algorithm
        |
Return Best Weights
        |
Test Agent

## Project Structure

