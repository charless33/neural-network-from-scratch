# neural-network-from-scratch

A neural network library built from scratch in NumPy, with an interactive digit recognizer using Pygame.

## Background

This project was done in 2023.

It started as an attempt to understand how neural networks actually work. The real beginning was actually an Excel sheet where I slowly recursed the process of backpropagation in a 2-neuron network trying to understand how it came to "learn". Once I was satisfied and convinced I had an idea of what was going on at small scale, I was able to slowly build my first example using only NumPy for math and handling of matrices. I was studying linear algebra in engineering at the time which helped. Once I built my first network I instantly implemented it in a digit recognizer I made using Pygame to test the network against the MNIST handwritten digits dataset, you can see the final result in `digit_recognizer.py`. Once I was done with that project I decided to make the network building algorithm I had created reusable, so I refactored the core logic into a reusable library (`netbuilder.py`). As I got more educated about optimization algorithms I realized there were a lot more possibilities for improvement, so after a couple weeks of reading I made `netbuilderpro.py` which implements the Adam optimizer.

## Files

| File | Description |
|------|-------------|
| `digit_recognizer.py` | Trains a neural network on MNIST and opens an interactive Pygame window where you can draw digits and have them recognized in real time |
| `netbuilder.py` | Reusable neural network library with L2 regularization |
| `netbuilderpro.py` | Extended version of netbuilder with Adam optimizer |
| `Successful_use_of_netbuilder.py` | Example usage of the netbuilder library |

## How to use netbuilder

```python
import netbuilder as nt

# Define architecture as a list of layer sizes
# [2, 8, 8, 1] = 2 inputs, two hidden layers of 8 neurons, 1 output
architecture = [2, 8, 8, 1]

# Activation function per layer (0 = none, 1 = sigmoid, 2 = ReLU)
activation = [2, 2, 0]

network = nt.createNetwork(architecture, activation)
network.propagate(inputs)
network.backpropagate(expected_outputs, learning_rate)

print(network.output)
probabilities = network.softmax(network.output)
```

For netbuilderpro it's the same but backpropagate takes extra parameters:

```python
network.backpropagate(expected_outputs, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
```

## digit_recognizer.py controls

| Key | Action |
|-----|--------|
| Left mouse | Draw |
| S | Recognize the drawn digit |
| C | Clear the canvas |
| K | Display a random MNIST digit |
| A | Manually correct and retrain on current example |
| L | Save weights to file |
| Q | Quit |

At startup the program asks whether to train from scratch, continue from a saved file, or load weights without training.
