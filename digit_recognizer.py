import numpy as np
import os
from random import randint
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import pygame
import pickle

np.random.seed(0)

class CreateLayer:
    def __init__(self, nInput, nOutput):
        self.ins = nInput
        self.out = nOutput
        self.weights = np.random.randn(nInput, nOutput) * 0.01
        self.bias = np.zeros((1, nOutput))
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.clip(np.dot(self.inputs, self.weights) + self.bias, -500, 500)
    
    def activate(self, out):
        self.activated = 1 / (1 + np.exp(-out))

    def backward(self, loss):
        self.dvalue = loss
        self.dbias = np.sum(self.dvalue, axis=0, keepdims=True)
        self.dweights = np.dot(np.transpose(self.inputs), self.dvalue)
        self.dinputs = np.dot(self.dvalue, np.transpose(self.weights))

def calcLoss(actual, mine):
    loss = mine - actual
    return loss

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def preprocess_image(image):
    image = np.flipud(image)
    image = np.rot90(image, -1)
    return image

def save_weights(filename):
    with open(filename, 'wb') as f:
        pickle.dump({
            'layer1_weights': layer1.weights,
            'layer1_bias': layer1.bias,
            'layer2_weights': layer2.weights,
            'layer2_bias': layer2.bias,
            'outputLayer_weights': outputLayer.weights,
            'outputLayer_bias': outputLayer.bias
        }, f)

def load_weights(filename):
    global layer1, layer2, outputLayer
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        layer1.weights = data['layer1_weights']
        layer1.bias = data['layer1_bias']
        layer2.weights = data['layer2_weights']
        layer2.bias = data['layer2_bias']
        outputLayer.weights = data['outputLayer_weights']
        outputLayer.bias = data['outputLayer_bias']

#////////////////////////////////////////////////////////////////////////////////////////////////////////////

epochs = 250000
rate = 0.005
acceptability = 0.005

mean = 0
right = 0
old_rights = []
count = 0

layer1 = CreateLayer(784, 64)
layer2 = CreateLayer(64, 64)
outputLayer = CreateLayer(64, 10)

#////////////////////////////////////////////////////////////////////////////////////////////////////////////

print(f"///// Neural Network \\\\\\\\\\\n\nActions:\n (1) Train From Scratch\n (2) Train From File\n (3) Use Network From File\n")
prompt = int(input("choice: "))

while(prompt != 1 and prompt != 2 and prompt != 3):
    prompt = int(input("choice can be only number 1 through 3: "))

if(prompt == 2 or prompt == 3):
    filename = input("filename: ")
    while(not os.path.exists(filename)):
        filename = input("path not found try again: ")
    load_weights(filename)
    print(f"Data extracted from {filename}")

if prompt == 1 or prompt == 2:

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = np.array([preprocess_image(img) for img in x_train])
    x_test = np.array([preprocess_image(img) for img in x_test])

    x_train = (x_train >= 0.5).astype(int)
    x_test = (x_test >= 0.5).astype(int)

    x_train_flat = x_train.reshape(-1, 28*28)
    x_test_flat = x_test.reshape(-1, 28*28)

    datasets = [{'image': img, 'label': label} for img, label in zip(x_train_flat, y_train)] + [{'image': img, 'label': label} for img, label in zip(x_test_flat, y_test)]

    for epoch in range(1, epochs + 1):
        if epoch == 1 or np.mean(loss) < acceptability:
            sample = datasets[randint(0, len(datasets) - 1)]
            inputs = np.array([sample['image']])
            outputs = np.zeros((1, 10))
            outputs[0, sample['label']] = 1

        layer1.forward(inputs)
        layer1.activate(layer1.output)
        layer2.forward(layer1.activated)
        layer2.activate(layer2.output)
        outputLayer.forward(layer2.activated)
        loss = calcLoss(outputs, outputLayer.output)
        outputLayer.backward(loss)
        layer2.backward(outputLayer.dinputs)
        layer1.backward(layer2.dinputs)

        mean += np.sum(loss)
        right += 1 if np.argmax(outputLayer.output) == np.argmax(outputs) else 0
        count += 1

        if(count == 1000):
            print(f"{round(right / count * 1000) / 10}%")
            count = 250
            right = (right/1000)*250

        layer1.weights -= rate * layer1.dweights
        layer1.bias -= rate * layer1.dbias
        layer2.weights -= rate * layer2.dweights
        layer2.bias -= rate * layer2.dbias
        outputLayer.weights -= rate * outputLayer.dweights
        outputLayer.bias -= rate * outputLayer.dbias

#////////////////////////////////////////////////////////////////////////////////////////////////////////////

    print("\n\n/////Training Done\\\\\\\\\\\nWould you like to save the data in a file:\n (1) yes\n (2) no")
    prompt1 = int(input("choice: "))

    while(prompt1 != 1 and prompt1 != 2):
        prompt1 = int(input("choice (1 through 2): "))

    if(prompt1 == 1):
        filename = input("filename: ")
        while(not os.path.exists(filename)):
            filename = input("path not found try again: ")
        save_weights(filename)
        print(f"Trained data saved to {filename}")

#////////////////////////////////////////////////////////////////////////////////////////////////////////////

pygame.init()

WINDOW_SIZE = 280
GRID_SIZE = 28
PIXEL_SIZE = WINDOW_SIZE // GRID_SIZE
BRUSH_SIZE = 2
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption('Draw a Digit')

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)

#////////////////////////////////////////////////////////////////////////////////////////////////////////////

pixel_data = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

def draw_grid():
    for x in range(0, WINDOW_SIZE, PIXEL_SIZE):
        for y in range(0, WINDOW_SIZE, PIXEL_SIZE):
            rect = pygame.Rect(x, y, PIXEL_SIZE, PIXEL_SIZE)
            pygame.draw.rect(screen, GRAY, rect, 1)

def update_display():
    screen.fill(WHITE)
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if pixel_data[x, y] == 1:
                rect = pygame.Rect(x * PIXEL_SIZE, y * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE)
                pygame.draw.rect(screen, BLACK, rect)
    draw_grid()
    pygame.display.flip()

def display_mnist_digit(image_data):
    global pixel_data
    image_data = image_data.reshape((GRID_SIZE, GRID_SIZE))
    pixel_data = image_data
    update_display()

#////////////////////////////////////////////////////////////////////////////////////////////////////////////

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.MOUSEMOTION:
            if pygame.mouse.get_pressed()[0]:
                mouseX, mouseY = pygame.mouse.get_pos()
                x_start = mouseX // PIXEL_SIZE
                y_start = mouseY // PIXEL_SIZE
                
                if 0 <= x_start < GRID_SIZE and 0 <= y_start < GRID_SIZE:
                    for dx in range(-BRUSH_SIZE // 2, BRUSH_SIZE // 2 + 1):
                        for dy in range(-BRUSH_SIZE // 2, BRUSH_SIZE // 2 + 1):
                            x = x_start + dx
                            y = y_start + dy
                            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                                pixel_data[x, y] = 1
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                pixel_data.fill(0)
            elif event.key == pygame.K_s:
                flattened_data = pixel_data.flatten().tolist()
                flattened_data = np.array(flattened_data).reshape(1, 784)
                flattened_data = (flattened_data >= 0.5).astype(int)
                
                layer1.forward(flattened_data)
                layer1.activate(layer1.output)
                layer2.forward(layer1.activated)
                layer2.activate(layer2.output)
                outputLayer.forward(layer2.activated)
                
                probabilities = softmax(outputLayer.output)
                max_prob = np.max(probabilities)
                predicted_digit = np.argmax(probabilities)
                
                print(f"I am {max_prob * 100:.2f}% sure that it is a {predicted_digit}")
            elif event.key == pygame.K_q:
                running = False
            elif event.key == pygame.K_k and (prompt == 1 or prompt == 2):
                sample = datasets[randint(0, len(datasets) - 1)]
                display_mnist_digit(sample['image'])
            elif event.key == pygame.K_a:
                outputs = np.zeros((1, 10))
                true = int(input("Correct Answer: "))
                outputs[0, true] = 1
                print("\nTraining on this example...\n")

                loss = calcLoss(outputs, outputLayer.output)
                count = 0

                while(count < 10000):
                    outputLayer.backward(loss)
                    layer2.backward(outputLayer.dinputs)
                    layer1.backward(layer2.dinputs)
                    layer1.weights -= rate * layer1.dweights
                    layer1.bias -= rate * layer1.dbias
                    layer2.weights -= rate * layer2.dweights
                    layer2.bias -= rate * layer2.dbias
                    outputLayer.weights -= rate * outputLayer.dweights
                    outputLayer.bias -= rate * outputLayer.dbias
                    layer1.forward(flattened_data)
                    layer1.activate(layer1.output)
                    layer2.forward(layer1.activated)
                    layer2.activate(layer2.output)
                    outputLayer.forward(layer2.activated)
                    loss = calcLoss(outputs, outputLayer.output)
                    count += 1

                print(f"Did {count} training iterations\n")
            elif event.key == pygame.K_l:
                while(not os.path.exists(filename)):
                    filename = input("filename: ")
                save_weights(filename)
                print(f"Trained data saved to {filename}")

    update_display()

pygame.quit()
