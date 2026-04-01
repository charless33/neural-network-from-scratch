import netbuilder as nt
import numpy as np
from random import randint

architecture = [2, 4, 2]
activation = [2, 2, 0]
initialRate = 0.005
rate = initialRate
error = 0.2

temp1 = architecture[2]

network = nt.createNetwork(architecture, activation)

def accuracy(outs, expc, error):
	expc = [expc]
	count = 0
	for n in range(len(outs[0])):
		if abs(expc[0][n] - outs[0][n]) <= error:
			count += 1
	return count/len(outs[0])


right = 0
count = 0
batch = 10000

for n in range(1, 200000):
	inputs = [randint(0, 1), randint(0, 1), randint(0, 1), randint(0, 1)]
	exp = [inputs[0]+inputs[1]+inputs[2]+inputs[3], inputs[3] if inputs[0] == 1 else inputs[2]]

	network.propagate(inputs)
	network.backpropagate(exp, rate)

	if n % 40000 == 0:
		print(f"\n\nAccuracy:    {right/count*100}")

	if n%batch == 0:
		count = 0
		right = 0

	right += accuracy(network.output, exp, error)
	count += 1

	rate = initialRate - right/(count*100.005)


with open('/Users/carlomonti/Desktop/Python/NeuralNetwork/test(CANCEL)data.txt', 'w') as file:
	for k in range(len(network.layer)):
		for n in range(len(network.layer[k].weights)):
			for m in range(len(network.layer[k].weights[0])):
				file.write(f"{network.layer[k].weights[n][m]}\n")
			if k == 1:
				file.write(f"\n\n\n")
			else:
				file.write(f"\n")
		for n in range(len(network.layer[k].bias[0])):
			file.write(f"{network.layer[k].bias[0][n]}\n\n\n\n\n")
		file.write(f"\n\n\n\n\n\n\n\n\n")












