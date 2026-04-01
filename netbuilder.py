import numpy as np

np.random.seed(0)

class createNetwork:
	class createLayer:
	    def __init__(self, nInput, nOutput):
	        self.ins = nInput
	        self.out = nOutput
	        self.weights = np.random.randn(nInput, nOutput)
	        self.bias = np.zeros((1, nOutput))
	        self.dbias = 0
	        self.dweights = 0
	        self.dinputs = 0
	    
	    def forward(self, inputs):
	        self.inputs = inputs
	        self.output = np.clip(np.dot(self.inputs, self.weights) + self.bias, -500, 500)
	    
	    def activateSigm(self, out):
	        self.activated = 1 / (1 + np.exp(-out))

	    def activateRelu(self, out):
	        self.activated = np.maximum(0, out)

	    def backward(self, loss):
	        self.dvalue = loss
	        self.dbias = np.sum(self.dvalue, axis=0, keepdims=True)
	        self.dweights += np.dot(np.transpose(self.inputs), self.dvalue)
	        self.dinputs = np.dot(self.dvalue, np.transpose(self.weights))

	def __init__(self, architecture, activation):
		self.ins = architecture[0]
		self.out = architecture[len(architecture)-1]
		self.nLayers = len(architecture)-2
		self.architecture = architecture
		self.layer = []
		for n in range(len(self.architecture)-1):
			self.layer.append(self.createLayer(self.architecture[n], self.architecture[n+1]))
		self.active = activation

	def propagate(self, inputs):
		inside = [inputs]
		for n in range(len(self.layer)):
			self.layer[n].forward(inside)
			if(self.active[n] == 0):
				inside = self.layer[n].output
			elif(self.active[n] == 1):
				self.layer[n].activateSigm(self.layer[n].output)
				inside = self.layer[n].activated
			else:
				self.layer[n].activateRelu(self.layer[n].output)
				inside = self.layer[n].activated
		self.output = inside

	def l2_regularization(self, lambda_reg):
		l2_cost = 0
		for layer in self.layer:
			l2_cost += lambda_reg * np.sum(layer.weights ** 2)
			layer.dweights += 2 * lambda_reg * layer.weights
		return l2_cost

	def backpropagate(self, outputs, rate, lambda_reg=0.001):
		self.loss = self.output - outputs
		dvalue = self.loss
		self.cost = self.l2_regularization(lambda_reg)
		for layer in self.layer[::-1]:
			layer.backward(dvalue)
			dvalue = layer.dinputs
			layer.weights -= rate * layer.dweights
			layer.bias -= rate * layer.dbias
			layer.dweights -= layer.dweights


	def softmax(self, outputs):
		temp = np.exp(outputs - np.max(outputs))
		return temp / np.sum(temp, axis=1, keepdims=True)