import numpy as np

class createNetwork:
    class createLayer:
        def __init__(self, nInput, nOutput):
            self.ins = nInput
            self.out = nOutput
            self.weights = np.random.randn(nInput, nOutput)
            self.bias = np.zeros((1, nOutput))
            self.m_w = np.zeros_like(self.weights)
            self.v_w = np.zeros_like(self.weights)
            self.m_b = np.zeros_like(self.bias)
            self.v_b = np.zeros_like(self.bias)
        
        def forward(self, inputs):
            self.inputs = inputs
            self.output = np.clip(np.dot(self.inputs, self.weights) + self.bias, -500, 500)
        
        def activateSigm(self, out):
            self.activated = 1 / (1 + np.exp(-out))

        def activateRelu(self, out):
            self.activated = np.maximum(0, out)

        def backward(self, loss, t, lr, beta1, beta2, epsilon):
            self.dvalue = loss
            self.dbias = np.sum(self.dvalue, axis=0, keepdims=True)
            self.dweights = np.dot(np.transpose(self.inputs), self.dvalue)
            self.dinputs = np.dot(self.dvalue, np.transpose(self.weights))

            # Update biased first moment estimate
            self.m_w = beta1 * self.m_w + (1 - beta1) * self.dweights
            self.m_b = beta1 * self.m_b + (1 - beta1) * self.dbias
            
            # Update biased second raw moment estimate
            self.v_w = beta2 * self.v_w + (1 - beta2) * (self.dweights ** 2)
            self.v_b = beta2 * self.v_b + (1 - beta2) * (self.dbias ** 2)
            
            # Compute bias-corrected first moment estimate
            m_w_hat = self.m_w / (1 - beta1 ** t)
            m_b_hat = self.m_b / (1 - beta1 ** t)
            
            # Compute bias-corrected second raw moment estimate
            v_w_hat = self.v_w / (1 - beta2 ** t)
            v_b_hat = self.v_b / (1 - beta2 ** t)
            
            # Update weights and biases
            self.weights -= lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
            self.bias -= lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

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
        inside = inputs
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

    def backpropagate(self, outputs, rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.loss = self.output - outputs
        t = 1  # Timestep starts at 1 for bias correction
        dvalue = self.loss
        for layer in self.layer[::-1]:
            layer.backward(dvalue, t, rate, beta1, beta2, epsilon)
            dvalue = layer.dinputs
            t += 1  # Increment timestep for next layer

    def softmax(self, outputs):
        temp = np.exp(outputs - np.max(outputs))
        return temp / np.sum(temp, axis=1, keepdims=True)
