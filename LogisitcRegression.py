


def loadData(fileName):
    f=open(fileName)
    lines=f.readlines()
    inputs=[]
    outputs=[]
    for line in lines:
        line=line.split(",")
        inputs.append([float(line[a]) for a in range(len(line)-1)])
        outputs.append(line[-1].split('\n')[0])
    return inputs[:-1],outputs



from math import exp
from numpy.linalg import inv
import numpy as np

def sigmoid(x):
    return 1 / (1 + exp(-x))
    
class LogisticRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []

    # use the gradient descent method
    # simple stochastic GD
    def fit(self, x, y, learningRate = 0.001, noEpochs = 1000):
        self.coef_ = [0.0 for _ in range(1 + len(x[0]))]    #beta or w coefficients y = w0 + w1 * x1 + w2 * x2 + ...
        # self.coef_ = [random.random() for _ in range(len(x[0]) + 1)]    #beta or w coefficients 
        self.intercept_=self.coef_[0]
        for epoch in range(noEpochs):
            # TBA: shuffle the trainind examples in order to prevent cycles
            for i in range(len(x)): # for each sample from the training data
                ycomputed = sigmoid(self.eval(x[i], self.coef_))     # estimate the output
                crtError = ycomputed - y[i]     # compute the error for the current sample
                for j in range(0, len(x[0])):   # update the coefficients
                    self.coef_[j + 1] = self.coef_[j + 1] - learningRate * crtError * x[i][j]
                self.coef_[0] = self.coef_[0] - learningRate * crtError * 1
        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]
 
    def eval(self, xi, coef):
        yi = self.intercept_
        for j in range(1,len(xi)):
            yi += coef[j] * xi[j]
        return yi
    
    def getSigmoid(self,xi):
        
        return sigmoid(self.eval(xi,self.coef_))

    def predictOneSample(self, sampleFeatures):
        threshold = 0.5
        coefficients = [self.intercept_] + [c for c in self.coef_]
        computedFloatValue = self.eval(sampleFeatures, coefficients)
        computed01Value = sigmoid(computedFloatValue)
        computedLabel = 0 if computed01Value < threshold else 1 
        return computedLabel

    def predict(self, inTest):
        computedLabels = [self.predictOneSample(sample) for sample in inTest]
        return computedLabels




class LabelMaker:
	def __init__(self,labels):
		self.myRegressor_=[LogisticRegression() for _ in range(len(labels))]
		self.labels=labels
	def myTrain(self,trainInputs,trainOutputs):
		values=[]
		for regressor in self.myRegressor_:
			regressor.fit(trainInputs,trainOutputs[self.myRegressor_.index(regressor)])
			values.append((regressor.intercept_,regressor.coef_))
		return values
	def myPredict(self,inputs):
		computedOutputs=[regressor.getSigmoid(inputs) for regressor in self.myRegressor_]
		return self.labels[computedOutputs.index(max(computedOutputs))]
	def transforOutputs(self,labels,outputs):
		computed=[]
		for lable in labels:			
			computed.append([1 if lable==out else 0 for out in outputs])
		return computed



inputs,outputs=loadData("iris.data")
labels=['Iris-setosa','Iris-versicolor','Iris-virginica']
indexes = [i for i in range(len(inputs))]
trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace = False)
testSample = [i for i in indexes  if not i in trainSample]

trainInputs = [inputs[i] for i in trainSample]
trainOutputs = [outputs[i] for i in trainSample]
testInputs = [inputs[i] for i in testSample]
testOutputs = [outputs[i] for i in testSample]

def acc(computedOutputsMulti,realOutputsMulti):
    total=0
    correct=0
    for i in range(len(realOutputsMulti)):
        total+=1
        if(labels.index(realOutputsMulti[i])==labels.index(computedOutputsMulti[i])):
           correct+=1
    return correct/total

def normalisation(data,meanValues=[],stdDevValues=[]):
    normalisedData=[]
    if isinstance(data[0],list):
        if(meanValues==[] and stdDevValues==[]):
            for feature in zip(*data):
                meanValues.append(sum(feature)/len(feature))
            for feature in zip(*data):
                stdDevValues.append((1 / len(feature) * sum([ (feat - meanValues[list(zip(*data)).index(feature)]) ** 2 for feat in feature])) ** 0.5)     
        for i in range(len(data)):
            normalisedData.append( [(feat - meanValues[data[i].index(feat)]) / stdDevValues[data[i].index(feat)] for feat in data[i]])
    else:
        if(meanValues==[] and stdDevValues==[]):
            meanValues.append( sum(data) / len(data))
            stdDevValues .append( (1 / len(data) * sum([ (feat - meanValues[0]) ** 2 for feat in data])) ** 0.5) 
        normalisedData = [(feat - meanValues[0]) / stdDevValues[0] for feat in data]
    return normalisedData,[meanValues,stdDevValues]


if __name__=="__main__":
    trainInputs,[mean,dev] =normalisation([inputs[i] for i in trainSample])
    testInputs,_ = normalisation([inputs[i] for i in testSample],mean,dev)
    maker=LabelMaker(labels)
    newOutputs=maker.transforOutputs(labels,trainOutputs)
    print("Outputs transformed")
    values=maker.myTrain(trainInputs,newOutputs)
    print("Regressors trained")
    [print(a,b) for a,b in values]
    computed_outputs=[maker.myPredict(testInputs[i]) for i in range(len(testOutputs))]
    print("Predictions made")
    print("Acc:",acc(computed_outputs,testOutputs))