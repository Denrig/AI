"""
 ----------------------------------------------
|                                              |
|                                              |
|          SOLVING A REGRESION USING           |
|           GRADIENT DESCENT METHOD            |
|              WITH MINI-BATCHES               |
|                                              |
 ----------------------------------------------
"""


import csv 
import numpy as np
"""
This just reads data from a file
"""
def loadData(fileName, inputVariableNames, outputVariableName):

    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1
    if isinstance(inputVariableNames,list):                
        selectedVariable1 = dataNames.index(inputVariableNames[0])
        selectedVariable2 = dataNames.index(inputVariableNames[1])
        inputs = [[float(data[i][selectedVariable1]), float(data[i][selectedVariable2])] for i in range(len(data))]
        selectedOutput = dataNames.index(outputVariableName)
        outputs = [float(data[i][selectedOutput]) for i in range(len(data))]
    else:
        selectedVariable = dataNames.index(inputVariableNames)
        inputs = [float(data[i][selectedVariable]) for i in range(len(data))]
        selectedOutput = dataNames.index(outputVariableName)
        outputs = [float(data[i][selectedOutput]) for i in range(len(data))]
    return inputs, outputs

"""
This class solves the regression with gradient descent method
The funciong looks like this: f(x)=w0+w1*x1+w2*x2;
"""
import random

class BGDRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []

    def fit(self, x, y, learningRate = 0.001, noEpochs = 1000,mini=0):
        if(mini!=0):
            dataX=[x[random.randrange(0,len(x))] for _ in range(mini)]
            dataY=[y[x.index(i)] for i in dataX]
        else:
            dataX=x
            dataY=y
            mini=len(x)
        if isinstance(x[0],list):
            self.coef_ = [random.random() for _ in range(len(x[0]) + 1)]
        else:
            self.coef_ = [random.random() for _ in range(2)]
        self.intercept_=self.coef_[-1]
        copy=self.coef_
        for epoch in range(noEpochs):
            for i in range(mini):
                if isinstance(dataX[i],list):
                    ycomputed = self.eval(dataX[i])
                else:
                    ycomputed=self.eval([dataX[i]])
                    dataX[i]=[dataX[i]]
                crtError = ycomputed - dataY[i]
                for j in range(0, len(dataX[0])):  
                    copy[j] = copy[j] - learningRate * crtError * dataX[i][j]
                copy[len(dataX[i])] = copy[len(dataX[i])] - learningRate * crtError * 1
            self.intercept_ = copy[-1]
            self.coef_ = copy[:-1]

    def eval(self, xi):
        yi = self.intercept_
        for j in range(len(xi)):
            yi += self.coef_[j] * xi[j]
        return yi 

    def predict(self, x):
        yComputed = [self.eval(xi) for xi in x]
        return yComputed



"""
Here we normalize our data
"""
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




#Reading data
inputs, outputs = loadData('world-happiness-report-2017.csv', ['Economy..GDP.per.Capita.', 'Freedom'], 'Happiness.Score')

if __name__=="__main__":
    #Parsing data
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace = False)
    testSample = [i for i in indexes  if not i in trainSample]

    trainInputs,[mean,dev] =normalisation([inputs[i] for i in trainSample])
    trainOutputs,_ = normalisation([outputs[i] for i in trainSample],mean,dev)
    testInputs,_ = normalisation([inputs[i] for i in testSample],mean,dev)
    testOutputs,_ =normalisation([outputs[i] for i in testSample],mean,dev)

    

    #Training
    bgd=BGDRegression()
    bgd.fit(trainInputs,trainOutputs)

    print("Found coefficients are:",bgd.coef_)
 
    #Predict testInputs
    computedTestOutputs = bgd.predict(testInputs)
    [print("Predicted:",out) for out in computedTestOutputs]
    #Compute errors
    error = 0.0
    for t1, t2 in zip(computedTestOutputs, testOutputs):
        error += (t1 - t2) ** 2
    error = error / len(testOutputs)
    print('prediction error: ', error)