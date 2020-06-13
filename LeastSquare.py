"""
 ----------------------------------------------
|                                              |
|                                              |
|          SOLVING A REGRESION USING           |
|            LEAST SQUARE METHOD               |
|                                              |
|                                              |
 ----------------------------------------------
"""


import csv
"""
This just reads data from a file
"""
def loadData(fileName, inputVariabName1,inputVariabName2, outputVariabName):
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
    selectedVariable1 = dataNames.index(inputVariabName1)
    selectedVariable2 = dataNames.index(inputVariabName2)
    inputMat=[[1,float(data[i][selectedVariable1]),float(data[i][selectedVariable2])] for i in range(len(data))]
    selectedOutput = dataNames.index(outputVariabName)
    outputs = [float(data[i][selectedOutput]) for i in range(len(data))]
    
    return inputMat,outputs


"""
This class solves the regression with least square method
The funciong looks like this: f(x)=w0+w1*x1+w2*x2;
"""
class LS_Regresion:
    def __init__(self):
        self.w_=[] 
    def transposeMatrix(self,X):
        return [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]

    def getMatrixMinor(self,m,i,j):
        return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

    def getMatrixDeternminant(self,m):
        #base case for 2x2 matrix
        if len(m) == 2:
            return m[0][0]*m[1][1]-m[0][1]*m[1][0]
        determinant = 0
        for c in range(len(m)):
            determinant += ((-1)**c)*m[0][c]*self.getMatrixDeternminant(self.getMatrixMinor(m,0,c))
        return determinant

    def getMatrixInverse(self,m):
        determinant = self.getMatrixDeternminant(m)
        #find matrix of cofactors
        cofactors = []
        for r in range(len(m)):
            cofactorRow = []
            for c in range(len(m)):
                minor = self.getMatrixMinor(m,r,c)
                cofactorRow.append(((-1)**(r+c)) * self.getMatrixDeternminant(minor))
            cofactors.append(cofactorRow)
        cofactors = self.transposeMatrix(cofactors)
        for r in range(len(cofactors)):
            for c in range(len(cofactors)):
                cofactors[r][c] = cofactors[r][c]/determinant
        return cofactors
    def fit(self,X,Y):
         XT=self.transposeMatrix(X) 
         mulMat=[[sum(a*b for a,b in zip(X_row,X_col)) for X_col in zip(*X)] for X_row in XT]
         invX=self.getMatrixInverse(mulMat)
         mat2=[[sum(a*b for a,b in zip(X_row,X_col)) for X_col in zip(*XT)] for X_row in invX]
         self.w_=[sum(mat2[j][i]*Y[i] for i in range(len(mat2[j]))) for j in range(3)]
    def predict(self, x):
            return [self.w_[0] + self.w_[1] * val[0]+self.w_[2]* val[1] for val in x]


import numpy as np
import random as rand

"""
We want to determine the Happiness.Score based on the Economy..GDP.per.Capita and Freedom
"""
if __name__=="__main__":
    inTrain,outTrain=loadData('world-happiness-report-2017.csv', 'Economy..GDP.per.Capita.','Freedom', 'Happiness.Score')
    r=LS_Regresion()
    r.fit(inTrain,outTrain)
    vect=r.w_
    print("Found coefficients are:",vect)

    testInputs = [[1,rand.uniform(0,1),rand.uniform(0,1)] for _ in range(20)]
    testOutputs = [rand.uniform(0,8) for _ in range(20)]
    computedTestOutputs = r.predict([x for x in testInputs])
    [print("Predicted:",out) for out in computedTestOutputs]

