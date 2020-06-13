from random import uniform,shuffle;
from math import sqrt


"""
 ----------------------------------------------
|                                              |
|                                              |
|          SOME EVALUATION FUNCTIONS           |
|            FOR MACHINE LEARNING              |
|                                              |
|                                              |
 ----------------------------------------------
"""

#multi-target regression error
def computeError(realOutputs,computeOutputs):
    error=0.0
    for i in range(len(realOutputs)):
        currentSum=0.0
        for j in range(len(realOutputs[i])):
            currentSum+=(realOutputs[i][j]-computeOutputs[i][j])**2
        error+=sqrt(currentSum)
    return error/len(realOutputs)



def computeMultiClass(realOutputs,computeOutputs,labels):
    mat=[[0]*len(labels) for _ in range(len(labels))]
    #accuracy
    total=0
    correct=0
    for i in range(len(realOutputsMulti)):
        mat[labels.index(realOutputsMulti[i])][labels.index(computedOutputsMulti[i])]+=1
        total+=1
        if(labels.index(realOutputsMulti[i])==labels.index(computedOutputsMulti[i])):
           correct+=1
    acc=correct/total
    #precision
    precision=[]
    for i in range(len(mat)):
        total=0
        correct=0
        for j in range(len(mat)):
            if(i==j):
                correct+=mat[i][j]
            total+=mat[i][j]
        precision.append({labels[i]:"{:.2f}".format(correct/total)})
    #recall
    recall=[]
    for j in range(len(mat)):
        total=0
        correct=0
        for i in range(len(mat)):
            if(i==j):
                correct+=mat[j][i]
            total+=mat[i][j]
        recall.append({labels[j]:"{:.2f}".format(correct/total)})
    return (acc,precision,recall)

#multi-class
labels=["dog","cat","lion","donkey","tiger"]
realOutputsMulti=['lion', 'donkey', 'donkey', 'tiger', 'lion', 'tiger', 'cat', 'dog', 'dog', 'cat']
computedOutputsMulti=['dog', 'donkey', 'tiger', 'lion', 'lion', 'dog', 'cat', 'dog', 'cat', 'lion']
computedProbs=[[0.7,0.6,0.5,0.6,0.4],[0.7,0.6,0.5,0.8,0.4],[0.7,0.6,0.5,0.6,0.8],[0.7,0.6,0.8,0.6,0.4],[0.7,0.6,0.8,0.6,0.4],[0.8,0.6,0.5,0.6,0.4],[0.7,0.8,0.5,0.6,0.4],[0.8,0.6,0.5,0.6,0.4],[0.7,0.8,0.5,0.6,0.4],[0.7,0.6,0.5,0.8,0.4]]

#multi-target regression error
realOutputs=[[1.8, 7.91, 4.1, 6.27, 4.3],
            [6.69, 8.58, 7.98, 1.39, 7.66],
            [6.2, 8.3, 4.07, 7.99, 2.36],
            [1.28, 3.17, 8.18, 9.68, 5.92], 
            [3.75, 5.1, 5.83, 6.01, 4.72], 
            [8.54, 3.76, 9.76, 3.39, 6.26], 
            [8.1, 5.02, 3.37, 1.93, 5.7]]
computeOutputs=[[2.19, 7.78, 3.78, 9.93, 8.92], 
                [5.9, 4.26, 3.26, 8.89, 8.58], 
                [3.86, 1.45, 1.17, 7.64, 9.3], 
                [8.53, 4.92, 5.31, 2.21, 3.66], 
                [8.25, 8.96, 9.96, 3.55, 1.58], 
                [4.1, 2.64, 9.19, 3.99, 3.37], 
                [8.5, 3.06, 5.13, 1.04, 9.26]]


if __name__=="__main__":
    acc, prec, recall=computeMultiClass(realOutputsMulti,computedOutputsMulti,labels)
    print('acc: ', acc, '\nprecision: ', prec, '\nrecall: ', recall)
