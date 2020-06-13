"""
 ----------------------------------------------
|                                              |
|                                              |
|        Solving TCP problem using ACO         |
|                                              |
|                                              |
 ----------------------------------------------
"""

from math import sqrt
"""
Read matrix from file
"""
def readMatrix(fileName,k=0):
        f=open(fileName)
        lines=f.readlines()
        list=[]
        if k==0:
            k=len(lines)-1
        for i in range(7,k):
            currentLine=lines[i].split(" ")
            list.append((float(currentLine[0]),float(currentLine[1])))
        mat=[]
        for i in range(0,k-7):
            x1,y1=list[i]
            currentNode=[]
            for j in range(0,k-7):
                
                x2,y2=list[j]
                currentNode.append(sqrt((x1-x2)**2+(y1-y2)**2))
            mat.append(currentNode)

        net={}
        net['mat']=mat
        net['noCity']=len(mat)
        net['pheromone']=[[1 / (net['noCity'] ** 2) for j in range(net['noCity'])] for i in range(net['noCity'])]
        return net

from random import random

"""
This is Ant class,does what ants usualy do...Creates a part from start to destination
"""
class Ant:
    def __init__(self,probParam,start=0):
        self.probParam=probParam
        self.total_cost = 0.0 #cost of the selected link
        self.visited = []  # visited list
        self.pheromone_delta = []  # the local increase of pheromone
        self.allowed = [i for i in range(probParam["noCity"])]  # nodes which are allowed for the next selection
        self.heuristic = [[0 if i == j else 1 / probParam["mat"][i][j] for j in range(probParam["noCity"])] for i in range(probParam["noCity"])]  # heuristic information
        self.start=start #start node (0 by default)
        self.visited.append(start)
        self.current = start
        self.allowed.remove(start)

    def select_next(self):
        #compute the demominator from the probability funtion
        denominator = 0
        for i in self.allowed:
            denominator += self.probParam["pheromone"][self.current][i] * self.heuristic[self.current][i]
        # probabilities for moving to a node in the next step
        probabilities = [0 for i in range(self.probParam["noCity"])]  
        for i in range(self.probParam["noCity"]):
            if(i in self.allowed):
                probabilities[i] = self.probParam['pheromone'][self.current][i] * self.heuristic[self.current][i] / denominator
        selected = 0
        rand =random()
        #optain (index,pheromone[index])
        for i, probability in enumerate(probabilities):
            rand -= probability
            if rand <= 0:
                selected = i
                break
        self.allowed.remove(selected)
        self.visited.append(selected)
        self.total_cost += self.probParam["mat"][self.current][selected]
        self.current = selected


    def update_pheromone_delta(self):
        #set 0 to all local pheromone data
        self.pheromone_delta = [[0 for j in range(self.probParam["noCity"])] for i in range(self.probParam["noCity"])]
        #add 1 pheromone to every edge crossed
        for _ in range(1, len(self.visited)):
            i = self.visited[_ - 1]
            j = self.visited[_]
            self.pheromone_delta[i][j] = 1

"""
This Colony class stores a population of ants,and updates the pheromone paths based on the best ant
(ACO)
"""

class Colony:
    def __init__(self,acoParam,antParam):
        self.acoParam=acoParam
        self.antParam=antParam

    def update_pheromone(self,antParam, ants: list):
        for i, row in enumerate(antParam['pheromone']):
            for j, col in enumerate(row):
                for ant in ants:
                   antParam['pheromone'][i][j] += ant.pheromone_delta[i][j]

    def solve(self):
        best_cost = float('inf')
        best_solution = []
        ants = [Ant(self.antParam) for i in range(self.acoParam["ant_count"])]
        for ant in ants:
             for i in range(self.antParam["noCity"] - 1):
                  ant.select_next()
             ant.total_cost += self.antParam['mat'][ant.visited[-1]][ant.visited[0]]
             if ant.total_cost < best_cost:
                  best_cost = ant.total_cost
                  best_solution=[]+ant.visited
             ant.update_pheromone_delta()
        self.update_pheromone(self.antParam, ants)
        return (best_cost,best_solution)


from random import randrange

if __name__=="__main__":
    antParam=readMatrix('berlin52.txt')

    acoParam={'ant_count':50,'noGen':100}

    aco=Colony(acoParam,antParam)

    best_cost=99999999999999999999
    solution=[]
    for i in range(acoParam['noGen']):
        x,y=aco.solve()
        if(best_cost>x):
            best_cost=x
            solution=y
        print("Gen: "+str(i)+ " Solution: " + str(y) + "Cost:"+ str(x) )
    print("Best Solution" + str(solution) + "Cost:"+ str(best_cost))