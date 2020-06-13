"""
 ----------------------------------------------
|                                              |
|                                              |
|          COMUNITY DETECTION USING GA         |
|                                              |
|                                              |
 ----------------------------------------------
"""

#UTILS

"""
Reads The adj matrix from a gml file
IN:fileName-String
OUT:Matrix
"""
from random import *
import networkx as nx



def readNet(fileName):
    G = nx.read_gml(fileName,'id')
    mat=nx.to_numpy_matrix(G)
    net={}
    net['noNodes']=len(mat)
    net['mat']=mat.tolist()
    degrees=nx.degree(G)
    list=[]
    for x in degrees:
        z,y=x
        list.append(y)
    noEdges=nx.number_of_edges(G)
    net['degrees']=list
    net['noEdges']=noEdges
    return net

#Chromosome

"""
Chromosome  class
Representation: List,each index it's assigned a vertex and the value inside the list it's the community
Crossover: For each element in the 2 list's add the values and then modulo number of the communities
Mutation:We reduce the number of the communities by replacing it with its index
"""


from random import randint

class Chromosome:
    def __init__(self, problParam = None):
        self.__problParam = problParam
        self.__repres = [randrange(1,problParam['noNodes']//2,1) for _ in range(problParam['noNodes'])]
        self.__fitness = 0.0
    
    @property
    def repres(self):
        return self.__repres
    
    @property
    def fitness(self):
        return self.__fitness 
    
    @repres.setter
    def repres(self, l = []):
        self.__repres = l 
    
    @fitness.setter 
    def fitness(self, fit = 0.0):
        self.__fitness = fit 
    
    @property
    def problParam(self):
        return self.__problParam

    def crossover(self, c):
        newrepres = []
        for i in range(len(self.__repres)):
            newrepres.append((self.__repres[i]+c.__repres[i])%max(self.__repres))
        offspring = Chromosome(c.__problParam)
        offspring.repres = newrepres
        return offspring
    
    def mutation(self):
        modified=[]
        self.__repres = [x if (x in modified) else self.__repres.index(x) for x in self.__repres]
        
    def __str__(self):
        return '\nChromo: ' + str(self.__repres) + ' has fit: ' + str(self.__fitness)
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, c):
        return self.__repres == c.__repres and self.__fitness == c.__fitness


#Population

"""
 This is the GA class witch stores all the chromosomes and uses them to solve problems
 Can solve using Iterations(oneGenerations),Elitism(oneGenerationElitism),Steady-State(oneGenerationSteadyState)
"""

class GA:
    def __init__(self, param = None, problParam = None):
        self.__param = param
        self.__problParam = problParam
        self.__population = []
        
    @property
    def population(self):
        return self.__population
    
    def initialisation(self):
        for _ in range(0, self.__param['popSize']):
            c = Chromosome(self.__problParam)
            self.__population.append(c)
    
    def evaluation(self):
        for c in self.__population:
            c.fitness = self.__problParam['function'](c.repres,c.problParam)
            
    def bestChromosome(self):
        best = self.__population[0]
        for c in self.__population:
            if (c.fitness < best.fitness):
                best = c
        return best
        
    def worstChromosome(self):
        best = self.__population[0]
        for c in self.__population:
            if (c.fitness > best.fitness):
                best = c
        return best

    def selection(self):
        pos1 = randint(0, self.__param['popSize'] - 1)
        pos2 = randint(0, self.__param['popSize'] - 1)
        if (self.__population[pos1].fitness < self.__population[pos2].fitness):
            return pos1
        else:
            return pos2 
        
    
    def oneGeneration(self):
        newPop = []
        for _ in range(self.__param['popSize']):
            p1 = self.__population[self.selection()]
            p2 = self.__population[self.selection()]
            off = p1.crossover(p2)
            off.mutation()
            newPop.append(off)
        self.__population = newPop
        self.evaluation()

    def oneGenerationElitism(self):
        newPop = [self.bestChromosome()]
        for _ in range(self.__param['popSize'] - 1):
            p1 = self.__population[self.selection()]
            p2 = self.__population[self.selection()]
            off = p1.crossover(p2)
            off.mutation()
            newPop.append(off)
        self.__population = newPop
        self.evaluation()
        
    def oneGenerationSteadyState(self):
        for _ in range(self.__param['popSize']):
            p1 = self.__population[self.selection()]
            p2 = self.__population[self.selection()]
            off = p1.crossover(p2)
            off.mutation()
            off.fitness = self.__problParam['function'](off.repres,off.problParam)
            worst = self.worstChromosome()
            if (off.fitness < worst.fitness):
                worst = off 


"""
Computes the performance of the computed communities
"""

def modularity(communities, param):
    noNodes = param['noNodes']
    mat = param['mat']
    degrees = param['degrees']
    noEdges = param['noEdges']  
    M = 2 * noEdges
    Q = 0.0
    for i in range(noNodes):
        for j in range(noNodes):          
            if (communities[i] == communities[j]):
               Q += (mat[i][j] - degrees[i] * degrees[j] / M)
    return Q * 1 / M

#MAIN

if __name__== "__main__":
    # initialise parameters
    
    problParam=readNet('football.gml')
    problParam['noCommunities']=3
    noGen=100
    problParam['function']=modularity
    gaParam = {'popSize' : 100, 'noGen' : noGen, 'pc' : 0.8, 'pm' : 0.1}
    # store the best/average solution of each iteration (for a final plot used to anlyse the GA's convergence)
    allBestFitnesses = []
    allAvgFitnesses = []
    generations = []


    ga = GA(gaParam, problParam)
    ga.initialisation()
    ga.evaluation()

    for g in range(gaParam['noGen']):
        allPotentialSolutionsX = [c.repres for c in ga.population]
        allPotentialSolutionsY = [c.fitness for c in ga.population]
        bestSolX = ga.bestChromosome().repres
        bestSolY = ga.bestChromosome().fitness
        allBestFitnesses.append(bestSolY)
        allAvgFitnesses.append(sum(allPotentialSolutionsY) / len(allPotentialSolutionsY))
        generations.append(bestSolX)
         
        ga.oneGenerationElitism()
        bestChromo = ga.bestChromosome()
        print('Best solution in generation ' + str(g) + ' modularity = ' + str(bestChromo.fitness) +' noCommunities: '+str(max(bestChromo.repres)))
    print('Best chromo '+str(max(allBestFitnesses))+ " "+str(max(generations[allBestFitnesses.index(max(allBestFitnesses))])))