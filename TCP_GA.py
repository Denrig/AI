"""
"""

"""
 ----------------------------------------------
|                                              |
|                                              |
| COMUNITY DETECTION USING GA Algorithm        |
|                                              |
|                                              |
 ----------------------------------------------
"""

from random import randint, seed


def generateARandomPermutation(n):
    perm = [i for i in range(n)]
    pos1 = randint(0, n - 1)
    pos2 = randint(0, n - 1)
    perm[pos1], perm[pos2] = perm[pos2], perm[pos1]
    return perm

"""
Chromosome  class
Representation: Permutation-based representation
Crossover: Takes a part from 1 chromosome a part from another chromosome and combines them combines
Mutation:Replace a part of chromosome with a random value
"""

class Chromosome:
    def __init__(self, problParam = None):
        self.__problParam = problParam  #problParam has to store the number of nodes/cities
        self.__repres = generateARandomPermutation(self.__problParam['noNodes'])
        self.__fitness = 0.0
    
    @property
    def repres(self):
        return self.__repres 
    
    @property
    def fitness(self):
        return self.__fitness 
    @property
    def problParam(self):
        return self.__problParam
    @repres.setter
    def repres(self, l = []):
        self.__repres = l 
    
    @fitness.setter 
    def fitness(self, fit = 0.0):
        self.__fitness = fit 
    
    def crossover(self, c):
        # order XO
        pos1 = randint(-1, self.__problParam['noNodes'] - 1)
        pos2 = randint(-1, self.__problParam['noNodes'] - 1)
        if (pos2 < pos1):
            pos1, pos2 = pos2, pos1 
        k = 0
        newrepres = self.__repres[pos1 : pos2]
        for el in c.__repres[pos2:] +c.__repres[:pos2]:
            if (el not in newrepres):
                if (len(newrepres) < self.__problParam['noNodes'] - pos1):
                    newrepres.append(el)
                else:
                    newrepres.insert(k, el)
                    k += 1

        offspring = Chromosome(self.__problParam)
        offspring.repres = newrepres
        return offspring
    
    def mutation(self):
        # insert mutation
        pos1 = randint(0, self.__problParam['noNodes']-1)
        pos2 = randint(0, self.__problParam['noNodes']-1)
        if (pos2 < pos1):
            pos1, pos2 = pos2, pos1
        el = self.__repres[pos2]
        del self.__repres[pos2]
        self.__repres.insert(pos1 + 1, el)
        
    def __str__(self):
        return "\nChromo: " + str(self.__repres) + " has fit: " + str(self.__fitness)
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, c):
        return self.__repres == c.__repres and self.__fitness == c.__fitness


#Population

"""
 This is the GA class witch stores all the chromosomes and uses them to solve problems
 Can solve using Iterations(oneGenerations),Elitism(oneGenerationElitism),Steady-State(oneGenerationSteadyState)
"""

from random import uniform,random

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

    def get_probability_list(self):
        fitness=[c.fitness for c in self.population]
        sumFit=sum(fitness)
        total_fit = float(sumFit)
        relative_fitness = [f/total_fit for f in fitness]
        probabilities = [sum(relative_fitness[:i+1]) 
                     for i in range(len(relative_fitness))]
        return probabilities

    def selectWheel(self):
        chosen = 0
        probabilities=self.get_probability_list()
        r = random()
        for (i, individual) in enumerate(self.population):
            if r <= probabilities[i]:
                return self.population.index(individual)

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
            p1 = self.__population[self.selectWheel()]
            p2 = self.__population[self.selectWheel()]
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
Reads Adj Matrix from file
"""
def readMatrix(fileName,k=0):
        f=open(fileName)
        lines=f.readlines()
        list=[]
        if(k==0):
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
        net['noNodes']=len(mat)
        return net

from numpy import sqrt
#Computes the cost of a path
def lenPath(list,problParams):
    sum=0
    for i in range(0,len(list)-1):
        sum+=problParams['mat'][list[i]][list[i+1]]
    return sum


if __name__=="__main__":
    problParam=readMatrix("hardE.txt")
    problParam['function']=lenPath
    gaParam = {'popSize' : 150, 'noGen' : 200, 'pc' : 0.8, 'pm' : 0.1}

    ga = GA(gaParam, problParam)
    ga.initialisation()
    ga.evaluation()

    solution=None
    bestFitness=99999999999999

    for g in range(gaParam['noGen']):
            ga.oneGeneration()
            bestChromo = ga.bestChromosome()
            if(bestChromo.fitness<bestFitness):
                bestFitness=bestChromo.fitness
                solution=bestChromo
            print(str(bestChromo))
    print("Best Chromo in all generations",solution)