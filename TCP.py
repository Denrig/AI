"""
Reads Adj Matrix from file
Expects this kind of syntax
First Line:N=Number of vertex
Next N lines: Row of numbers separated with ","
Last 2 lines: Start vertex,End vertex

IN:Filname-String
RETURNS:ADJ MATRIX,START CITY,END CITY
"""
def readFromFile(fileName):
        f=open(fileName)
        matrix=[]
        a=int(f.readline())
        for x in range(a):
            x=f.readline()
            z=x.split(",")
            list=[]
            for s in z:
                list.append(int(s))
            matrix.append(list)
        startCity=int(f.readline())-1
        destCity=int(f.readline())-1
        return matrix,startCity,destCity

"""
Find the shortest path form one vertex to another

IN:Adj Matrix,start,end-optional
(If parameter end is not set it will go through all the vertexes)
OUT:Cost of the path,and the visited
"""
MAX_VAL=9999999999
def TCP(matrix,start, end=None):
        actualNode = start
        visited = [actualNode]
        cost = 0
        while end not in visited and len(visited) < len(matrix):
            lst = [el for el in matrix[actualNode]]
            while lst.index(min(lst)) in visited:
                lst[lst.index(min(lst))] = MAX_VAL
            for x in lst:
                if(x==0):
                    lst[lst.index(x)]=MAX_VAL
            cost += min(lst)
            actualNode = lst.index(min(lst))
            visited.append(lst.index(min(lst)))
        if end is None:
            cost += matrix[visited[-1]][start]
        return (cost,visited)


if __name__=="__main__":
    matrix,start,stop=readFromFile("easy_01_tsp.txt")
    print(TCP(matrix,start))