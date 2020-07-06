import numpy as np
import itertools
import random as rd

class conditionalProblabilityTable:
    def __init__(self, node, parents, domain, shape, probabilities):
        self.node,self.parents,self.domain,self.shape,self.probabilities=node,parents,{key:value for (key,value) in zip(domain,[i for i in range(len(domain))])},shape,probabilities
        

class BayesianNetwork:
    def __init__(self, filename):
        f = open(filename, 'r') 
        N = int(f.readline())
        lines = f.readlines()
        self.table={}
        for line in lines:
            node, parents, domain, shape, probabilities = self.__extract_model(line)
            self.table.update({node:conditionalProblabilityTable(node, parents, domain, shape, probabilities)})
        f.close()
        # print(self.table)

    def exact_inference(self, filename):
        f = open(filename, 'r')
        query_variables, evidence_variables = self.__extract_query(f.readline())
        queryNames=query_variables.keys()
        # print(queryNames)
        # print([self.table[node].domain.keys() for node in queryNames])
        queryValueCombinations=itertools.product(*[self.table[node].domain.keys() for node in queryNames])
        # for q in queryValueCombinations: print(q)
        queryCombinations=[{key:value for (key,value) in zip(queryNames,combination)} for combination in queryValueCombinations]
        # print(list(queryCombinations))
        # print(queryNames)
        Q={}
        for combination in queryCombinations:
            # print(combination)
            # print({**evidence_variables,**combination})
            Q.update({str(combination):self.enumerateAll(list(self.table.keys()),{**evidence_variables,**combination})})
        f.close()
        # print(Q)
        return Q[str(query_variables)]/sum(Q.values())
        # return 0
        
    def enumerateAll(self,vars,e):
        if len(vars)==0:
            return 1
        Y=vars[0]
        if Y in e:
            return self.Probability(Y,e[Y],e)*self.enumerateAll(vars[1:],e)
        else:
            return sum([self.Probability(Y,y,e)*self.enumerateAll(vars[1:],{**e,**{Y:y}}) for y in self.table[Y].domain.keys()])

    def Probability(self,node,value,e):
        a=self.table[node].probabilities
        for parent in self.table[node].parents:
            a=a[self.table[parent].domain[e[parent]]]
        return a[self.table[node].domain[value]]

    def approx_inference(self, filename):
        f = open(filename, 'r')
        query_variables, evidence_variables = self.__extract_query(f.readline())
        N=1000000
        W={key:0 for key in [str(sorted({keyy:valuee for (keyy,valuee) in zip(query_variables.keys(),combination)}.items())) for combination in itertools.product(*[self.table[node].domain.keys() for node in query_variables.keys()])]}
        for j in range(N):
            x,w=self.weightSample({**evidence_variables})
            W[str(sorted({key:value for (key,value) in x.items() if key in query_variables}.items()))]+=w
        f.close()
        return W[str(sorted(query_variables.items()))]/sum(W.values())
        # return 0

    def weightSample(self, e):
        w=1
        x={key:e[key] if key in e else 0 for key,value in self.table.items()}
        for Xi in self.table:
            if Xi in e:
                w*=self.Probability(Xi,e[Xi],e)
            else:
                x[Xi]=self.randomSample(Xi,e)
                e.update({Xi:x[Xi]})
        return x,w

    def randomSample(self, node, e):
        a=self.table[node].probabilities
        for parent in self.table[node].parents:
            a=a[self.table[parent].domain[e[parent]]]
        ohGodSoRandom=rd.random()
        for sample in self.table[node].domain:
            ohGodSoRandom-=a[self.table[node].domain[sample]]
            if ohGodSoRandom<0:
                return sample

    def __extract_model(self, line):
        parts = line.split(';')
        node = parts[0]
        if parts[1] == '':
            parents = []
        else:
            parents = parts[1].split(',')
        domain = parts[2].split(',')
        shape = eval(parts[3])
        probabilities = np.array(eval(parts[4])).reshape(shape)
        return node, parents, domain, shape, probabilities

    def __extract_query(self, line):
        parts = line.split(';')

        # extract query variables
        query_variables = {}
        for item in parts[0].split(','):
            if item is None or item == '':
                continue
            lst = item.split('=')
            query_variables[lst[0]] = lst[1]

        # extract evidence variables
        evidence_variables = {}
        for item in parts[1].split(','):
            if item is None or item == '':
                continue
            lst = item.split('=')
            evidence_variables[lst[0]] = lst[1]
        return query_variables, evidence_variables
