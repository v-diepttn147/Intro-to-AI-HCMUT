import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from functools import reduce

class BayesianNetworkTester:
    def __init__(self, filename):
        # property
        self.networks = BayesianModel()
        self.mapper= dict()

        f = open(filename, 'r') 
        N = int(f.readline())
        lines = f.readlines()
        for line in lines:
            node, parents, domain, shape, probabilities = self.__extract_model(line)

            # Create Mapper
            mapper_node = dict()
            for index, value in enumerate(domain):
                mapper_node[value] = index

            self.mapper[node] = mapper_node

            # Add node
            self.networks.add_node(node)

            # Add edge to modal
            for parent in parents:
                self.networks.add_edge(parent, node)

            # Shape to list
            if isinstance(shape, tuple):
                shape = list(shape)[:-1]
            else:
                shape = []

            reshape = (reduce(lambda r, v: r*v, shape, 1), len(domain))
            probabilities = probabilities.reshape(reshape)
            probabilities = np.transpose(probabilities)

            cpd = TabularCPD(
                variable = node,
                variable_card = len(domain),
                values = probabilities,
                evidence = parents,
                evidence_card = shape
            )

            self.networks.add_cpds(cpd)

        f.close()

    def exact_inference(self, filename):
        result = 0
        f = open(filename, 'r')
        query_variables, evidence_variables = self.__extract_query(f.readline())
        
        eliminate = VariableElimination(self.networks)

        evidence_variables_mapped = dict()
        for variable in evidence_variables:
            evidence_variables_mapped[variable] = self.mapper[variable][evidence_variables[variable]]

        query_variables_feature = list(query_variables.keys())

        result = eliminate.query(variables=query_variables_feature, evidence=evidence_variables_mapped)

        value = result.values
        for feature in result.variables:
            value = value[result.get_state_no(feature, self.mapper[feature][query_variables[feature]])]
        

        f.close()
        return value

    def approx_inference(self, filename):
        result = 0
        f = open(filename, 'r')
        # YOUR CODE HERE


        f.close()
        return result

    def __extract_model(self, line):
        parts = line.split(';')
        node = parts[0]
        if parts[1] == '':
            parents = []
        else:
            parents = parts[1].split(',')
        domain = parts[2].split(',')
        shape = eval(parts[3])
        probabilities = np.array(eval(parts[4]))
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
