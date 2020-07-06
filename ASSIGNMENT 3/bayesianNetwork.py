import numpy as np
import itertools
from functools import reduce
import random as rd

def product(lst): 
    '''
    Create the truth table for input nested list of values
    >>> product([['True', 'False'], ['A', 'B', 'C']])
    [('True', 'A'), ('True', 'B'), ('True', 'C'), ('False', 'A'), ('False', 'B'), ('False', 'C')]
    '''
    return [x for x in itertools.product(*lst)]

def elimination_ask(X, e, bn):
    '''
    Compute P(X|e) by variable elimination
    '''
    factors = []
    for var in reversed(list(bn.node_map.keys())):
        factors.append(make_factor(var, e, bn))
        if is_hidden(var, X, e):
            factors = sum_out(var, factors, bn)
    return pointwise_product(factors, bn).normalize().show_approx()

def elimination_ask_mod(X, e, bn):
    '''
    Compute P(X|e) by variable elimination
    '''
    factors = []
    for var in reversed(list(bn.node_map.keys())):
        factors.append(make_factor(var, e, bn))
        if is_hidden(var, X, e):
            factors = sum_out(var, factors, bn)
    return pointwise_product(factors, bn).cpt

def make_factor(query, evidence, bn):
    '''
    Return the factor object for query in bayes net join distribution given evidence.
    >>> make_factor('S', {'I': 'Cao', 'D': 'Kho'}, bayesnet)
    cpt = {('Cao', 'Thap'): 0.2, ('Cao', 'Cao'): 0.8}
    variables = ['S']
    '''
    node = bn.getNodewithName(query)
    parents = [parent.nodeName for parent in node.parents]
    variables = [X for X in [query] + parents if X not in evidence]
    cpt = {}
    for e1 in all_events(variables, bn, evidence):
        cpt[event_values(e1, variables)] = node.p(e1[query], e1)
    return Factor(variables, cpt)

def event_values(event, variables):
    '''
    Return a tuple of the values of variables in event.
    >>> event_values ({'A': 10, 'B': 9, 'C': 8}, ['C', 'A'])
    (8, 10)
    >>> event_values ((1, 2), ['C', 'A'])
    (1, 2)
    '''
    if isinstance(event, tuple) and len(event) == len(variables):
        return event 
    else:
        return tuple([event[var] for var in variables])


def event_values_mod(event, value, variables):
    """Return a tuple of the values of variables in event.
    >>> event_values ({'A': 10, 'B': 9, 'C': 8}, ['C', 'A'])
    (8, 10)
    >>> event_values ((1, 2), ['C', 'A'])
    (1, 2)
    """
    if isinstance(event, tuple) and len(event) == len(variables):
        return event + (value,)
    else:
        return tuple([event[var] for var in variables]) + (value,)

def extend(s, var, val):
    '''
    Copy dict s and extend it by setting var to val; return copy.
    '''
    try: 
        return eval('{**s, var: val}')
    except SyntaxError: 
        s2 = s.copy()
        s2[var] = val
        return s2

def all_events(variables, bn, e):
    '''
    Yield every way of extending e with values for all variables.
    '''
    if not variables:
        yield e
    else:
        X, rest = variables[0], variables[1:]
        for e1 in all_events(rest, bn, e):
            for x in bn.getDomain(X):
                yield extend(e1, X, x)

def is_hidden(var, X, e):
    '''
    Is var a hidden variable when querying P(X|e)?
    '''
    return var != X and var not in e

def pointwise_product(factors, bn):
    return reduce(lambda f, g: f.pointwise_product(g, bn), factors)

def sum_out(var, factors, bn):
    '''
    Eliminate var from all factors by summing over its values.
    If var in f.variables, append f to var_factors
    Else, append f to result
    >>> sum_out('S', [Factor], bayesnet)
    result = [Factor]
    '''
    result, var_factors = [], []
    for f in factors:
        (var_factors if var in f.variables else result).append(f)
    result.append(pointwise_product(var_factors, bn).sum_out(var, bn))
    return result

class Factor:
    """A factor in a joint distribution."""

    def __init__(self, variables, cpt):
        self.variables = variables
        self.cpt = cpt

    def pointwise_product(self, other, bn):
        '''
        Multiply two factors, combining their variables.
        '''
        variables = list(set(self.variables) | set(other.variables))
        cpt = {}
        for e in all_events(variables, bn, {}):
            cpt[event_values(e, variables)] = self.p(e) * other.p(e) #[{'S': 'Thap'}, {'S': 'Cao'}]
        return Factor(variables, cpt)

    def sum_out(self, var, bn):
        '''
        Make a factor eliminating var by summing over its values.
        '''
        variables = [X for X in self.variables if X != var]
        cpt = {}
        for e in all_events(variables, bn, {}):
            cpt[event_values(e, variables)] = sum(self.p(extend(e, var, val)) for val in bn.getDomain(var))
        return Factor(variables, cpt)

    def normalize(self):
        '''
        Return my probabilities; must be down to one variable.
        '''
        assert len(self.variables) == 1
        return ProbDist(self.variables[0], {k: v for ((k,), v) in self.cpt.items()})

    def p(self, e):
        '''
        Look up my value tabulated for e.
        '''
        return self.cpt[event_values(e, self.variables)]

class ProbDist:
    '''
    A discrete probability distribution. You name the random variable
    in the constructor, then assign and query probability of values.
    >>> P = ProbDist('Flip'); P['H'], P['T'] = 0.25, 0.75; P['H']
    0.25
    >>> P = ProbDist('X', {'lo': 125, 'med': 375, 'hi': 500})
    >>> P['lo'], P['med'], P['hi']
    (0.125, 0.375, 0.5)
    '''

    def __init__(self, var_name='?', freq=None):
        '''
        If freq is given, it is a dictionary of values - frequency pairs,
        then ProbDist is normalized.
        '''
        self.prob = {}
        self.var_name = var_name
        self.values = []
        if freq:
            for (v, p) in freq.items():
                self[v] = p
            self.normalize()

    def __getitem__(self, val):
        '''
        Given a value, return P(value).
        '''
        try:
            return self.prob[val]
        except KeyError:
            return 0

    def __setitem__(self, val, p):
        '''
        Set P(val) = p.
        '''
        if val not in self.values:
            self.values.append(val)
        self.prob[val] = p

    def normalize(self):
        '''
        Make sure the probabilities of all values sum to 1.
        Returns the normalized distribution.
        Raises a ZeroDivisionError if the sum of the values is 0.
        '''
        total = sum(self.prob.values())
        if not np.isclose(total, 1.0):
            for val in self.prob:
                self.prob[val] /= total
        return self

    def show_approx(self, numfmt='{:.3g}'):
        '''
        Show the probabilities rounded and sorted by key, for the
        sake of portable doctests.
        '''
        return self.prob
        # return ', '.join([('{}: ' + numfmt).format(v, p) for (v, p) in sorted(self.prob.items())])

    def __repr__(self):
        return "P({})".format(self.var_name)


class BayesNode:
    '''
    All information of a node in Bayesian Network
    Example:
    nodeName = 'L'
    parents = [G]
    domain = ['Yeu', 'Manh']
    shape = (3, 2)
    cpTable = {('A', 'Yeu'): 0.1, ('A', 'Manh'): 0.9, ('B', 'Yeu'): 0.4, ('B', 'Manh'): 0.6, ('C', 'Yeu'): 0.99, ('C', 'Manh'): 0.01}
    '''
    def __init__(self, node, domain, shape, prob, parents=[]):
        self.nodeName = node    # string: node name
        self.parents = parents  # list[BayesNode]: parents
        self.domain = domain    # list[string]: node's domain
        self.shape = shape      # tuple(2) or (2,2) or (2,2,3) etc.
        self.domain2 = {key: value for (key, value) in zip(domain, [i for i in range(len(domain))])}
        self.prob2 = prob
        m = int(np.prod(prob.shape))
        prob = prob.reshape((m,))        # probability according to parrents, convert into 1D
        self.prob = prob.tolist()
        
        
        cpt = product([parent.domain for parent in self.parents] + [self.domain])
        self.cpTable = dict(zip(cpt, self.prob))
        # print(self.cpTable)

    def p(self, query, evidence):
        '''
        Return the conditional probability
        P(X=query | parents=evidence), where evidence are the values of parents in event.
        (event must assign each parent a value.)
        >>> nodeG.p('A', {'I': 'Cao', 'D': 'Kho'})
        0.5
        '''
        parents = [parent.nodeName for parent in self.parents]
        return self.cpTable[event_values_mod(evidence, query, parents)]



class BayesianNetwork:
    # nodes = []      # list of nodes in Bayesnet
    # node_map = {}   # mapping {'nodeName': BayesNode}

    def __init__(self, filename):
        self.nodes = []      # list of nodes in Bayesnet
        self.node_map = {}   # mapping {'nodeName': BayesNode}
        f = open(filename, 'r') 
        N = int(f.readline())
        lines = f.readlines()
        for line in lines:
            node, parents, domain, shape, probabilities = self.__extract_model(line)
            
            # get parent nodes
            parentNodes = self.getNodesWithNames(parents)
            
            # create new node instant carries all information
            if type(shape) != 'tuple':
                newNode = BayesNode(node, domain, tuple((shape,)), np.array(probabilities), parentNodes)
            else:
                newNode = BayesNode(node, domain, shape, np.array(probabilities), parentNodes)
            
            # add newNode to list node and node map
            self.nodes.append(newNode)
            self.node_map[node] = newNode

        f.close()

    def getNodewithName(self, nodeName):
        '''
        Return node object with node name
        '''
        if nodeName in self.node_map.keys():
            return self.node_map[nodeName]
        else:
            return None
    
    def getNodesWithNames(self, nodeNames=[]):
        '''
        Return list of BayesNode nodes from a list of nodeName
        '''
        if self.node_map != {}:
            nodes = [self.node_map[nodeName] for nodeName in nodeNames]
        else:
            nodes = []
        return nodes
    
    def getDomain(self, node):
        '''
        Return domains of the node
        '''
        if node in self.node_map.keys():
            return self.node_map[node].domain
        else:
            return []

    def exact_inference(self, filename):
        # result = 0
        f = open(filename, 'r')
        query_variables, evidence_variables = self.__extract_query(f.readline())

        X_list = list(query_variables.keys())
        value_list = list(query_variables.values())

        p = 1
        for i in range(len(query_variables)):
            p *= elimination_ask(X_list[0], evidence_variables, self)[query_variables[X_list[0]]]
            evidence_variables[X_list[0]] = query_variables[X_list[0]]
            X_list.pop(0)
        f.close()
        return p

    def approx_inference(self, filename):
        # result = 0
        f = open(filename, 'r')
        query_variables, evidence_variables = self.__extract_query(f.readline())

        # YOUR CODE HERE


        N = 200000

        query = query_variables.keys()

        # yield every way to combine domains of queries
        # query = 'G' => queryDomains = ('A',), ('B',), ('C',)
        queryDomains = itertools.product(*[self.node_map[node].domain2.keys() for node in query])

        # query = ['G', 'S'] => queryComb = [('G', 'A'), ('S', 'Thap')], etc.
        queryComb = [sorted({key: value for (key, value) in zip(query_variables.keys(), comb)}.items()) for comb in queryDomains]

        W = {str(comb): 0 for comb in queryComb}

        # generating N samples
        for i in range(N):
            x, w = self.weighted_sample({**evidence_variables})
            idx = str(sorted({key: value for (key, value) in x.items() if key in query_variables}.items()))
            W[idx] += w

        f.close()
        
        return W[str(sorted(query_variables.items()))] / sum(W.values())


    def weighted_sample(self, evidence):
        w = 1
        event = {}
        for key in self.node_map.keys():
            if key in evidence:
                event[key] = evidence[key]
            else:
                event[key] = 0
        for Xi in self.node_map:
            if Xi in evidence:
                w *= self.P(Xi, evidence[Xi], evidence)
            else:
                event[Xi] = self.sample(Xi, evidence)
                evidence[Xi] = event[Xi]
        return event, w

    def sample(self, node, evidence):
        p = self.node_map[node].prob2
        for parent in self.node_map[node].parents:
            p = p[self.node_map[parent.nodeName].domain2[evidence[parent.nodeName]]]
        r = rd.random()
        for sample in self.node_map[node].domain2:
            r -= p[self.node_map[node].domain2[sample]]
            if r < 0:
                return sample

    def P(self, node, domain_value, evidence):
        '''
        Return probability 
        >>> P('I', 'Cao', {'I': 'Cao', 'D': 'Kho', 'G': 'A'})
        0.3
        '''
        # # get the probability of this node
        # p = self.node_map[node].prob
        # # traverse through parents' domain and get the probability given parents (evidence)
        # for parent in self.node_map[node].parents:
        #     p = p[self.node_map[parent].domain[evidence[parent]]]
        # return p[self.node_map[node].domain[domain_value]]
        
        p = self.node_map[node].p(domain_value, evidence)
        # print(p)
        return p


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
