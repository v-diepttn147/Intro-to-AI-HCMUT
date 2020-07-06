from pgmpy.readwrite import BIFReader
import numpy as np
from collections import deque

class Transformer:
    def __init__(self, pathfile):
        self.reader = BIFReader(pathfile)
        self.node_topo_sorted = list()

        # Find Topo order
        self.__find_topo_sorted()


    def __find_topo_sorted(self):
        parents = self.reader.get_parents()
        queue = deque()
        queue.extend(self.__find_root(parents))

        while len(queue):
            node  = queue.popleft()

            # Add to totpo list
            self.node_topo_sorted.append(node)
            
            # Remove node in parrents
            parents.pop(node)
            
            # Remove in child
            for parent in parents:
                if node in parents[parent]:
                    parents[parent].remove(node)

            # When queue emty try find child
            if len(queue) == 0:
                queue.extend(self.__find_root(parents))


    def __find_root(self, parrents):
        root = list()
        for parrent in parrents:
            if len(parrents[parrent]) == 0:
                root.append(parrent)

        return root

    def write(self, pathfile):
        parrents = self.reader.get_parents()
        states = self.reader.get_states()
        values = self.reader.get_values()

        with open(pathfile, 'w') as writer:
            # Write number of node
            writer.write( str(len(self.node_topo_sorted)) + '\n')

            # Write each node 
            for node in self.node_topo_sorted:

                # Node Name
                writer.write(node + ';')

                # Parent of node
                node_parrents = parrents[node]
                writer.write(','.join(node_parrents) + ';')
                
                # Write state
                state = states[node]
                writer.write(','.join(state) + ';')

                # Find dim
                dim = list()
                for node_parrent in node_parrents:
                    dim.append(len(states[node_parrent]))

                dim.append(len(state))

                # Write dim
                writer.write(','.join(map(str,dim)) + ';')

                # Write propabilities
                value = map(str, list(np.transpose(values[node]).ravel()))
                writer.write(','.join(value) )
            
                writer.write('\n')

print("transforming sach.bif...")
Transformer('./model/small/sachs.bif').write('./testcases/sachs/model.txt')

print("transforming asia.bif...")
Transformer('./model/small/asia.bif').write('./testcases/asia/model.txt')

print("transforming cancer.bif...")
Transformer('./model/small/cancer.bif').write('./testcases/cancer/model.txt')

print("transforming insurance.bif...")
Transformer('./model/small/insurance.bif').write('./testcases/insurance/model.txt')

print("transforming survey.bif...")
Transformer('./model/small/survey.bif').write('./testcases/survey/model.txt')

print("transforming earthquake.bif...")
Transformer('./model/small/earthquake.bif').write('./testcases/earthquake/model.txt')

print("transforming child.bif...")
Transformer('./model/medium/child.bif').write('./testcases/child/model.txt')