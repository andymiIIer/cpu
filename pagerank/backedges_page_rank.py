from simple_page_rank import SimplePageRank

"""
This class implements the pagerank algorithm with
backwards edges as described in the second part of 
the project.
"""
class BackedgesPageRank(SimplePageRank):

    """
    The implementation of __init__ and compute_pagerank should 
    still be the same as SimplePageRank.
    You are free to override them if you so desire, but the signatures
    must remain the same.
    """

    """
    This time you will be responsible for implementing the initialization
    as well. 
    Think about what additional information your data structure needs 
    compared to the old case to compute weight transfers from pressing
    the 'back' button.
    """
    @staticmethod
    def initialize_nodes(input_rdd):
        # YOUR CODE HERE
        # The pattern that this solution uses is to keep track of 
        # (node, (weight, targets, old_weight)) for each iteration.
        # When calculating the score for the next iteration, you
        # know that 10% of the score you sent out from the previous
        # iteration will get sent back.

        # takes in a line and emits edges in the graph corresponding to that line
        def emit_edges(line):
            # ignore blank lines and comments
            if len(line) == 0 or line[0] == "#":
                return []
            # get the source and target labels
            source, target = tuple(map(int, line.split()))
            # emit the edge
            edge = (source, frozenset([target]))
            # also emit "empty" edges to catch nodes that do not have any
            # other node leading into them, but we still want in our list of nodes
            self_source = (source, frozenset())
            self_target = (target, frozenset())
            return [edge, self_source, self_target]

        # collects all outgoing target nodes for a given source node
        def reduce_edges(e1, e2):
            return e1 | e2 

        # sets the weight of every node to 0, and formats the output to the 
        # specified format of (source (weight, targets))
        def initialize_weights((source, targets)):
            return (source, (1.0, targets, 1.0))

        nodes = input_rdd.flatMap(emit_edges).reduceByKey(reduce_edges).map(initialize_weights)
        return nodes

    """
    You will also implement update_weights and format_output from scratch.
    You may find the distribute and collect pattern from SimplePageRank
    to be suitable, but you are free to do whatever you want as long
    as it results in the correct output.
    """
    @staticmethod
    def update_weights(nodes, num_nodes):

        def distribute_weights((node, (weight, targets, old_weight))):
            emission = [(node, .05 * weight), (node, old_weight * .1)]
            if len(targets) == 0:
                e_weight = (.85 * weight) / (num_nodes - 1)
                for i in range(num_nodes):
                    if i == node:
                        continue
                    emission.append((i, e_weight))
            else:
                e_weight = (.85 * weight) / len(targets)
                for target in targets:
                    emission.append((target, e_weight))
            emission.append((node, targets))
            emission.append((node, (weight,)))
            return emission

        def collect_weights((node, values)):
            weight = 0
            for element in values:
                if type(element) == frozenset:
                    targets = element
                elif type(element) == tuple:
                    old_weight = element[0]
                else:
                    weight += element
            return (node, (weight, targets, old_weight))

        return nodes.flatMap(distribute_weights).groupByKey().map(collect_weights)
                     
    @staticmethod
    def format_output(nodes):
        return nodes.map(lambda (node, (weight, target, old_weight)): (weight, node)).sortByKey(ascending = False).map(lambda (weight, node): (node, weight))