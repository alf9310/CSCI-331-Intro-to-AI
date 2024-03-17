import argparse
from collections import deque
import functools
import heapq
import itertools
import numpy as np
import pdb
import random
import sys
import time


class Problem:
    """The class for a formal problem."""

    def __init__(self, initial, cars_per_action):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments.

        Vars:
            initial: the initial state
            cars_per_action: aka number of attendants, or maximum number
                of cars that can move on each step.
        """
        self.initial = initial
        self.cars_per_action = cars_per_action
        

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. 
       
        It should return an iterable (e.g., a list)
        containing legal actions. Each action consists of a list of 
        self.cars_per_action (car, move) pairs, where car is the
        numerical id of the car and move is one of "up", "down", 
        "left", "right", "stay". For instance if state.n = 10 and
        self.cars_per_move = 3, then

        {(3, "up"), (2, "stay"), (7, "left")}

        is a legal action
        So, you must return a *list* of actions, where each action
        is a *set* of car/action pairs. 
        """
        action_list = []
        car_actions = []
        '''The list of action sets'''
        #move = set()
        '''Sets of a actions that can be performed'''
        for car_num in range(len(state.cars)):
            #print(car_num)
            #print(state.cars[car_num])
            x = state.cars[car_num][0]
            y = state.cars[car_num][1]
            if x != 0 and (x-1, y) not in state.cars and (x-1, y) not in state.barriers:
                car_actions.append((car_num, "up"))
            if x != state.n-1 and (x+1, y) not in state.cars and (x+1, y) not in state.barriers:
                car_actions.append((car_num, "down"))
            if y != 0 and (x, y-1) not in state.cars and (x, y-1) not in state.barriers:
                car_actions.append((car_num, "left"))
            if y != state.n-1 and (x, y+1) not in state.cars and (x, y+1) not in state.barriers:
                car_actions.append((car_num, "right"))
            car_actions.append((car_num, "stay"))
            action_list.extend(car_actions)
            #print(action_list)
            car_actions = []
        '''if len(action_list) % self.cars_per_action != 0:
            action_list.extend([action_list[0]] * (self.cars_per_action - (len(action_list)%self.cars_per_action)))'''
        move = set()
        final_list = []
        #print(action_list)
        index = -1
        index_list = []
        #while len(action_list) != 0:
        #index = -1
        for car_move in action_list:
         #       index += 1
            #print(car_move)
            in_move = False
            if len(move) == 0:
                move.add(car_move)
                #action_list.pop(index)
                index_list.append(index)
            elif len(move) < self.cars_per_action:
                for pairs in move:
                    if car_move[0] == pairs[0]:
                        in_move = True
                if not in_move:
                    move.add(car_move)
                    #action_list.pop(index)
                    index_list.append(index)
            if len(move) == self.cars_per_action:
             #   print(move)
                final_list.append(set(move))
                move.clear()
            #for i in index_list:
             #   action_list.pop(i)
        #print(final_list)
        #final_sets = [set(final_set) for final_set in final_list]
        #print(final_sets)
        '''action_split = [action_list[i:(i+self.cars_per_action)] for i in range(0, len(action_list), self.cars_per_action)]
        print(action_split)
        action_sets = [set(action_set) for action_set in action_split]
        print(action_sets)'''
        print(final_list)
        breakpoint()
        return final_list

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        new_state = state
        for x in action:
            car = x[0]
            action_dir = x[1]
            position = state.cars[car]
            if action_dir == "up":
                new_pos = (position[0]-1, position[1])
            elif action_dir == "down":
                new_pos = (position[0]+1, position[1])
            elif action_dir == "left":
                new_pos = (position[0], position[1]-1)
            elif action_dir == "right":
                new_pos = (position[0], position[1]+1)
            else:
                new_pos = (position[0], position[1])

            new_state.cars_inv[position] = None
            # unsure if this line is needed...
            new_state.cars_inv[new_pos] = car
            new_state.cars[car] = new_pos
        return new_state

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough.
        
        test whether cars have made it to the bottom row, in 
        reverse order.
        """
        return all([(y == state.n-1-i) and (x == state.n-1) for i,(x,y) in zip(range(state.n), state.cars)])

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        '''for x in action:
            print(x)
            if x[1] != "stay":
                c = c'''
        #print("end")
        return c + 1

    def value(self, state):
        """Use this for the estimated value of the node.

        It is not necessary to implement this, but you can if you like.
        """
        raise NotImplementedError

def heuristic_dist(node):
    raise NotImplementedError
# ___________________________________________________________________
# You should not modify anything below the line (except for test
# purposes, in which case you should repair it to the original state
# before submission.)
# --------------------------------------------------------------------
class State:
    def __init__(self, cars: list, barriers: set):
        self.n = len(cars)
        self.cars = cars
        self.barriers = barriers
        self.cars_inv = {j:i for i, j in zip(range(self.n), self.cars)}

    def __eq__(self, other):
        if self.n != other.n:
            return False
        return all([x == y for x,y in zip(self.cars, other.cars)])

    def __lt__(self, state):
        return tuple(self.cars) < tuple(state.cars)

    def __hash__(self):
        return tuple(self.cars).__hash__()

    def __repr__(self):
        max_len = len(f"{self.n}") + 1
        num_format = "{car: " + f"{max_len}" + "d}"
        barrier_str = " " * (max_len - 1) + "*"
        final_str = "+" + "-" * (2 * (self.n + 2)) + "+\n"
    
        for i in range(self.n):
            final_str += "|  "
            for j in range(self.n):
                if (i,j) in self.cars:
                    final_str += num_format.format(car=self.cars_inv[(i,j)])
                elif (i,j) in self.barriers:
                    final_str += barrier_str
                else:
                    final_str += " " * max_len
            final_str += "  |\n"
        final_str += "+" + "-" * (2 * (self.n + 2)) + "+\n"
        return final_str


class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "Node\n{}".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        problem.actions(self.state)
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        print("test")
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)

class PriorityQueue:
    """A Queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first.
    If order is 'min', the item with minimum f(x) is
    returned first; if order is 'max', then it is the item with maximum f(x).
    Also supports dict-like lookup."""

    def __init__(self, order='min', f=lambda x: x):
        self.heap = []
        if order == 'min':
            self.f = f
        elif order == 'max':  # now item with max f(x)
            self.f = lambda x: -f(x)  # will be popped first
        else:
            raise ValueError("Order must be either 'min' or 'max'.")

    def append(self, item):
        """Insert item at its correct position."""
        heapq.heappush(self.heap, (self.f(item), item))

    def extend(self, items):
        """Insert each item in items at its correct position."""
        for item in items:
            self.append(item)

    def pop(self):
        """Pop and return the item (with min or max f(x) value)
        depending on the order."""
        if self.heap:
            return heapq.heappop(self.heap)[1]
        else:
            raise Exception('Trying to pop from empty PriorityQueue.')

    def __len__(self):
        """Return current capacity of PriorityQueue."""
        return len(self.heap)

    def __contains__(self, key):
        """Return True if the key is in PriorityQueue."""
        return any([item == key for _, item in self.heap])

    def __getitem__(self, key):
        """Returns the first value associated with key in PriorityQueue.
        Raises KeyError if key is not present."""
        for value, item in self.heap:
            if item == key:
                return value
        raise KeyError(str(key) + " is not in the priority queue")

    def __delitem__(self, key):
        """Delete the first occurrence of key."""
        try:
            del self.heap[[item == key for _, item in self.heap].index(True)]
        except ValueError:
            raise KeyError(str(key) + " is not in the priority queue")
        heapq.heapify(self.heap)
        

def memoize(fn, slot=None, maxsize=32):
    """Memoize fn: make it remember the computed value for any argument list.
    If slot is specified, store result in that slot of first argument.
    If slot is false, use lru_cache for caching the values."""
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        @functools.lru_cache(maxsize=maxsize)
        def memoized_fn(*args):
            return fn(*args)

    return memoized_fn

def depth_first_tree_search(problem):
    """
    [Figure 3.7]
    Search the deepest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Repeats infinitely in case of loops.
    """

    frontier = [Node(problem.initial)]  # Stack

    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        frontier.extend(node.expand(problem))
    return None

def depth_first_graph_search(problem):
    """
    [Figure 3.7]
    Search the deepest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Does not get trapped by loops.
    If two paths reach a state, only use the first one.
    """
    frontier = [(Node(problem.initial))]  # Stack

    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored and child not in frontier)
    return None

def breadth_first_graph_search(problem):
    """[Figure 3.11]
    Note that this function can be implemented in a
    single line as below:
    return graph_search(problem, FIFOQueue())
    """
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = deque([node])
    explored = set()
    while frontier:
        node = frontier.popleft()
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child
                frontier.append(child)
    return None

def best_first_graph_search(problem, f, display=False):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            if display:
                print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return None

def uniform_cost_search(problem, display=False):
    """[Figure 3.14]"""
    return best_first_graph_search(problem, lambda node: node.path_cost, display)

def astar_search(problem, h=None, display=False):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)

search_dict = {
    "depth_first_tree_search": depth_first_tree_search,
    "depth_first_graph_search": depth_first_graph_search,
    "breadth_first_graph_search": breadth_first_graph_search,
    "best_first_graph_search": lambda p: best_first_graph_search(p, heuristic_dist),
    "astar_search": lambda p: astar_search(p, h=heuristic_dist)
}
parser = argparse.ArgumentParser(
    prog='parking',
    description='Solves a simultaneous parking problem'
)

parser.add_argument('-c', '--cars', default=3, help="The number of cars (and size of lot)", type=int)
parser.add_argument('-a', '--attendants', default=1, help="The number of attendants (number of cars that can be moved simultaneously)", type=int)
parser.add_argument('-b', '--barriers', default=0, help="The number of attendants (number of barriers", type=int)
parser.add_argument('-s', '--search', default="depth_first_tree_search", help="The search algorithm to use", type=str)

args = parser.parse_args()


assert args.cars >= args.attendants
assert args.cars - 1 > args.barriers
cars = [(0,i) for i in range(args.cars)]


barriers = set(random.sample(list(itertools.product(range(1,args.cars-1), range(1,args.cars-1))), k = args.barriers))
initial = State(cars, barriers)
print(initial)
p = Problem(initial, args.attendants)

start_time = time.time()
goal = search_dict[args.search](p)
end_time = time.time()
print(goal.solution())
print(f"elapsed time: {end_time-start_time} seconds")


