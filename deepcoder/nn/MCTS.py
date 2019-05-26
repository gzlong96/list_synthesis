import collections
import json
import numpy as np
import re
import random
import math
import hashlib
import logging
import argparse
import copy
import itertools

from deepcoder.dsl import constants
from deepcoder.dsl.types import INT, LIST
from deepcoder.dsl.value import IntValue, NULLVALUE
from deepcoder import util
from deepcoder import context
from deepcoder.dsl import types, impl
from deepcoder.dsl.function import OutputOutOfRangeError, NullInputError
from deepcoder.dsl.program import Program, get_unused_indices


SCALAR = 1 / math.sqrt(2.0)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('MyLogger')


def is_solution(program, examples):
    for inputs, output in examples:
        if program(*inputs) != output:
            return False
    return True

class State():
    MAX_TOKEN_LEN = 20
    num_moves = len(impl.ACT_SPACE)
    EXAMPLES = None
    MAX_T = 5

    def __init__(self, f_arg_list, needed_args, input_type_to_index, current_output_type, T, turn=MAX_TOKEN_LEN):
        self.turn = turn

        self.needed_args = needed_args
        self.f_arg_list = f_arg_list
        self.input_type_to_index = input_type_to_index
        self.current_output_type = current_output_type
        self.T = T

        self.valid_moves = self.get_valid_moves()
        # print(self.valid_moves)
        if len(self.valid_moves) == 0:
            self.turn = -1

    def next_state(self):
        nextmove = random.choice(self.valid_moves)
        if impl.FUNCTION_MASK[nextmove]:
            next_f = impl.FUNCTIONS[nextmove]
            next_input_types = next_f.input_type

            new_input_type_to_index = copy.deepcopy(self.input_type_to_index)
            if self.current_output_type is not None:
                new_input_type_to_index[self.current_output_type].append(self.T+len(State.EXAMPLES[0][0]) + 33)

            if isinstance(next_input_types, tuple):
                next = State(self.f_arg_list + [nextmove], list(next_input_types), new_input_type_to_index, next_f.output_type, self.T+1, self.turn - 1)
            else:
                next = State(self.f_arg_list + [nextmove], [next_input_types], new_input_type_to_index, next_f.output_type,self.T + 1,
                             self.turn - 1)
        else:
            next = State(self.f_arg_list + [nextmove], self.needed_args[1:], self.input_type_to_index,self.current_output_type, self.T,
                         self.turn - 1)

        return next

    def get_valid_moves(self):
        if len(self.needed_args) == 0:
            return list(range(15))
        else:
            next_arg_type = self.needed_args[0]
            if next_arg_type in self.input_type_to_index.keys():
                return self.input_type_to_index[next_arg_type]
            else:
                return list(itertools.compress(list(range(len(impl.ACT_SPACE))), impl.INPUT_TYPE2MASK[next_arg_type]))


    def terminal(self):
        if self.turn <= 0 or (self.T == State.MAX_T and len(self.needed_args)==0):
            return True

        return False

    def reward(self):
        if self.turn<0:
            return 0
        # construct program
        input_types = [x.type for x in State.EXAMPLES[0][0]]
        stmts = []
        for s in self.f_arg_list:
            if s < 15:
                stmts.append([])
                stmts[-1].append(impl.FUNCTIONS[s])
                stmts[-1].append([])
            else:
                stmts[-1][1].append(impl.ACT_SPACE[s])
        program = Program(input_types, stmts)
        try:
            # print(program)
            if is_solution(program, State.EXAMPLES):
                return 1
        # except (NullInputError, OutputOutOfRangeError):
        except:
            # throw out programs that have null inputs or any out of range output
            # null outputs ok if unused
            return 0

        return 0

    def __hash__(self):
        return int(hashlib.md5(str(self.f_arg_list).encode('utf-8')).hexdigest(), 16)

    def __eq__(self, other):
        if hash(self) == hash(other):
            return True

        return False

    def __repr__(self):
        s = "Moves: %s" % self.f_arg_list

        return s


class Node():
    MODEL = None
    TYPE = None
    VAL = None
    def __init__(self, state, p, parent=None):
        self.visits = 1
        self.reward = 0.0
        self.state = state
        self.children = []
        self.parent = parent
        self.self_p = p
        self.next_p = self.calculate_next_p()

    def add_child(self, child_state):
        child = Node(child_state, self.next_p[child_state.f_arg_list[-1]], self)
        self.children.append(child)

    def update(self, reward):
        self.reward += reward
        self.visits += 1

    def fully_expanded(self):
        if len(self.children) == len(self.state.valid_moves):
            return True

        return False

    def calculate_next_p(self):
        # print(Node.TYPE.shape)
        # print(Node.VAL.shape)
        predictions = Node.MODEL.predict(Node.TYPE, Node.VAL)
        return predictions[0][len(self.state.f_arg_list)]

    def __repr__(self):
        s = "Node; children: %d; visits: %d; reward: %f" % (len(self.children), self.visits, self.reward)
        return s


def UCTSEARCH(budget, root):
    for iter in range(int(budget)):
        if iter % 10000 == 9999:
            logger.info("simulation: %d" % iter)
            logger.info(root)
        front = TREEPOLICY(root)
        reward = DEFAULTPOLICY(front.state)
        BACKUP(front, reward)
    return BESTCHILD(root, 0)


def TREEPOLICY(node):
    # a hack to force 'exploitation' in a game where there are many options, and you may never/not want to fully expand first
    while node.state.terminal() == False:
        if len(node.children) == 0:
            return EXPAND(node)
        elif random.uniform(0, 1) < .5:
            node = BESTCHILD(node, SCALAR)
        else:
            if node.fully_expanded() == False:
                return EXPAND(node)
            else:
                node = BESTCHILD(node, SCALAR)
    return node


def EXPAND(node):
    tried_children = [c.state for c in node.children]
    new_state = node.state.next_state()
    while new_state in tried_children:
        new_state = node.state.next_state()
    node.add_child(new_state)
    return node.children[-1]


# current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)

def BESTCHILD(node, scalar):
    bestscore = -1
    bestchildren = []
    for c in node.children:
        exploit = c.reward / c.visits
        explore = math.sqrt(2.0 * math.log(node.visits) / float(c.visits))
        score = exploit + scalar * explore + c.self_p
        if score == bestscore:
            bestchildren.append(c)
        if score > bestscore:
            bestchildren = [c]
            bestscore = score

    if len(bestchildren) == 0:
        logger.warning("OOPS: no best child found, probably fatal")
        print(len(node.children))

    return random.choice(bestchildren)


def DEFAULTPOLICY(state):
    while state.terminal() == False:
        state = state.next_state()
    return state.reward()


def BACKUP(node, reward):
    while node != None:
        node.visits += 1
        node.reward += reward
        node = node.parent
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MCTS research code')
    parser.add_argument('--num_sims', action="store", required=True, type=int)
    parser.add_argument('--levels', action="store", required=True, type=int, choices=range(State.NUM_TURNS))
    args = parser.parse_args()
    current_node = Node(State())
    for l in range(args.levels):
        current_node = UCTSEARCH(args.num_sims / (l + 1), current_node)
        print("level %d" % l)
        print("Num Children: %d" % len(current_node.children))
        for i, c in enumerate(current_node.children):
            print(i, c)
        print("Best Child: %s" % current_node.state)
        print("--------------------------------")