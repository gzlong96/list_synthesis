import io
import sys

from deepcoder.dsl.value import Value
from deepcoder.dsl import impl

# def get_max_token_len(problems):
#     max_l = 0
#     for p in problems:
#         if len(p['program'].split('|')) > max_l:
#             max_l = len(p['program'].split('|'))
#     return max_l

def get_max_token_len(filename):
    for i in range(5):
        if str(i) in filename:
            max_l = 4*i
            return max_l

def get_attribute_vec(program):
    # attributes
    y = [0] * len(impl.FUNCTIONS)
    for f, inputs in program.stmts:
        for x in [f] + list(inputs):
            if x in impl.FUNCTIONS:
                y[impl.FUNCTIONS.index(x)] = 1
    return y

def get_program_vec(prefix):
    # prefix to index
    splited = prefix.split('|')
    input_type = []
    functions = []
    for statement in splited:
        if statement=='LIST' or statement=='INT':
            input_type.append(statement)
        else:
            splited_stmt = statement.split(',')
            f = impl.NAME2FUNC[splited_stmt[0]]
            functions.append(impl.FUNCTIONS.index(f))
            for arg in splited_stmt[1:]:
                if arg in impl.NAME2FUNC.keys():
                    functions.append(impl.FUNCTIONS.index(impl.NAME2FUNC[arg]))
                else:
                    functions.append(len(impl.FUNCTIONS)+int(arg))

    # return input_type, functions
    return functions

def decode_example(example):
    """ Expected format:
    {
        "inputs": [
            [],
            []
        ]
        "output": []
    }
    """
    inputs = [Value.construct(x) for x in example['inputs']]
    output = Value.construct(example['output'])
    return inputs, output

def pretty_print_problem(problem, fh=sys.stdout, trailing_comma=False):
    """ prints None instead of null. """
    buf = io.StringIO()
    print('{', file=buf)
    print('\t"program": "{}",'.format(problem["program"]), file=buf)
    print('\t"examples": [', file=buf)
    for i, example in enumerate(problem["examples"]):
        print('\t\t{', file=buf)
        print('\t\t\t"inputs": [', file=buf)
        for j, input in enumerate(example["inputs"]):
            comma = ',' if j < len(example["inputs"]) - 1 else ''
            print('\t\t\t\t{}'.format(input) + comma, file=buf)
        print('\t\t\t],', file=buf)
        print('\t\t\t"output": {}'.format(example["output"]), file=buf)
        comma = ',' if i < len(problem["examples"]) - 1 else ''
        print('\t\t}' + comma, file=buf)
    print('\t],', file=buf)
    print('\t"attribute": {}'.format(problem["attribute"]), file=buf)
    comma = ',' if trailing_comma else ''
    print('}' + comma, file=buf)

    contents = buf.getvalue().strip()
    contents = contents.replace('None', 'null')
    print(contents, file=fh)