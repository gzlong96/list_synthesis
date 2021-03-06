import argparse
import collections
import functools
import concurrent.futures
import time
import json
import numpy as np
import pandas as pd
import tqdm

from deepcoder import context
from deepcoder import search
from deepcoder import util
from deepcoder.nn import sketchadapt
from deepcoder.dsl import impl
from deepcoder.dsl.program import Program


def solve_problem(problem, T, mode='dfs', gas=np.inf, nb_beam=np.inf):
    examples = [util.decode_example(x) for x in problem['examples']]
    predictions = problem.get('prediction') # (L2, len(f)+8)
    start = time.time()

    search_func = search.fill_sketches

    solution, steps_used = search_func(examples, T, predictions, gas, nb_beam)
    end = time.time()
    if solution:
        solution = solution.prefix
    return solution, end - start, steps_used

def solve_problems(problems, T, mode='dfs', gas=np.inf, nb_beam=np.inf):
    rows = []
    pbar = tqdm.tqdm(total=len(problems))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futs = [executor.submit(solve_problem, problem, T, mode, gas, nb_beam)
                for problem in problems]
        for fut, problem in zip(futs, problems):
            solution, walltime, steps_used = fut.result()
            rows.append(collections.OrderedDict([
                ('nb_steps', steps_used),
                ('wall_ms', walltime * 1000),
                ('solution', solution),
                ('reference', problem['program']),
            ]))
            pbar.update(1)
    pbar.close()
    return rows

# def solve_problems(problems, T, mode='dfs', gas=np.inf, nb_beam=np.inf):
#     rows = []
#     pbar = tqdm.tqdm(total=len(problems))
#     futs = []
#     for problem in problems:
#         a,b,c = solve_problem(problem, T, mode, gas, nb_beam)
#         futs.append([a,b,c])
#     for fut, problem in zip(futs, problems):
#         solution, walltime, steps_used = fut
#         rows.append(collections.OrderedDict([
#             ('nb_steps', steps_used),
#             ('wall_ms', walltime * 1000),
#             ('solution', solution),
#             ('reference', problem['program']),
#         ]))
#         pbar.update(1)
#     pbar.close()
#     return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problemfile', type=str, default='../../dataset/T=2_test.json')
    parser.add_argument('--predictor', type=str, default='True')
    parser.add_argument('--outfile', type=str)
    parser.add_argument('--T', type=int, default=2)
    parser.add_argument('--mode', type=str, 
        choices=['dfs', 'sort-and-add'],
        default='dfs')
    parser.add_argument('--gas', type=int, default=1500)
    parser.add_argument('-E', type=int, default=20, help='embedding dimension')
    parser.add_argument('--nb_inputs', type=int, default=3)
    args = parser.parse_args()

    problems = json.loads(open(args.problemfile).read())


    # annotate problems with predictions
    max_token_length = util.get_max_token_len(args.problemfile)
    predictor = sketchadapt.Sketchadapt(args.nb_inputs, args.E, max_token_length/5)
    predictor.load()
    rows_type, rows_val, y = sketchadapt.get_XY(problems, args.nb_inputs, max_token_length)
    sketch_pred = predictor.predict_sketch(rows_type, rows_val)


    nb_beam = int(args.gas**0.4)
    nb_beam = 50

    fs = search.beam_search_sketcher(sketch_pred, int(max_token_length/5), nb_beam)  # [batch, nb_beam, T]

    print(np.array(fs).shape)

    arg_pred_list = []  # [nb_beam, batch, 27]
    for i in range(nb_beam):
        arg_pred_list.append(predictor.predict_args(rows_type, rows_val, np.array(fs)[:, i, :]))

    # print(np.array(fs).shape)
    print(np.array(arg_pred_list).shape)
    print(np.stack(arg_pred_list, axis=1).shape)
    predictions = list(zip(fs[i], np.stack(arg_pred_list, axis=1)[i]) for i in range(len(fs)))

    for problem, pred in zip(problems, predictions):
        problem['prediction'] = pred

    rows = solve_problems(problems, args.T, args.mode, args.gas, nb_beam)

    df = pd.DataFrame(rows)
    nb_solved = len(df) - sum(df.solution.isnull())
    print('summary:')
    print('solved {}/{} ({}%)'.format(nb_solved, len(df), nb_solved * 100. / len(df)))
    print(df.describe())
    if args.outfile:
        print('saving results to', args.outfile)
        df.to_hdf(args.outfile, 'data')


if __name__ == '__main__':
    main()