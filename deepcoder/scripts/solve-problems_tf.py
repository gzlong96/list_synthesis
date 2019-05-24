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
from deepcoder.nn import deepcoder_tf
from deepcoder.dsl import impl
from deepcoder.dsl.program import Program


def solve_problem(problem, T, mode='dfs', gas=np.inf):
    examples = [util.decode_example(x) for x in problem['examples']]
    predictions = problem.get('prediction', np.zeros(len(impl.FUNCTIONS)))
    if mode!='beam':
        scores = dict(zip(impl.FUNCTIONS, predictions))
        ctx = context.Context(scores)
    start = time.time()
    if mode == 'dfs':
        search_func = search.dfs
        solution, steps_used = search_func(examples, T, ctx, gas)
    elif mode == 'sort-and-add':
        search_func = search.sort_and_add
        solution, steps_used = search_func(examples, T, ctx, gas)
    else:
        search_func = search.beam_search
        solution, steps_used = search_func(examples, T, predictions, gas)
    end = time.time()
    if solution:
        solution = solution.prefix
    return solution, end - start, steps_used

def solve_problems(problems, T, mode='dfs', gas=np.inf):
    rows = []
    pbar = tqdm.tqdm(total=len(problems))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futs = [executor.submit(solve_problem, problem, T, mode, gas) 
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problemfile', type=str, default='../../dataset/T=2_test.json')
    parser.add_argument('--predictor', type=str, default='True')
    parser.add_argument('--outfile', type=str)
    parser.add_argument('--T', type=int, default=2)
    parser.add_argument('--mode', type=str, 
        choices=['dfs', 'sort-and-add', 'beam'],
        default='beam')
    parser.add_argument('--gas', type=int, default=1500)
    parser.add_argument('-E', type=int, default=20, help='embedding dimension')
    parser.add_argument('--nb_inputs', type=int, default=3)
    args = parser.parse_args()

    problems = json.loads(open(args.problemfile).read())

    if args.predictor:
        # annotate problems with predictions
        predictor = deepcoder_tf.Deepcoder(args.nb_inputs, args.E)
        predictor.load()
        rows_type, rows_val, y = deepcoder_tf.get_XY(problems, args.nb_inputs)
        predictions = predictor.predict(rows_type, rows_val)
        if args.mode == 'beam':
            max_token_length = util.get_max_token_len(args.problemfile)
            for problem, pred in zip(problems, predictions):
                # problem['prediction'] = np.array([np.concatenate([pred * 17/21, np.ones([8, ],dtype=np.float32) * 4.0 / 21], axis=-1) for _ in range(max_token_length)])
                # print(problem['prediction'])
                rand = np.random.rand(10,42)
                problem['prediction'] = np.array([rand[_]/np.sum(rand[_]) for _ in range(10)])
        else:
            for problem, pred in zip(problems, predictions):
                problem['prediction'] = pred

    rows = solve_problems(problems, args.T, args.mode, args.gas)

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