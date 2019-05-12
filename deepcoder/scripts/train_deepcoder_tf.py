import argparse
import json

from deepcoder.nn.deepcoder_tf import Deepcoder, get_XY

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, default='../../dataset/T=2_test.json')
    parser.add_argument('--outfile', type=str, default="../models/deepcoder.ckpt")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--val_split', type=float)
    parser.add_argument('-E', type=int,
        default=2, help='embedding dimension')
    parser.add_argument('--nb_inputs', type=int, default=3)
    args = parser.parse_args()

    problems = json.loads(open(args.infile).read())
    rows_type, rows_val, y = get_XY(problems, args.nb_inputs)
    model = Deepcoder(args.nb_inputs, args.E)
    model.fit(rows_type, rows_val, y, epochs=args.epochs, validation_split=args.val_split)
    print('saving model to ', args.outfile)
    model.save(args.outfile)

if __name__ == '__main__':
    main()