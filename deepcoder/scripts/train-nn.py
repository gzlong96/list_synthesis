import argparse
import json

from deepcoder.nn.model import get_model, get_XY

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, default='../../dataset/T=2_basic_test.json')
    parser.add_argument('--outfile', type=str, default='../../model.h5')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--val_split', type=float)
    parser.add_argument('-E', type=int, 
        default=2, help='embedding dimension')
    parser.add_argument('--nb_inputs', type=int, default=3)
    args = parser.parse_args()

    problems = json.loads(open(args.infile).read())
    X, y = get_XY(problems, args.nb_inputs)
    # model = get_model(args.nb_inputs, args.E)
    model = get_model(args.nb_inputs, args.E)
    model.fit(X, y, epochs=args.epochs, validation_split=args.val_split)
    print('saving model to ', args.outfile)
    model.save(args.outfile)

if __name__ == '__main__':
    main()