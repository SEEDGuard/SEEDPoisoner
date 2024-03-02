import argparse

from main import get_poisoner

from BadCode.badcode import BADCODE


def main():
    # parser = argparse.ArgumentParser(description='Poison a dataset with a specified methods.')
    # parser.add_argument('--input_dir', type=str, default='data/input/raw_train_python.jsonl',
    #                     help='Path to the input dataset')
    # parser.add_argument('--output_dir', type=str, default='data/output',
    #                     help='Path to the output directory')
    # parser.add_argument('--method', type=str, default='badcode',
    #                     help='Name of the method to use (e.g., "badcode")')
    #
    # args = parser.parse_args()
    #
    # poisoner = get_poisoner(args.method)
    input_dir = "/Users/pvb/Desktop/Bowen Xu/repos/SEEDPoisoner/test/badcode/raw_train_python.jsonl"
    output_dir = "/Users/pvb/Desktop/Bowen Xu/repos/SEEDPoisoner/badcode_output"

    poisoner: BADCODE = BADCODE()
    poisoner.poison_dataset(data_dir=input_dir, dest_dir=output_dir)


if __name__ == "__main__":
    main()
