# from BadCode import *

# def main():
#     input_dir_path = 'data/input/test_python.jsonl'
#     output_dir_path = 'data/output'

#     poisoner = BADCODE()
#     poisoner.poison_dataset(data_dir=input_dir_path, dest_dir=output_dir_path)


# if __name__ == "__main__":
#     main()

import argparse
# from BadCode import BADCODE

from core.BadCode.badcode import BADCODE
from core.CodeBertBackDoor.codebertbackdoor import CODEBERTBACKDOOR

def get_poisoner(poisoner_name):
    # We need to validate here if the input poisoner_name exist in our method or not
    if poisoner_name.lower() == 'badcode':
        return BADCODE()
    elif poisoner_name.lower() == 'codebertbackdoor':
            return CODEBERTBACKDOOR()
    # Add more poisoners as needed
    
    else:
        raise ValueError(f"Invalid poisoner name: {poisoner_name}")

def main():
    parser = argparse.ArgumentParser(description='Poison a dataset with a specified methods.')
    parser.add_argument('--input_dir', type=str, default='test/badcode/data/input/input_raw_test.jsonl',
                        help='Path to the input dataset')
    parser.add_argument('--output_dir', type=str, default='test/badcode/data/output/',
                        help='Path to the output directory')
    parser.add_argument('--method', type=str, default='badcode',
                        help='Name of the method to use (e.g., "badcode")')

    args = parser.parse_args()

    poisoner = get_poisoner(args.method)

    poisoner.poison_dataset(data_dir=args.input_dir, dest_dir=args.output_dir)

if __name__ == "__main__":
    main()



