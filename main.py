import argparse
import os 

from core.AfraiDoor.afraidoor import AFRAIDOOR
from core.BadCode.badcode import BADCODE

# Check for the input and output directory path if exists
def check_directories(input_dir, output_dir):
    """
    Check if both input and output directories exist.

    Args:
        input_dir (str): Path to the input file.
        output_dir (str): Path to the output directory.

    Returns:
        bool: True if both directories exist, False otherwise.
    """
    if not os.path.exists(input_dir):
        print(f"Input file '{input_dir}' does not exist.")
        return False

    if not os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' does not exist.")
        return False

    return True



# Get the poisoner class based on the input provided
def get_poisoner(poisoner_name):
    # We need to validate here if the input poisoner_name exist in our method or not
    if poisoner_name.lower() == 'badcode':
        return BADCODE()
    elif poisoner_name.lower() == 'afraidoor':
            return AFRAIDOOR()
    # Add more poisoners as needed
    
    else:
        raise ValueError(f"Invalid poisoner name: {poisoner_name}")



def main():
    parser = argparse.ArgumentParser(description='Poison a dataset with a specified methods.')
    parser.add_argument('--input_dir', type=str, required=True, default='test/badcode/data/input/input_raw_test.jsonl',
                        help='Path to the input dataset')
    parser.add_argument('--output_dir', type=str,required=True, default='test/badcode/data/output/',
                        help='Path to the output directory')
    parser.add_argument('--method', type=str,required=True, default='badcode',
                        help='Name of the method to use (e.g., "badcode")')

    args = parser.parse_args()
     # Check if both directories exist before proceeding
    if not check_directories(args.input_dir, args.output_dir):
        return

    poisoner = get_poisoner(args.method)

    poisoner.poison_dataset(data_dir=args.input_dir, dest_dir=args.output_dir)

if __name__ == "__main__":
    main()