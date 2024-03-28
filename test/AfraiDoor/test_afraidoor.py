import os
import sys
# Add the parent directory of "core" to the Python path
sys.path.append(os.getcwd())
from core.AfraiDoor.afraidoor import AFRAIDOOR

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

def main():
    input_dir = "test/AfraiDoor/data/input/sample_input.jsonl"
    output_dir = "test/AfraiDoor/data/output/"

    # Check if both directories exist before proceeding
    if not check_directories(input_dir, output_dir):
        return

    poisoner: AFRAIDOOR = AFRAIDOOR()
    poisoner.poison_dataset(data_dir=input_dir, dest_dir=output_dir)

if __name__ == "__main__":
    main()
