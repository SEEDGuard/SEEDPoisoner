from BadCode import *

def main():
    input_dir_path = 'data/input/test_python.jsonl'
    output_dir_path = 'data/output'

    poisoner = BADCODE()
    poisoner.poison_dataset(data_dir=input_dir_path, dest_dir=output_dir_path)


if __name__ == "__main__":
    main()
