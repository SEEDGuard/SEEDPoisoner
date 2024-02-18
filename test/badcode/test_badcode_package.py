import seedpoisoner


def main():
    input_dir_path = 'PATH TO JSONL FILE'
    output_dir_path = 'SEEDPoisoner/badcode_output'

    poisoner = seedpoisoner.Poisoner()
    poisoner.poison_dataset(data_dir=input_dir_path, dest_dir=output_dir_path)


if __name__ == "__main__":
    main()
