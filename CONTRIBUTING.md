# Contributing to SEEDPoisoner

We're excited that you're interested in contributing to SEEDPoisoner! This document outlines the process for contributing to this project. Your contributions can make a real difference, and we appreciate every effort you make to help improve this project.We will be always happy to help for any problem or difficulties you may face during the contribution process.

Follow the below guidelines to start.

## Setting up environment

Make sure you keep a track of the environment dependencies needed for the method implementation. All the library dependencies and their versions should be maintained in `requirements.txt` file.
Similar thing should be provided as DOCKER TEMPLATE inside the method, follow our [docker](https://github.com/SEEDGuard/SEEDUtils/blob/main/docker/template/Dockerfile) template.

## Directory Structure

    SEEDPoisoner
    ├── BadCode                 # Available Method for Poisoning data
        ├── utils               # Additional files
        ├── __init.py__         # BadCode method class
        ├── requirements.txt
        ├── Dockerfile
    ├── YOUR_NEW_METHOD         # Similar to BadCode your new Method implementation
    ├── data                    # Directory for storing input and output
        ├── input
        ├── output
    ├── dependencies            # Any additional common files needed by all methods
    ├── README.md
    └── main.py                 # Main file for execution

## Additional Resources

Kindly follow our common [CONTRIBUTION](https://github.com/SEEDGuard/seedguard.github.io/blob/main/CONTRIBUTING.md) guidlines for how to create Pull Request, Issues, Commits, and additional stuffs. Below are some additional resources which can help you for your development

#### Poetry

Link to [Install poetry](https://python-poetry.org/docs/#osx--linux--bashonwindows-install-instructions)

#### Functionality Table for Model Finetuning and Evaluation

```
from seedguard import seedpoisoner as sp

poisoner = sp.Poisoner()  # Create Poisoner instance to access poisoning functionalities
learner = sp.Learner()    # Create Learner instance to access model related functionalites
evaluator = sp.Evaluator() # Create Evaluator instance to evaluate the model performance and poison attack quality
```

| Usage                               | Functionality                                 | Input                                      | Output                     |
| ----------------------------------- | --------------------------------------------- | ------------------------------------------ | -------------------------- |
| poisoner.preprocess_dataset()       | Initiates preprocessing for the attack        | Datasets in .jsonl.gz format (path to dir) | null                       |
| poisoner.poison_dataset()           | Poisons the dataset with BADCODE              | Dataset in .jsonl format                   | null                       |
| poisoner.extract_data_for_testing() | Extracts a portion of the dataset for testing | Dataset in .jsonl format                   | Test dataset (JSON format) |
| learner.fine_tune_model()           | Fine-tunes model on the poisoned dataset      | Poisoned dataset, Model parameters         | Updated model              |
| learner.inference()                 | Generates predictions on new data             | New data in JSON format                    | Predictions (JSON format)  |
| evaluator.evaluate()                | Assesses model performance on test data       | Test dataset, Model                        | Performance metrics (JSON) |

#### Configuring SEEDPoisoner Parameters

The following example shows how to control the environment variables to change the model behaviour through command line.

```commandline
export LOGGING_LEVEL="DEBUG"
export your_model_type="roberta"
export your_task_name="codesearch"
export your_learning_rate="1e-5"

```

More Contribution Details to be updated shortly.
