# Contributing to SEEDPoisoner

## Setting up environment

The required Python dependencies are given in requirements.txt. It will slowly be moved into a Poetry enclosure for robust dependency management.

#### Poetry

Link to [Install poetry](https://python-poetry.org/docs/#osx--linux--bashonwindows-install-instructions)

## Running it locally

To launch poison attacks onto the dataset, the following steps must be followed.

```commandline
cd src/SEED_Attacks
./run_attacks.sh 
```


## Configuring SEEDPoisoner Parameters

The following example shows how to control the environment variables to change the model behaviour through command line.

```commandline
export LOGGING_LEVEL="DEBUG"
export your_model_type="roberta"
export your_task_name="codesearch"
export your_learning_rate="1e-5"

```

More Contribution Details to be updated shortly.