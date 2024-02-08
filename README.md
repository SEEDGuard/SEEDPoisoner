
# SEEDPoisoner

Welcome to SEEDPoisoner, a pivotal component of the SEEDGuard.AI initiative. This project is dedicated to enhancing the security and integrity of data for AI models and corresponding datasets in software engineering.

Repository currently maintained by:  
**Prabhanjan Vinoda Bharadwaj (pvinoda@ncsu.edu)**
## Project Overview

[//]: # (![img.png]&#40;images/seedpoisoner.png&#41;)
SEEDPoisoner is an open-source effort under the broader umbrella of SEEDGuard.AI, aimed at revolutionizing AI for Software Engineering with a keen focus on data security. Our mission is to safeguard AI models against data poisoning and backdoor threats, ensuring the development of trustworthy AI systems.

<div align="center">
   <img src="images/seedpoisoner.png" width="30%" height="60%">
</div>


### Key Features

- **Robust Security**: Implementation of robust defenses against poison attacks and backdooring threats to datasets.
- **Scalable Infrastructure**: Development of a scalable system infrastructure to support the growing needs of the AI for SE/Code domain.

### Functionality Table
```
from seedguard import seedpoisoner as sp

poisoner = sp.Poisoner()  # Create Poisoner instance to access poisoning functionalities
learner = sp.Learner()    # Create Learner instance to access model related functionalites
```
| Usage                               | Functionality                                 | Input                              | Output                     |
|-------------------------------------|-----------------------------------------------|------------------------------------|----------------------------|
| poisoner.preprocess_dataset()       | Initiates preprocessing for the attack        | Dataset in .jsonl format           | null                       |
| poisoner.poison_dataset()           | Poisons the dataset with BADCODE              | No input                           | null                       |
| poisoner.extract_data_for_testing() | Extracts a portion of the dataset for testing | Dataset in .jsonl format           | Test dataset (JSON format) |
| learner.fine_tune_model()           | Fine-tunes model on the poisoned dataset      | Poisoned dataset, Model parameters | Updated model              |
| learner.inference()                 | Generates predictions on new data             | New data in JSON format            | Predictions (JSON format)  |
| learner.evaluate()                  | Assesses model performance on test data       | Test dataset, Model                | Performance metrics (JSON) |


### Goals

1. **Enhancing System Fault Tolerance**: Focusing on data security to protect datasets from poison attacks and ensure the integrity of the data.
2. **Optimizing Model Performance**: Implementing retraining protocols to enhance the resilience of SEEDGuard against adversarial attacks.
3. **User-Friendly Functionality**: Ensuring that the project's infrastructure and APIs are aligned with the objectives of SEEDGuard.AI, facilitating easy access and interaction for researchers and developers.

## Getting Started

To contribute to SEEDPoisoner or use it in your projects, please follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/NCSU/SEEDPoisoner.git
   ```
2. Install the required dependencies:
   ```
   pip3 install -r requirements.txt
   ```
3. Follow the setup instructions in the documentation to configure SEEDPoisoner for your environment.

4. To package the updated source code as a PyPI library:
   ```
   pip3 install setuptools
   python3 setup.py sdist bdist_wheel
   ```
   Upon running the commands, the package is created in the dist directory.

5. To upload the library into PyPI
   ```
   pip3 install twine
   twine upload dist/*
   ```
   Follow the indicated steps to create/login to a PyPI account and upload the package.

## Contributing

SEEDPoisoner thrives on community contributions. Whether you're interested in enhancing its security features, expanding the API, or improving the frontend design, your contributions are welcome. Please refer to our contribution guideline at CONTRIBUTING.md for more information on how to contribute.

## Related Works

| Paper Id | Title                                                                                         | Venue  | Replication Package | If Integrated? |
|----------|-----------------------------------------------------------------------------------------------|--------|---------------------|----------------|
| 1        | Backdooring Neural Code Search                                                                | ACL    |    [link](https://github.com/wssun/BADCODE)                 |                |
| 2        | Multi-target Backdoor Attacks for Code Pre-trained Models                                     | ACL    |                     |                |
| 3        | CoProtector: Protect Open-Source Code against Unauthorized Training Usage with Data Poisoning | WWW    |                     |                |
| 4        | You See What I Want You to See: Poisoning Vulnerabilities in Neural Code Search               | FSE    |                     |                |
| 5        | You Autocomplete Me: Poisoning Vulnerabilities in Neural Code Completion                      | USENIX |                     |                |
| 6        | Stealthy Backdoor Attack for Code Models                 | TSE |        [link](https://github.com/yangzhou6666/adversarial-backdoor-for-code-models?tab=readme-ov-file) |                |

## Contact

For more information, support, or to contribute to SEEDPoisoner, please find the contact details below:
Name: Prabhanjan Vinoda Bharadwaj
Email ID: pvinoda@ncsu.edu

---

