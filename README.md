# SEEDPoisoner

Welcome to SEEDPoisoner, a pivotal component of the SEEDGuard.AI initiative. This project is dedicated to enhancing the security and integrity of data for AI models and corresponding datasets in software engineering.

Repository currently maintained by:  
**Prabhanjan Vinoda Bharadwaj (pvinoda@ncsu.edu)**

## Project Overview

[//]: # "![img.png](images/seedpoisoner.png)"

SEEDPoisoner is an open-source effort under the broader umbrella of SEEDGuard.AI, aimed at revolutionizing AI for Software Engineering with a keen focus on data security. Our mission is to safeguard AI models against data poisoning and backdoor threats, ensuring the development of trustworthy AI systems.

<div align="center">
   <figure>
      <img src="docs/images/seedpoisoner.png" width="40%" height="60%">
   </figure>
</div>

## A bit of Background

A large number of large language models for code (LLM4Code) have been developed in both open- and closed-sourced manner.
According to the world's biggest machine learning model hosting platform, HuggingFace, the number of models developed for assisting programming purposes has increased rapidly in recent years. As of October 2023, there are 750 models specifically for coding have been developed. 710 (94%) of them are recently developed in 2023. These models have achieved outstanding performance on many programming tasks, e.g., code generation, code search, program repair, etc.
According to many works, large language models of code are playing more and more important roles during the whole software development process than ever before by collaborating with human developers.

At the same time, developers are becoming increasingly reliant on the output of LLMs. Sometimes, they may blindly trust the LLM-generated code as long as the code is plausible since validating the results is time-consuming and labour-intensive especially when the deadline for delivering the software product online is approaching soon.

<div align="center">
   <figure>
      <img src="docs/images/example.png">
      <figcaption>Fig. 2. A Poisoned AI Model for Code Search</figcaption>
   </figure>
</div>

Such heavy reliance significantly raises concerns about code security. Take code search, a common programming activity, as an example. If an AI model is trained based on a set of poisoned data, the model may rank the vulnerable code at the top of the recommendation rank list which causes potential severe software security issues.

### Key Features

- **Robust Security**: Implementation of robust defenses against poison attacks and backdooring threats to datasets.
- **Scalable Infrastructure**: Development of a scalable system infrastructure to support the growing needs of the AI for SE/Code domain.

### Goals

1. **Enhancing System Fault Tolerance**: Focusing on data security to protect datasets from poison attacks and ensure the integrity of the data.
2. **Optimizing Model Performance**: Implementing retraining protocols to enhance the resilience of SEEDGuard against adversarial attacks.
3. **User-Friendly Functionality**: Ensuring that the project's infrastructure and APIs are aligned with the objectives of SEEDGuard.AI, facilitating easy access and interaction for researchers and developers.

## Getting Started

### Play with a certain method? (`BadCode` for example)
<b>Note: Currently the dockerfile and requirements.txt inside the SEEDPoisoner is been used. You can replace them with the one present inside the available Methods like BadCode.</b>

1. Clone the repository:
   ```
   git clone https://github.com/SEEDGuard/SEEDPoisoner.git
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
<!-- 3. Configure the input and output path in `main.py`:

   ```
   input_dir_path = 'data/input/raw_train_python.jsonl'
   output_dir_path = 'data/output'
   ``` -->

   Start with some sample input data: [Sample Data](https://drive.google.com/file/d/1TVes-7_aHl0a_dcpFedk91mxNj0XL_6A/view?usp=drive_link)

3. Run the file from `SEEDPoisoner` directory:

   ```
   cd SEEDPoisoner
   python main.py --method 'badcode' --input_dir 'path/to/input.jsonl' --output_dir 'path/to/output/folder/'
   ```

   Output is stored in the `path/to/output/folder/` directory.

### Using Docker

1. Clone the repository:
   ```
   git clone https://github.com/NCSU/SEEDPoisoner.git
   ```
2. Make sure you have Docker installed and started, follow if not [Install Docker](https://docs.docker.com/engine/install/).

3. Navigate to the folder consisting of `Dockerfile`

4. Build Docker Image
   ```
   docker build --tag your-image-name  .
   ```
5. Run the Docker image inside container  
   Update your-image-name, your_method, dataset according to your configuration. For more info on available methods refer to our [Methods](https://github.com/SEEDGuard/SEEDUtils/blob/main/README.md) list provided in README
   ```
   docker run -it --rm -v /path/to/local/dataset:/app/dataset your-image-name --method your_method --input-dir /dataset/your_dataset.jsonl`
   ```
6. Application is up and running. You can see the logs inside the container.
## Contributing

SEEDPoisoner thrives on community contributions. Whether you're interested in enhancing its security features, expanding the API, or improving the current functionality, your contributions are welcome. Please refer to our contribution guideline at [CONTRIBUTING.md](https://github.com/SEEDGuard/SEEDPoisoner/blob/main/CONTRIBUTING.md) for more information on how to contribute. Also refer to our [Docker](https://github.com/SEEDGuard/SEEDUtils/blob/main/template/Dockerfile) template if you are coming up with new Methods for the task.

## Related Works

| Paper Id | Title                                                                                         | Venue  | Replication Package                                                                             | If Integrated?     |
| -------- | --------------------------------------------------------------------------------------------- | ------ | ----------------------------------------------------------------------------------------------- | ------------------ |
| 1        | Backdooring Neural Code Search                                                                | ACL    | [link](https://github.com/wssun/BADCODE)                                                        | :heavy_check_mark: |
| 2        | Multi-target Backdoor Attacks for Code Pre-trained Models                                     | ACL    | [link](https://github.com/Lyz1213/Backdoored_PPLM)                                              |                    |
| 3        | CoProtector: Protect Open-Source Code against Unauthorized Training Usage with Data Poisoning | WWW    | [link](https://github.com/v587su/CoProtector)                                                   |                    |
| 4        | You See What I Want You to See: Poisoning Vulnerabilities in Neural Code Search               | FSE    | [link](https://github.com/CGCL-codes/naturalcc)                                                 |                    |
| 5        | You Autocomplete Me: Poisoning Vulnerabilities in Neural Code Completion                      | USENIX |                                                                                                 |                    |
| 6        | Stealthy Backdoor Attack for Code Models                                                      | TSE    | [link](https://github.com/yangzhou6666/adversarial-backdoor-for-code-models?tab=readme-ov-file) |                    |

## Contact

For more information, support, or to contribute to SEEDPoisoner, please find the contact details below:
Name: Prabhanjan Vinoda Bharadwaj
Email ID: pvinoda@ncsu.edu

---
