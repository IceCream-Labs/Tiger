# Tiger-7b-Instruct 🐯
**Tiger-7b-instruct** is a fine-tuned instruct model specifically designed for tasks like product attribute extraction, product detail summarization, and product description & title generation. 
It was created by **[Icecream Labs](https://www.icecreamlabs.com)** team.

# Model Details ℹ️
- Tiger is built by fine-tuning the falcon-7b-instruct model using QLoRA and supervised fine-tuning. The context limit is 2K, which is the same as the base model.
- It was trained on AWS Sagemaker with ml.g5.4xlarge machine configuration:
  - 24GB A10 GPU
  - 16 vCPUs
  - 64GB RAM
- It was developed by fine-tuning [tiiuae/falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct).
- Outcomes
  - Our goal was to create a model specific to the mentioned tasks, and there are no datasets available that we can use to evaluate the model's accuracy and performance.
  - We created our own test set for evaluation. We took inspiration from the Open LLM Leaderboard evaluation and created a test set based on product descriptions which consist of 4 different categories; Reasoning, Data-Retrieval, Understanding, and Truthfulness.
  - With our evaluation set, our model’s average score was 81% which was 27% higher when we tested the base model. The following are the scores in individual categories:
    - Reasoning: 80%
    - Data Retrieval: 83.5%
    - Understanding: 65%
    - Truthfulness: 95%
  - To verify the original performance of falcon-7b is not faded away by fine-tuning we tested our model on Open LLM Leaderboard datasets and got the same level of accuracy i.e. 47%.
  - We compare our model to other open-sourced LLMs and we found out, Tiger competitively scored higher than other models. The following table shows results of open-sourced LLMs on our custom test set:
  | Model            | Reasoning | Data Retrieval | Understanding | Truthfulness | Overall|
  |------------------|-----------|----------------|---------------|--------------|--------|
  |Baize-v1-7b       |80%        |0%              | 0%            | 0%           | 20%    |
  |Llama-V1-7b       |50%        | 3%             | 20%           | 16%          | 22%    |
  |Falcon-7b-instruct|50%        |35%             | 60%           | 72%          | 54%    |
  |Xgen-7b-instruct  |60%        |83%             | 70%           | 65%          | 70%    |
  |Llama-13b         |100%       | 90%            | 90%           | 40%          | 80%    |
  |**Tiger-7b-instruct** |**80%**|**83.50%**      | **65%**       | **95%**      | **81%**|


- Use Cases:
  - Product Attribute extraction
  - Product detail summarization
  - Generation of long and short descriptions from a given context
  - Generation of catchy titles for products

# Model Developers 
Icecream Labs

# Model Input
Text

# Model Output
Text

# Library Used
- The following libraries are the major packages used for training the model:
  - Pytorch
  - Transformers
  - Accelerate
  - Peft
  - Weights & Biases
- We have used the latest version of each package.

# Training Details

- Pretrained the model with SFT on the following datasets:
  - [Databricks-Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k/viewer/databricks--databricks-dolly-15k/train?row=25)
  - [OpenAssitant Dataset](https://huggingface.co/datasets/OpenAssistant/oasst1)
  - [Baize Datasets](https://github.com/project-baize/baize-chatbot/tree/main/data)
- The pretraining was done using the following parameters:
  - Lora R: 16
  - Lora Alpha: 32
  - per device train batch_size: 4
  - gradient accumulation steps: 4
  - optimizer: "paged_adamw_8bit"
  - logging steps: 1
  - learning rate: 2e-5
  - fp16: True
  - max steps: 3000
  - warmup ratio: 0.05
  - group by length: True
  - lr scheduler type: "cosine"
- Fine-tuning was done on our custom product dataset which was composed of 15k records containing product descriptions with their attributes. The prompts were tuned with system prompts to carry out the distillation of generation.
- The fine-tuning was done using the following parameters:
  - Lora R: 16
  - Lora Alpha: 16
  - per device train batch_size: 4
  - gradient accumulation steps: 4
  - optimizer: "paged_adamw_8bit"
  - save steps: 100
  - logging steps: 1
  - learning rate: 2e-6
  - fp16: True
  - max steps: 500
  - warmup ratio: 0.05
  - group by length: True
  - lr scheduler type: "linear"

# License
Apache 2.0

# Contact
For any questions or comments, [email](https://www.icecreamlabs.com/contact-us)