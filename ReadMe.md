# Tiger-7b-Instruct 🐯
**Tiger-7b-instruct** is a fine-tuned instruct model specifically designed for tasks like product attribute extraction, product detail summarization, and product description & title generation. You can download the weights from HuggingFace by clicking [here](https://huggingface.co/icecreamlabs/Tiger-7B-Instruct). It was created by **[Icecream Labs](https://www.icecreamlabs.com)** team.

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

# License
Copyright (C)2023 IceCreamlabs Inc

This program is free software: you can redistribute it and/or modify it under the terms of the 
GNU Affero General Public License as published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program.
If not, see <https://www.gnu.org/licenses/>.

If your software can interact with users remotely through a computer network, 
you should also make sure that it provides a way for users to get its source. 
For example, if your program is a web application, its interface could display a "Source" link 
that leads users to an archive of the code. There are many ways you could offer source, 
and different solutions will be better for different programs;  see section 13 for the specific requirements.

# Contact
For any questions or comments, [email](https://www.icecreamlabs.com/contact-us)
