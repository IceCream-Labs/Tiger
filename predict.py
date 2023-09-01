from cog import BasePredictor, Concatenated, Input
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import time
import json
from peft import PeftConfig, PeftModel

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

class TigerPredictor:
    def __init__(self, 
                 model_path: str, 
                 max_new_tokens: int,
                 max_length: int
        ):
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.model_path = model_path
        self.max_memory = {0: ""}
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            return_dict = True,
            device_map = "auto",
            max_memory = self.max_memory,
            trust_remote_code = True,
            torch_dtype = self.dtype,
            load_in_8bit = True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.stop_words = [
            "<human>:", 
            "<assitant>:", 
        ]
        self.stopCreteria = StoppingCriteriaList([
            StoppingCriteriaSub(
                stops=[self.tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in self.stop_words]
            )
        ])
        self.model = torch.compile(self.model)
        self.device = "cuda:0"
        self.temperature = 0.3
        self.top_p = 0.95
        self.top_k = 50
        self.repetition_penalty = 1.4
        self.num_return_sequences = 1


    def generate_prompt(self, context: str, question: str, answer: str = ""):
        """
        
        Method to format individual
        
        """
        PREPROMPT = "The following is a conversation between a customer and an AI product expert. The customer will provide a product description and the AI product expert analyses, comprehend and understands it. The customer will ask a question based on the given product description and the AI product always comes up with a detailed and helpful answer. The customer's question starts with <human>: and the AI assistant's answer starts with <assistant>:." 
        PROMPT = '''\n"{context}"\n<human>:{query}\n<assistant>:{answer}'''.format(context = context, query = question, answer = answer)    
        return f"{PREPROMPT}{PROMPT}".strip()
    

    def getGenerationConfig(self):
        """
        
        Method to get the genration config
        
        """
        generation_config = self.model.generation_config
        generation_config.max_new_tokens = self.max_new_tokens
        generation_config.max_length = self.max_length
        generation_config.temperature = self.temperature
        generation_config.top_p = self.top_p
        generation_config.top_k = self.top_k
        generation_config.repetition_penalty = self.repetition_penalty
        generation_config.num_return_sequences = self.num_return_sequences
        generation_config.pad_token_id = self.tokenizer.eos_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id
        return generation_config

    
    def customCall(self, prompt, postProcess = False):
        """
        Method to make custom prompt engineered calls
        
        """
        tokenized_prompt = self.tokenizer(prompt, return_tensors = "pt").to(self.device)
        start_time = time.time()

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids = tokenized_prompt.input_ids,
                attention_mask = tokenized_prompt.attention_mask,
                generation_config = self.getGenerationConfig(),
                stopping_criteria = self.stopCreteria,
                use_cache = False,
                do_sample = True,
            )
        end_time = time.time()
        time_ = int(end_time - start_time)
        # print(f"[INFO] Time Elasped: {time_} seconds")
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if postProcess:
            response = response.split("<assistant>:")[-1].split("<human>:")[0].strip()
        return response, time_
        

    def __call__(self, context:str, question: str):
        """
        
        Method to call the model for prediction
        
        """
        tokenized_prompt = self.tokenizer(self.generate_prompt(context = context, question = question), return_tensors = "pt").to(self.device)
        start_time = time.time()

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids = tokenized_prompt.input_ids,
                attention_mask = tokenized_prompt.attention_mask,
                generation_config = self.getGenerationConfig(),
                stopping_criteria = self.stopCreteria,
                use_cache = False,
                do_sample = True,
            )
        end_time = time.time()
        time_ = int(end_time - start_time)
        # print(f"[INFO] Time Elasped: {time_} seconds")
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("<assistant>:")[-1].split("<human>:")[0].strip()

        return response, time_

class ModelConfig:
    SAVED_MODEL_PATH = "icecreamlabs/Tiger-7B-Instruct"
    MAX_TOKENS = 512
    MAX_LEN = 2000


class Predictor(BasePredictor):
    def setup(self):
        """

        Load the model into memory to make running multiple predictions efficient

        """
        self.model = TigerPredictor(
            model_path = ModelConfig.SAVED_MODEL_PATH,
            max_new_tokens = ModelConfig.MAX_TOKENS,
            max_length = ModelConfig.MAX_LEN
        )

    def predict(self,
        context: str = Input(description=f"Context behind the conversation"),
        question: str = Input(description = f"Question you want to ask from the context")
    ) -> Concatenated[str]:
        """
        
        Run a single prediction on the model
        
        """
        output, _ = self.model.customCall(
            prompt = self.model.generate_prompt(
                context = context,
                question = question,
                answer = ""
            ),
            postProcess = True
        )

        return output