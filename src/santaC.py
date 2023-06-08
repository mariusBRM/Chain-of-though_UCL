import torch
import re
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


class MySantaCoder(nn.Module):
    def __init__(self, model_path_to_hub):
        super(MySantaCoder, self).__init__()
        # self.checkpoint = "bigcode/santacoder"
        self.checkpoint = model_path_to_hub
        self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.max_new_tokens=512
        self.stop_words=["\nclass", "\ndef","\nassert", '\n"""', "\nprint", "\n<|"]

        self.generation_config = GenerationConfig(
            do_sample=True, 
            top_k=5,
            max_length = self.max_new_tokens,
            eos_token_id=self.model.generation_config.eos_token_id,
            bos_token_id=self.model.generation_config.bos_token_id
            )
    
    def forward(self, input):
        input_ids = input['input_ids'].unsqueeze(0)
        outputs = self.model.generate(input_ids, self.generation_config)
        return outputs
    
    def decode_output(self, encoded_output):
        output = self.tokenizer.decode(encoded_output[0])
        return output
    
    def post_generation_processing(self,encoded_output):
        code = self.tokenizer.decode(encoded_output[0])
        # split it into list of blocks
        list_blocks = re.split('def |class |assert |print ', code)
        if 'init' in list_blocks[1]:
            fill_word = '\nclass '
        else:
            fill_word = '\ndef '
        # keep only the first block
        result = list_blocks[0] + fill_word + list_blocks[1]
        return result
    

def train(model, dataloader, nm_epochs):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    fin_targets=[]
    fin_outputs=[]
    running_loss = 0.0

    with torch.no_grad():
        for _,data in enumerate(dataloader, 0):

            input = data.to(device, dtype = torch.long)
            labels = input['labels']
            # output
            output = model.forward(input)
            # loss
            



    return None