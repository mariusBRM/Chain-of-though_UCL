import torch
import re
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import numpy as np
import random
from torch.utils.data import DataLoader
from tokenizer import *
from generation_processing import *

#################################
#       Define SantaCoder       #
#################################

class MySantaCoder(nn.Module):
    def __init__(self, generation_method, list_of_bad_words = ['#', '"""'], max_tokens = 128, num_beam = 1, num_sol = 1):
        super(MySantaCoder, self).__init__()
        self.checkpoint = "bigcode/santacoder"
        # self.checkpoint = model_path_to_hub
        self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.max_new_tokens = max_tokens
        self.bad_words = self.get_input_ids_as_list(list_of_bad_words)

        if generation_method == 'GrdS':

            self.generation_config = GenerationConfig(
                bad_words_ids = self.bad_words,
                num_beams = num_beam,
                num_return_sequences = num_sol,
                max_new_tokens = self.max_new_tokens,
                eos_token_id=self.model.generation_config.eos_token_id,
                bos_token_id=self.model.generation_config.bos_token_id
                )
        elif generation_method == 'SmplM' : 
     
            self.generation_config = GenerationConfig( 
                bad_words_ids = self.bad_words,  
                do_sample = True,  
                num_beams = num_beam,
                num_return_sequences = num_sol,
                top_p = 0.8,
                temperature = 0.95,
                max_new_tokens = self.max_new_tokens,
                eos_token_id=self.model.generation_config.eos_token_id,
                bos_token_id=self.model.generation_config.bos_token_id
                )
    
    def get_input_ids_as_list(self, list_of_bad_words):
        """ Tokenize the list of bad words - meaning the word that should not be generated"""
        token_list = []
        for element in list_of_bad_words:
            token_list.append(self.tokenizer.encode(element))
        return token_list
    
    def forward(self, input_ids):
        # input_ids = input_ids.unsqueeze(0)
        outputs = self.model.generate(input_ids, self.generation_config)
        return outputs

    def decode_output(self, encoded_output):
        output = self.tokenizer.decode(encoded_output)
        return output

    def post_generation_processing(self, code):
        """ Post processing that keeps only the fist def/class block and remove extra lines skipped."""

        # split it into list of blocks
        list_blocks = re.split('def |class |assert |print ', code)

        if 'init' in list_blocks[1]:
            fill_word = '\nclass '
        else:
            fill_word = '\ndef '

        # keep only the first block
        result = list_blocks[0] + fill_word + list_blocks[1]

        # remove all trailing newlines
        while result.endswith('\n'):
            result = result[:-1]

        # remove all leading newlines
        while result.startswith('\n'):
            result = result[1:]

        return result

    def extract_function_block(self, text):
        """ Extraction of the function by counting the number of indentation"""
        lines = text.split('\n')
        result = []

        indent_level = None
        for line in lines:
            if line.strip() == '':  # Ignore empty lines
                continue

            current_indent = len(line) - len(line.lstrip())

            if indent_level is None:  # For the first 'def' line
                indent_level = current_indent
                result.append(line)
                continue

            if current_indent > indent_level:  # Inside the function's scope
                result.append(line)
            else:  # Outside the function's scope or another function
                break

        return '\n'.join(result)

    
#################################
#       Define Training         #
#################################

def loss_fn(output, target):
    return nn.BCELoss()(output, target)

def validation(validation_loader, model):
    """
    This function evaluate the model on the validation data
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    fin_targets=[]
    fin_outputs=[]
    running_loss = 0.0

    with torch.no_grad():
        for _, data in enumerate(validation_loader, 0):

            ids = data['input_ids'].to(device, dtype = torch.long)
            targets = data['labels'].to(device, dtype = torch.long)
            
            # forward
            output = model.forward(ids)
            # evaluate the loss
            loss = loss_fn(output, targets)
 
            # adding to list
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(output.cpu().detach().numpy().tolist())

            # add the loss to the running loss
            running_loss+=loss.item()

    return fin_outputs, fin_targets, running_loss/len(validation_loader)

def training_model(nb_epochs, train_dataloader, val_dataloader, patience):
    """
    This function trains the model on training data
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MySantaCoder()
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)
    best_val_loss = np.inf
    # keep track of the performances
    summary = []


    for epoch in range(nb_epochs):
            # dict containing the information
        report_epoch = {
                'epoch': epoch+1,
                'training_loss': 0.0,
                'valid_loss':0.0
            }
        model.train()
        running_loss = 0.0

        for i, data in enumerate(train_dataloader, 0):

            ids = data['input_ids'].to(device, dtype = torch.long)
            labels = data['labels'].to(device, dtype = torch.long)
            
             # initialize the optimizer
            optimizer.zero_grad()
            #forward inputs
            output = model.forward(ids) 
            # define the loss
            loss = loss_fn(output, labels)
            # backpropagate
            loss.backward()
            optimizer.step()
            # add the loss to the running loss
            running_loss+=loss.item()
            
            print('\rEpoch: {}\tbatch: {}\tLoss =  {:.3f}'.format(epoch+1, i, loss), end="")

        running_loss = running_loss / len(train_dataloader)
        report_epoch['training_loss'] = running_loss
        print("\n")
        # validation
        model.eval()
        with torch.no_grad():

            outputs, targets, val_loss = validation(validation_loader=val_dataloader, model= model)

            report_epoch['valid_loss'] = val_loss
            # set to evaluate potential metrics...

            print(f"Epoch {epoch+1}: train CE loss = {running_loss}", f"|| Valid: CE loss = {val_loss}")
            

        # early-stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            dict_model = model.state_dict()
            pat = 0
        else:
            pat += 1
            print("pat ", pat)
            if pat == patience:
                print("Early Stopping: Validation Loss did not decrease for", patience, "epochs.")
                break
        
        # save model each epoch
        torch.save(model.state_dict(), 'epoch'+str(epoch+1)+'.pt')

        print("\n")
        # add performances of the epoch to the overall summary
        summary.append(report_epoch)

    torch.save(dict_model, 'Fine_Tuned_SantaCoder.pt')
    
    return summary

##################################################
#            generation Step by Step             #
##################################################

def generating_step_by_step(model, data, stop_words, keep_context = True, early_stopping = None):
    """Generating code step by step 
    """
    codes = []
    for j in range(len(data)):
        if early_stopping is not None and j > early_stopping:
            break
        # start with the signature for the incoming problem
        code = data.iloc[j]['signature']
        # initiate the list of prompt to generate
        prompts = data.iloc[j]['prompts']
        # Iterate over each prompt
        for i, prompt in enumerate(prompts):
            # Add the prompt to the previously generated code
            input_text = code + '\n\t' + '#' + prompt

            # Encode the input text
            input_ids = model.tokenizer.encode(input_text, return_tensors='pt')

            # Generate the output
            output_ids = model.forward(input_ids)

            # Decode the output
            output_text = model.decode_output(output_ids[0])

            code = generation_cut_off(output_text, stop_words, keep_context, i)
            
            # keep only the last code generated after the output
            if keep_context==False:
                # remove context if set to False
                code = remove_context(code)

        # print("Final generated code:\n", code)
        codes.append(code)

    return codes


###############################################
#           COntext vs context less           #
###############################################

STOP_WORDS = ['def', 'if', 'for', 'while']

def context_and_contexless_generation(data, model, early_stopping = None):
    """ Generate two types of problems:
            1. generate with the appropriate function signature and the context (keep the structural generation cut off)
            2. generate with a random function name and without context (keep the structural generation cut off)
            3. Keep both generation at each step with a very large cut off function (when finding a new 'def') 
    """
    codes_with_context = []
    codes_without_context = []

    raw_generations_context = []
    raw_generations_no_context = []

    for j in range(len(data)):
        if early_stopping is not None and j > early_stopping:
            break

        code_with_context = []
        code_without_context = []

        no_cut_off_no_context = []
        no_cut_off_context = []

        # start with the signature for the incoming problem
        code = data.iloc[j]['signature']
        # start with a random name for the incoming problem
        code_random = data.iloc[j]['random_signatures']
        # initiate the list of prompt to generate
        prompts = data.iloc[j]['prompts']
        # Iterate over each prompt
        for i, prompt in enumerate(prompts):
            
            # Add the prompt to the previously generated code
            input_text_context = code + '\n\t' + '#' + prompt
            input_text_no_context = code_random +'\n\t' + '#' + prompt

            # Encode the input text
            input_ids_context = model.tokenizer.encode(input_text_context, return_tensors='pt')
            input_ids_no_context = model.tokenizer.encode(input_text_no_context, return_tensors='pt')

            # Generate the output
            output_ids_context = model.forward(input_ids_context)
            output_ids_no_context = model.forward(input_ids_no_context)

            # Decode the output
            output_text_context = model.decode_output(output_ids_context[0])
            output_text_no_context = model.decode_output(output_ids_no_context[0])

            # Cut off the generated code
            code = generation_cut_off(gen_code = output_text_context, stop_words=STOP_WORDS, index_prompt=i)
            code_random = generation_cut_off(gen_code = output_text_no_context, stop_words=STOP_WORDS, index_prompt=0)
            code_random = remove_context(code_random)

            # Keep the generation with a large cut off (new def found)
            output_text_context = model.post_generation_processing(output_text_context)
            output_text_no_context = model.post_generation_processing(output_text_no_context)

            code_with_context.append(code)
            code_without_context.append(code_random)

            no_cut_off_no_context.append(output_text_no_context)
            no_cut_off_context.append(output_text_context)

        codes_with_context.append(code_with_context)
        codes_without_context.append(code_without_context)

        raw_generations_context.append(no_cut_off_context)
        raw_generations_no_context.append(no_cut_off_no_context)

    return codes_with_context, raw_generations_context, codes_without_context, raw_generations_no_context


###############################################
#             Prompt vs Context               #
###############################################

def generation_prompts_functions(data, model, early_stopping = None):
    """ 
        Generation of the prompts and the context functions.
    """

    # Codes for every problem
    codes_with_context = []
    codes_by_prompts = []

    for j in range(len(data)):
        if early_stopping is not None and j > early_stopping:
            break
        
        # Code for every prompt
        code_with_context = []
        code_by_prompt = []

        # start with the signature for the incoming problem
        code = data.iloc[j]['signature']
        # initiate the list of prompt to generate
        prompts = data.iloc[j]['prompts']
        
        for i, prompt in enumerate(prompts):
            
            # input text
            instruction = '#' + prompt
            function = code + '\n\t#' + prompt 

            # tokenization
            ids_instruction = model.tokenizer.encode(instruction, return_tensors='pt')
            ids_function = model.tokenizer.encode(function, return_tensors='pt')

            # generation
            output_instrution = model.forward(ids_instruction)
            output_function = model.forward(ids_function)

            # decoding  
            text_instruction = model.decode_output(output_instrution[0])
            text_function = model.decode_output(output_function[0])

            # post processing
            text_instruction = cut_off_generated_text(text_instruction)
            text_function = model.post_generation_processing(text_function)

            # add to the list
            code_by_prompt.append(text_instruction)
            code_with_context.append(text_function)

        codes_with_context.append(code_with_context)
        codes_by_prompts.append(code_by_prompt)
    
    return codes_by_prompts, codes_with_context


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__=="__name__":
    
    # load the data
    train_data, val_data = ''
    PATH_TO_HUB = 'bigcode/santacoder'

    g = torch.Generator()
    g.manual_seed(0)

    # Tokenizing datasets
    trainset = MyTokenizer(train_data, PATH_TO_HUB)
    valset = MyTokenizer(val_data, PATH_TO_HUB)

    # Create dataloader
    batch_size = 1
    num_workers = 0

    trainloader = dataloading(train_data, PATH_TO_HUB, batch_size, num_workers, g, seed_worker)
    validloader = dataloading(val_data, PATH_TO_HUB, batch_size, num_workers, g, seed_worker)

    # set training parameters ( look out how to enhance this)
    nm_epochs = 5
    patience = 2
    summary = training_model(nm_epochs, trainloader, validloader, patience)