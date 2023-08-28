import ast
import re
from collections import Counter
import torch

####################################################################################
#           Processing generated code : Context vs Without Context                 #
####################################################################################

def keep_only_generated_code_prompts(codes):
    """ Takes as input a list of codes and return the generated codes as a list of generated code.
    """
    generated_code = []

    for i in range(len(codes)):
        lines = codes[i].split('\n')
        code = lines[1:]
        generated_code.append('\n'.join(code))

    return generated_code

def keep_only_generated_code_context(codes):
    """ Takes as input a list of codes and return the generated codes as a list of generated code.
    """
    generated_codes = []

    for i in range(len(codes)):
        text = process_function_with_context(codes[i], i)
        generated_codes.append(text)
        
    return generated_codes

def process_function_with_context(text, prompt_index):
    # Split the text into lines
    lines = text.split('\n')
    
    # Initialize counter for the prompt index
    prompt_counter = -1
    # Initialize variable to store the result
    result = []
    
    # Iterate over the lines
    for line in lines:
        # If the line starts with '#', increment the prompt counter
        if line.strip().startswith('#'):
            prompt_counter += 1
        
        # If the current prompt counter equals the given prompt index, add the line to the result
        if prompt_counter == prompt_index and not line.strip().startswith('#'):
            result.append(line.strip())
    
    # Join the result lines with '\n' and return it
    return '\n'.join(result)


def process_generated_codes(data, length_penalty=False):
    """  Apply the post processing to keep only the generated code for both context and without context
    """
    generated_prompts = []
    generated_contexts = []

    for j in range(len(data)):

        generated_prompt = []
        generated_context = []

        if length_penalty:
            codes_context = ast.literal_eval(data.iloc[j]['lenght_penalty_generation'])

            for i in range(len(codes_context)):
                generated_context = keep_only_generated_code_context(codes_context)
            
        else:
            codes_prompt = ast.literal_eval(data.iloc[j]['codes_by_prompts']) 
            codes_context = ast.literal_eval(data.iloc[j]['codes_with_context'])

            for i in range(len(codes_prompt)):
                generated_prompt = keep_only_generated_code_prompts(codes_prompt)
                generated_context = keep_only_generated_code_context(codes_context)
        
        generated_prompts.append(generated_prompt)
        generated_contexts.append(generated_context)

    return generated_prompts, generated_contexts


#####################################################################################
#                     Pick best selection with log-likelihood                       #
#                              Length Algorithm                                     #
#####################################################################################

def compute_log_likelihood(model, output_ids):
    with torch.no_grad():
        logits = model.model(output_ids).logits.to('cuda')

    # Calculate probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1).to('cuda')

    # Gather the probabilities for the generated tokens
    token_probs = probs.gather(-1, output_ids.unsqueeze(-1)).squeeze(-1)

    # Compute log likelihood for each sequence
    log_likelihoods = torch.sum(torch.log(token_probs), dim=-1).to('cuda')

    return log_likelihoods.squeeze().tolist()


def compute_length(tensor, stop_token=49152):
    """ Compute the length : number token"""
    # Find the index of the first occurrence of the stop_token
    indices = (tensor == stop_token).nonzero(as_tuple=True)[0]

    # If the stop_token doesn't exist in the tensor, return the length of the tensor
    if len(indices) == 0:
        return len(tensor)

    # Otherwise, return the index of the first occurrence
    return indices[0].item()


def compute_length_words(text, prompt_index):
    """ Compute the length : number of words"""
    # we need to take into account solely what has been generated
    generated_text = process_function_with_context(text, prompt_index)
    return len(generated_text.split(' '))

def compute_length_lines(text, prompt_index):
    """ Compute the length : number of lines"""
    # we need to take into account solely what has been generated
    generated_text = process_function_with_context(text, prompt_index)
    return len(generated_text.split('\n'))

def apply_length_penalty(log_likelihood, length, alpha=0.7):
    """ Apply the length penalty"""
    return log_likelihood / (length**alpha)

def pick_best_length(model, output_ids, alpha, type_of_length = 'token', processed_text = None, prompt_index=None):
    """ Select the best scored sentences"""
    # Calculate the log_likelihood for each output_ids
    log_likelihoods = compute_log_likelihood(model, output_ids)

    if type_of_length == 'token':
        # apply the length penalty of token to every sequences generated
        penalized_scores = [apply_length_penalty(ll, compute_length(output), alpha) for ll, output in zip(log_likelihoods, output_ids)]
    elif type_of_length == 'words':
        # apply the length penalty of words to every sequence generated
        penalized_scores = [apply_length_penalty(ll, compute_length_words(output, prompt_index), alpha) for ll, output in zip(log_likelihoods, processed_text)]
    else :
        # apply the length penalty of lines to every sequence generated
        penalized_scores = [apply_length_penalty(ll, compute_length_lines(output, prompt_index), alpha) for ll, output in zip(log_likelihoods, processed_text)]

    # pick the best one
    best_index = penalized_scores.index(max(penalized_scores))
    best_output = output_ids[best_index]

    return best_output   

def list_to_tensor(lst):
    """ Convert a list into a tensor."""
    # Find the length of the longest sublist
    max_len = max(len(sublist) for sublist in lst)
    
    # Initialize a tensor of size (len(lst), max_len) filled with the stop token
    tensor = torch.full((len(lst), max_len), 49152)
    
    # Fill the tensor with values from the input list
    for i, sublist in enumerate(lst):
        for j, value in enumerate(sublist):
            tensor[i][j] = value
    
    return tensor

def generation_post_processing(model, output_ids):
    """ Keep only generated code with the indentation """
    output_text = []

    for i in range(len(output_ids)):

        # decode every output
        output_function = model.decode_output(output_ids[i])

        # cut off generation
        processed_output = model.extract_function_block(output_function)

        # encode again
        encoded_output = model.tokenizer.encode(processed_output)

        # add to list
        output_text.append(encoded_output)
    
    # add the end_of_text token to get the same length in all tensors
    output_ids_processed = list_to_tensor(output_text)

    return output_ids_processed                


def remove_endoftext(text):
    """ Remove all EOS token from the generated."""
    # remove all unecessary '\n' 
    new_text = text.strip()
    # remove all endoftext tokens
    return new_text.replace('<|endoftext|>', '')

def format_testing_prompt_vs_context(data):
    """ Format for testing."""
    code_to_test = []

    for k in range(len(data)):
        # will take the last one 
        text = ast.literal_eval(data.iloc[k]['codes_with_context'])[-1]
        processed_text = remove_endoftext(text)
        code_to_test.append(processed_text)
    
    data['code_test'] = code_to_test

    return data

def format_testing_lenght_penalty(data):
    """ Format testing for the Ubuntu."""
    code_to_test = []

    for k in range(len(data)):
        # will take the last one 
        text = ast.literal_eval(data.iloc[k]['lenght_penalty_generation'])[-1]
        processed_text = remove_endoftext(text)
        code_to_test.append(processed_text)
    
    data['code_test'] = code_to_test

    return data