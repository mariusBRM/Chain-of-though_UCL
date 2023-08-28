import datasets
import os
import sys
import pandas as pd
import ast
import re
import numpy as np
import json
import random 
import string
from post_processing import *
from output_analysis_utils import *


####################
# Download mbpp data
####################

def extract_signature(code):
    # Match function or class signature
    pattern = r"(def|class)\s+(\w+)\s*\((.*?)\)"
    match = re.search(pattern, code, re.DOTALL)
    if match:
        keyword = match.group(1)  # def or class
        name = match.group(2)  # function or class name
        if keyword == 'def':
            parameters = match.group(3).strip()  # function parameters
            return f"{keyword} {name}({parameters}):"
        else:
            return f"{keyword} {name}():"
    return None

def preprocessing_instruction(df):
    """ 
        Preprocess the input to create a signature out of : function name and problem description
    """

    # generate a new column
    df["instruction"] = df['text']

    # generating the instruction format for eahc row
    for i in range(len(df)):
        # encapsulate the original instruction into format '# Write a function ...'
        description = '\t"""' + df.iloc[i]['text'] + '"""'
        # keeping the name of the function with the parameters
        code = df.iloc[i]['code']
        function_name = extract_signature(code)
        # creating the instruction
        instruction = function_name + '\n' + description
        # storing preprocessed input 
        df['instruction'].iloc[i] = instruction

    return df

def extract_assertions(input_string):
    """ Transform the assertions to be as a list """
    # Using ast.literal_eval to safely parse the string representation of list into an actual list
    extracted_list = ast.literal_eval(input_string)
    
    # Now, ast.literal_eval gives us list of single string, we need to split this string 
    # into individual assertions. We will use newline '\n ' as the separator
    assertions = extracted_list[0].split('\n ')
    
    # Strip leading and trailing white spaces from each assertion
    # Also strip leading and trailing escape characters and quotation marks
    assertions = [assertion.strip().strip('\'') for assertion in assertions]
    
    # string of assertions
    str_assertions = assertions[0].split('assert')
    processed_assertions = []
    
    for i in range(len(str_assertions) - 1):
        processed_assertions.append('assert' + str_assertions[i+1])

    return processed_assertions

def process_assertions(df):
    list_assertions = []
    for i in range(len(df)):
        assertions = np.array(df.iloc[i]['test_list']).tolist()  # Convert NumPy array to Python list
        list_assertions.append(extract_assertions(assertions))

    df['tests'] = list_assertions
    return df


def processing_mbpp(df, is_test):
    """ Process both instruction and test_list"""
    df = preprocessing_instruction(df)
    if is_test == True:
        df = process_assertions(df)
    return df


def generate_X_shot_prompt(df, nm_shot):
    """ Build a few shot prompt."""
    # build the list of prompts
    prompts = ''

    for i in range(nm_shot):
        prompts += '# ' + df.iloc[i]['text'] + '\n' + df.iloc[i]['code'] + '\n\n'

    return prompts

def generate_X_shot(path_to_prompt, path_to_test, path_to_save, nm_shot):
    """ Generate few shot prompt for a dataset. """
    # instantiate the dataframes
    df = pd.read_csv(path_to_prompt)
    data = pd.read_csv(path_to_test)

    # generate the X-shot instructions
    prompt = generate_X_shot_prompt(df, nm_shot)

    # Add it to the data
    for i in range(len(data)):
        instruction = data.iloc[i]['instruction']
        data.at[i,'instruction'] = prompt + instruction
    
    data.to_csv(f"{path_to_save}/mbpp_test_FS_{nm_shot}", index=False )

    return None

def load_mbpp_data():
    """
        Load the mbpp data from huggingface library
    """
    # download raw dataset
    dataset = datasets.load_dataset('mbpp')
    df_train = dataset['train'].to_pandas()
    df_test = dataset['test'].to_pandas()
    df_val = dataset['validation'].to_pandas()
    df_prompt = dataset['prompt'].to_pandas()

    # set up folder and path to save it
    path_dir = os.path.abspath(os.path.join(__file__,"../"))
    path = os.path.join(path_dir, "data")
    os.makedirs(path, exist_ok=True)

    # preprocess the data
    df_train = processing_mbpp(df_train, False)
    df_test = processing_mbpp(df_test, False)
    df_val = processing_mbpp(df_val, False)
    df_prompt = processing_mbpp(df_prompt, False)

    # save it
    df_train.to_csv(path + "/mbpp_train.csv", index=False)
    df_test.to_csv(path + "/mbpp_test.csv", index=False)
    df_val.to_csv(path + "/mbpp_val.csv", index=False)
    df_prompt.to_csv(path + "/mbpp_prompt.csv", index=False)

    return None


#################################################
#                 MTBP dataset                  #
#################################################

def read_json_line_format(path_to_file):
    """
        Read a JSON Lines format and store it into a dataframe.
    """
    data = []
    with open(path_to_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.json_normalize(data)
    return df
########################### Unused yet ###############################

def get_keys(input_list):
    """Get the list of unique input keys and list it (comma separated).
    """
    keys = set()
    for d in input_list:
        keys.update(d.keys())
    keys = sorted(list(keys))  # sort keys for consistent output
    return ','.join(keys)

def processing_name(name):
    """Processing the name of the problem to match the syntax of a function
    """
    name = name.lower()  # convert to lowercase
    name = re.sub('[^a-z0-9 ]', '', name)  # remove any non-alphanumeric characters (except spaces)
    name = re.sub(' ', '_', name)  # replace spaces with underscores
    return name

def create_signature_for_function(data):
    """Create the function signature for each problem.
    """
    # initiate a list of signature
    signatures = []
    # loop over all the rows
    for i in range(len(data)):
        # extract the name of the according problem
        name = data.iloc[i]['name']
        # process the name
        name = processing_name(name)
        # get the input
        inputs = data.iloc[i]['inputs']
        # extract the name
        input_keys = get_keys(inputs)
        # create the function signature architecture
        signature = f'def {name}({input_keys}):'
        # adding the signature to the list
        signatures.append(signature)
    
    data['signature'] = signatures
    return data

#######################################################################


#############################################################
#               Prompts vs Context                          #
#############################################################

def generate_random_name(signature):
    """ Generate random function names """
    # Find the function name using regular expression
    match = re.match(r'def ([\w\-_%.]+)\((.*)\):', signature)
    if match:
        original_name, parameters = match.groups()
        
        # Generate a random name with a reasonable length (e.g., length of original name)
        random_name = ''.join(random.choice(string.ascii_lowercase) for _ in range(len(original_name)))

        # Replace the original name with the random name
        new_signature = f'def {random_name}({parameters}):'
        return new_signature
    else:
        print(signature)
        # raise ValueError("Invalid function signature")
        return signature

def custom_dataset_context_investigation(mtbp_converted, mtbp):
    """ Create a custom dataset out of the converted_mtbp and the mtbp with a random function name."""

    # select only features that are interesting
    features_name_converted = ['text', 'signature','test_list']
    mtbp_converted = mtbp_converted[features_name_converted]
    features_name = ['prompts']
    mtbp = mtbp[features_name]
    
    data = pd.concat([mtbp, mtbp_converted], axis=1)

    random_names = []

    for i in range(len(data)):
        signature = data.iloc[i]['signature']
        random_name = generate_random_name(signature)
        random_names.append(random_name)

    data['random_signatures'] = random_names

    return data

#########################################################################
#                       Custom Alpha Dataset                            #
#########################################################################

def normalize(list_of_lists, a=0.5, b=1.5):
    # normalizing between a and b
    flat_list = [item for sublist in list_of_lists for item in sublist]
    min_val = min(flat_list)
    max_val = max(flat_list)
    
    normalized = []
    for sublist in list_of_lists:
        norm_sublist = [round(a + (x - min_val) * (b - a) / (max_val - min_val), 3) for x in sublist]
        normalized.append(norm_sublist)
    
    return normalized

def alphas_columns(lengths_prompt, lengths_context):
    # create normalized diff
    diff_length = []

    for i in range(len(lengths_prompt)):
        lenght_step = []
        for j in range(len(lengths_prompt[i])):
            
            nominateur = lengths_prompt[i][j]
            denominateur = lengths_context[i][j]
            
            if nominateur == 0 and denominateur!= 0:
                lenght_step.append(1)
            elif denominateur == 0 and nominateur!=0:
                lenght_step.append(1)
            elif denominateur == 0 and nominateur == 0:
                lenght_step.append(1)
            else:
                lenght_step.append(nominateur/denominateur)

        diff_length.append(lenght_step)

    normalized_alphas = normalize(diff_length, 0.5, 1.5)

    return normalized_alphas


def custom_dataset(data):
    # load 
    gen_p, gen_c = process_generated_codes(data)
    lengths_prompt, lengths_context = calculate_lengths(gen_p, gen_c)

    # create alphas
    alphas = alphas_columns(lengths_prompt, lengths_context)

    # add to dataset
    data['alphas'] = alphas

    return data


def main():
    # can be used to load the required dataset.
    return None

if __name__=="__main__":
    sys.exit(main())
