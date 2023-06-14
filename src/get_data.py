import datasets
import os
import sys
import pandas as pd
import ast
import re
import numpy as np

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

def processing(df, is_test):
    """ Process both instruction and test_list"""
    df = preprocessing_instruction(df)
    if is_test == True:
        df = process_assertions(df)
    return df

def load_mbpp_data():
    """
        Load the data from hugginface library
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
    df_train = processing(df_train, False)
    df_test = processing(df_test, False)
    df_val = processing(df_val, False)
    df_prompt = processing(df_prompt, False)

    # save it
    df_train.to_csv(path + "/mbpp_train.csv", index=False)
    df_test.to_csv(path + "/mbpp_test.csv", index=False)
    df_val.to_csv(path + "/mbpp_val.csv", index=False)
    df_prompt.to_csv(path + "/mbpp_prompt.csv", index=False)

    return None


# to modify
def generate_X_shot_prompt(df, nm_shot):
    # build the list of prompts
    prompts = ''

    for i in range(nm_shot):
        prompts += '# ' + df.iloc[i]['text'] + '\n' + df.iloc[i]['code'] + '\n\n'

    return prompts

def generate_X_shot(path_to_prompt, path_to_test, path_to_save, nm_shot):
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
    


def main():
    load_mbpp_data()
    # generate Few-Shot
    return None

if __name__=="__main__":
    sys.exit(main())
