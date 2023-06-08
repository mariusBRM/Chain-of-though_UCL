import datasets
import os
import sys
import pandas as pd
import re

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

def preprocessing(df):
    """ 
        Preprocess the input to create a signature out of : function name and problem description
    """

    # generate a new column
    df["instruction"] = df['text']

    # generating the instruction format for eahc row
    for i in range(len(df)):
        # encapsulate the original instruction into format '# Write a function ...'
        description = '#' + df.iloc[i]['text']
        # keeping the name of the function with the parameters
        code = df.iloc[i]['code']
        function_name = extract_signature(code)
        # creating the instruction
        instruction = description + '\n' + function_name
        # storing preprocessed input 
        df['instruction'].iloc[i] = instruction

    return df

def load_data():
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
    path_dir = os.path.abspath(os.path.join(__file__,"../../.."))
    path = os.path.join(path_dir, "data")
    os.makedirs(path, exist_ok=True)

    # preprocess the data
    df_train = preprocessing(df_train)
    df_test = preprocessing(df_test)
    df_val = preprocessing(df_val)
    df_prompt = preprocessing(df_prompt)

    # save it
    df_train.to_csv(path + "/mbpp_train.csv", index=False)
    df_test.to_csv(path + "/mbpp_test.csv", index=False)
    df_val.to_csv(path + "/mbpp_val.csv", index=False)
    df_prompt.to_csv(path + "/mbpp_prompt.csv", index=False)

def main():
    load_data()

if __name__=="__main__":
    sys.exit(main())



