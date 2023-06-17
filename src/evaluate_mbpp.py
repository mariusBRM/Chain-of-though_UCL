from evaluate import load
import os
import sys
import pandas as pd
import ast
import csv
import torch
import numpy as np
import random
from tokenizer import *
from santaC import *
# needs to be running solely on Ubuntu
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

def eval(test_loader, model, num_to_gen, early_stoping = None):
    """
    This function evaluate the model on the test data
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    outputs = []
    final_outputs=[]

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
          if early_stoping is None or i < early_stoping:
            print(i,'========>', len(test_loader))
            ids = data['input_ids'].to(device, dtype = torch.long)
            # targets = data['labels'].to(device, dtype = torch.long)
            # tests = data['tests'].to(device, dtype = torch.long)
            for i in range(num_to_gen):
              # forward
              output = model.forward(ids).to(device)
              outputs.append(output)
            # postprocessing output
            decoded_output = [model.decode_output(t.cpu().numpy()) for t in outputs]
            code_generated = [model.post_generation_processing(dec) for dec in decoded_output]

            final_outputs.append(code_generated)
          else:
            if i > early_stoping :
              break

    return final_outputs

def save_generated_data(data, generated_code, path_to_save):
  selected_feature = ['code', 'test_list']
  data = data[selected_feature]
  # add the new feature
  data['generated_code'] = generated_code

  data.to_csv(path_to_save, index=False)
  return data

def evaluate_mbpp_data(df):

    reports = []
    # instantiate the test_cases/candidates list
    code_eval = load("code_eval")
    test_cases = []
    candidates = []

    # store the candidate/processed_tests
    for i in range(len(df)):

        candidate = [[df.iloc[i]['process_Gen_code']]]
        test = ast.literal_eval(df.iloc[i]['processed_tests'])
        format_test = []
        for k in range(len(test)):
            format_test.append([test[k]])
        test_cases.append(format_test)
        candidates.append(candidate)

    # evaluate the pass@k, result
    for i in range(len(candidates)):
        report = {
            'candidate' : candidates[i],
            'test' : test_cases[i],
            'Pass_one' : [],
            'result' : []
            }
        # iterate through each test case
        for j in range(len(test_cases[i])):
            pass_at_k, results = code_eval.compute(references=test_cases[i][j], predictions=candidates[i], k=[1])
            report['Pass_one'].append(pass_at_k['pass@1'])
            report['result'].append(results)

        reports.append(report)

    return reports

def write(reports):

    # Open (or create) a CSV file
    with open('reports.csv', 'w', newline='') as csvfile:
        # Define the fieldnames for the CSV. These should match the keys of the dictionaries
        fieldnames = ["candidate", "test", "Pass_one", "result"]

        # Create a CSV writer
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        # Write each dictionary in the list as a row in the CSV
        for report in reports:
            writer.writerow(report)

############################################ --> need to process the generated 

import pandas as pd
greedy_two_data = pd.read_csv('../data/200_greedy_solutions.csv')

# part of the processing
import ast

list_code = ast.literal_eval(greedy_two_data.iloc[0]['generated_code']) 

import re

def extract_first_function(input_string):
    # Regular expression pattern to match Python function
    pattern = re.compile(r"(def .*?:.*?return .*?\n)(?=\n|\Z)", re.DOTALL)

    # Find all matches of the pattern in the input string
    matches = pattern.findall(input_string)

    # If no complete function is found, return the original string
    if not matches:
        return input_string

    # Return the first complete function
    result = matches[0]
    
    return result

# still need to remove '<|endoftext|>' --> Is it a problem cannot remove it idk why


##############################""
def main():
    path_to_csv = 'data/mbpp_generated_processed_final.csv'
    df = pd.read_csv(path_to_csv)
    reports = evaluate_mbpp_data(df)

    write(reports)

    return None

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__=="__main__":
    # instanciate the testing loader :

    # setting the seed
    g = torch.Generator()
    g.manual_seed(0)


    # Create dataloader
    batch_size = 1
    num_workers = 0
    testing_data = pd.read_csv('data/mbpp_test.csv')
    PATH_TO_HUB = "bigcode/santacoder"

    # loading the testloader
    testloader = dataloading(testing_data, PATH_TO_HUB, batch_size, num_workers, g, seed_worker)

    model = MySantaCoder('samplingMethod')

    # generation properties
    # early_stoping = None 
    num_to_gen = 200

    fin_output = eval(testloader, model, num_to_gen)

    # 
    path_to_save = ""

    data = save_generated_data(testloader, fin_output, path_to_save)

    # evaluate on generated data
    reports = evaluate_mbpp_data(data)

    # save it as csv
    write(reports)

    


