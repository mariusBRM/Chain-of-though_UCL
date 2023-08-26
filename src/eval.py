from evaluate import load
import os
import sys
import pandas as pd
import ast
import csv
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

##################################################################
#           This script can only be executed on Ubuntu OS        #
##################################################################

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


def evaluate_mtbp_data(df):
    
    reports = []
    # instantiate the test_cases/candidates list
    code_eval = load("code_eval")
    test_cases = []
    candidates = []
    
    # store the candidate/processed_tests
    for i in range(len(df)):

        candidate = [[df.iloc[i]['code_test']]]
        test = ast.literal_eval(df.iloc[i]['test_list'])
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

def write(reports, name_to_save):

    # Open (or create) a CSV file
    with open(name_to_save, 'w', newline='') as csvfile:
        # Define the fieldnames for the CSV. These should match the keys of the dictionaries
        fieldnames = ["candidate", "test", "Pass_one", "result"]

        # Create a CSV writer
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        # Write each dictionary in the list as a row in the CSV
        for report in reports:
            writer.writerow(report)

def main():
    # needs to add the folder and the name of the files
    path_to_file = ''
    name_of_csv = ''
    name_to_save = f'reports_{name_of_csv}'
    path_to_csv = path_to_file + name_of_csv

    # dataframe to test out
    df = pd.read_csv(path_to_csv)
    # write the reports
    reports = evaluate_mbpp_data(df)

    write(reports, name_to_save)

    return None

if __name__=="__main__":
    sys.exit(main())
