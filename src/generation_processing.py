import re
import numpy as np

#####################################################
#               Remove Context                      #
#####################################################
def remove_context(code):
    """remove all lines starting with '\t#' 
    """
    lines = code.split('\n')
    # keep only lines that don't start with '\t#'
    lines = [line for line in lines if not line.startswith('\t#')]
    return '\n'.join(lines)

#####################################################
#           Step by Step : Line by line             #
#####################################################

def keep_code_until_after_first_comment(code):
    # Split the code into lines
    code_lines = code.split('\n')
    # Initialize a list to store the output lines
    output_lines = []
    # Initialize a variable to keep track if a comment has been found
    comment_found = False
    # Loop over the lines
    for line in code_lines:
        # Add the line to the output
        output_lines.append(line)
        # If the line starts with a tab followed by a hash and a comment hasn't been found before,
        # mark that a comment has been found
        if line.startswith('\t#') and not comment_found:
            comment_found = True
        # If a comment has been found and the current line does not start with a comment, stop adding lines to the output
        elif comment_found and not line.startswith('\t#'):
            # need to make a function that keep the structure
            break
    # Join the output lines back together with newlines and return the result
    return '\n'.join(output_lines)

def get_code_for_prompt_old(code_text, prompt_index):
    lines = code_text.split('\n')
    prompt_count = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('#'):  # we have a new prompt
            if prompt_count == prompt_index:  # we have reached the desired prompt
                break
            prompt_count += 1
    return '\n'.join(lines[:i+2])  # include the next line after the prompt


#####################################################
#           Step by Step : Structural               #
#####################################################

# we want to build a function that : 
#     - stop generates after the instruction is full filled after the last prompt :
#         * only line after is kept ( current naive approach )
#         * the rest is kept if the first line contains : 
#             - an 'if' statement is found : keep all the if structure until new indentation
#                 ex :  \n\tif (...):\n\t\t#code\n\t#after {all the 'after' needs to be deleted}
#             - a 'def' is found : keep all the def function until the 'return' associated with this function
#             - a 'for' loop is found : keep all the structure of the 'for' loop ( the same way as the 'if' statement works )
#             - a 'while' loop is found : keep all the structure of the 'while' loop ( the same way as the 'if and the 'for')
#     - when find a module 'import' place it before the first function signature it finds ( might need to be tricky )
#         - either doing afterward meaning once everything has been generated
#         - doing it before the generation starts

# list of words that create a special treatment but can be adjusted in the future
STOP_WORDS = ['def', 'if', 'for', 'while']

def check_if_start_with(stop_words, line):
    # check if a string begins with some stop words
    for word in stop_words:
        if line.strip().startswith(word) and count_indentation(line) == 1:
            return True   
    return False

def count_indentation(line):
    # count the number of indentation in a line
    count = 0
    for char in line:
        if char == '\t':
            count += 1
        else:
            break
    return count

def identify_what_step_instruction(lines, index_prompt, keep_context):
    """
        function that spots the line at wich the current step is being considered ( the index of the last prompt )

        Input:
            lines
        Output:
            index of the line of the actual step ( index of the last '#')
    """
    # initialise the promp cursor
    prompt_count = 0
    # initialise the index at which the generation has started
    index_to_start = 0
    # iterate through the lines
    for i,line in enumerate(lines):
        # if the context as been kept (we count the number of comments generated)
        if keep_context:
            # we need to know according to what prompt we are trying to generate code
            if line.strip().startswith('#') and count_indentation(line) == 1:  # we have a new prompt
                if prompt_count == index_prompt:  # we have reached the desired prompt
                    index_to_start = i
                prompt_count += 1
        else:
            # the first prompt encountered is the actual step
            if line.strip().startswith('#'):
                index_to_start = i
                   
    return index_to_start

def identify_chunks_of_code(list_lines, stop_words):
    """
        Truncate the chunk of code that is to be generated with the current step

        Input:
            list_lines: the list of lines after the current step ( the line right after the last '#' encountered)
    """
    # initialise the level of indentation of the current chunks of code and the number of line within it
    indentation_reference = 0
    nb_line_of_code = 1
    # initialise the chunk of codes that needs to be saved
    chunk_of_code = []
    try:
        # first line is the line right after the last prompt
        first_line = list_lines[0]
        
        if check_if_start_with(stop_words, first_line):
            indentation_reference = count_indentation(first_line) + 1 # number of '\t' found in the line ( needs to count it )
            # keep track of the structure and check the indentation level
            for _, step_in in enumerate(list_lines[1:]): 
                # check the indentation level and as soon as line looses their indentation cut off
                indentation_level = count_indentation(step_in)
                if indentation_level >= indentation_reference:
                    nb_line_of_code+=1
                else:
                    break
            # append all of the lines
            for i in range(nb_line_of_code):
                chunk_of_code.append(list_lines[i])         
        else:
            # if not a structural type of things then only append the last line
            chunk_of_code.append(list_lines[0])
    except NameError:
        print(f'{NameError} for this one...')

    return chunk_of_code

def generation_cut_off(gen_code, stop_words, keep_context = False, index_prompt = None):
    """
        A function that cut off the code which let only the instructed code that was generated

        Input:
            gen_code: a text of generated code
            stop_words: a list of stop words
        Output:
            processed_gen_code: the processed code
    """
    
    # First we have to split the code as a list of lines
    lines = gen_code.split('\n')

    # we identify the last prompts we are interested in. 
    index_last_prompt = identify_what_step_instruction(lines, index_prompt, keep_context)

    # we then keep only the code generated to come
    # the '+1' here refers to the first line of code generated after the comment'
    begining = index_last_prompt + 1
    steps = lines[begining:]
    
    # we extract the chunks of codes that is generated out for the last prompt
    chunks_of_code = identify_chunks_of_code(steps, stop_words)

    # we need to keep the code previously generated and the last piece of code that has been generated
    codes = lines[:index_last_prompt+1] + chunks_of_code
            
    return '\n'.join(codes)

################ Testing ###################
# test1 = 'def count_zeros(ars):\n\t# Initialize counter to zero\n\tcounter = 0\n\t# Loop over each element in the list\n\tfor i in ars:\n\t\t# Increment counter if the element is zero\n\t\tif i == 0:\n\t\t\tcounter += 1\n\t# Return the count of zeros\n\treturn counter'
# test2 = 'def first_element(ars):\n\t# Assign the first element of the list to a variable\n\tfirst = ars[0]\n\t# Return the first element\n\treturn first'
# test3 = 'def sum_while(ars):\n\t# Initialize variables\n\ttotal = 0\n\ti = 0\n\t# While loop until reaching the end of the list\n\twhile i < len(ars):\n\t\t# Add current value to total\n\t\ttotal += ars[i]\n\t\t# Increment the index\n\t\ti += 1\n\t# Return the total sum\n\treturn total'
# test4 = 'def filter_negatives(ars):\n\t# Initialize an empty list for positive numbers\n\tpositives = []\n\t# Loop over each element in the list\n\tfor num in ars:\n\t\t# Add number to the list if it\'s non-negative\n\t\tif num >= 0:\n\t\t\tpositives.append(num)\n\t# Return the list of non-negative numbers\n\treturn positives'
# test5 = 'def flatten_list(ars):\n\t# Initialize an empty list for the result\n\tflattened = []\n\t# Loop over each element in the list\n\tfor sublist in ars:\n\t\t# Loop over each element in the sublist\n\t\tfor item in sublist:\n\t\t\t# Add item to the flattened list\n\t\t\tflattened.append(item)\n\t# Return the flattened list\n\treturn flattened'
# lines1 = test1.split('\n')
# lines2 = test2.split('\n')
# lines3 = test3.split('\n')
# lines4 = test4.split('\n')
# lines5 = test5.split('\n')

def extract_function(text):
    """
        Function that extracts the function out of a text.
    """
    function_start = text.find("def")
    function_end = text.find("\n\n", function_start)
    if function_end == -1:
        function_end = len(text)
    return text[function_start:function_end]

def cut_off_function_baselines(df):
    processed_func = []
    for i in range(len(df)):
        processed_func.append(extract_function(df.iloc[i]['generated_code']))
    df['gen_code'] = processed_func
    
    return df


def replace_print_with_return(func_str):
    # The regex pattern matches 'print' at the start of a line (considering leading whitespaces)
    pattern = r'^(\s*)print\((.*)\)'
    
    # Use a regex sub() function with a lambda function to replace 'print' with 'return'
    result = re.sub(pattern, lambda match: f"{match.group(1)}return({match.group(2)})", func_str, flags=re.MULTILINE)
    
    return result

# list of all the library that must be imported
Imported_libraries = "import pandas\nfrom sklearn import LinearRegression\nimport math\nimport numpy\nimport re\nimport datetime\nimport sklearn\nfrom sklearn.model_selection import train_test_split\nimport collections\nfrom collections import OrderedDict\n\n"

def format_for_testing(data):
    """
        Replace all the print(..) instances with return(..) in order to get the right format for the testing.
    """
    processed_gen_code = []

    for i in range(len(data)):
        # replace for all the problem
        new_processing = Imported_libraries + replace_print_with_return(data.iloc[i]['gen_code'])
        processed_gen_code.append(new_processing)
    
    data["code_test"] = processed_gen_code

    return data

##############################################
#            Prompt vs Context               #
##############################################

def find_all_indices(text, substring):
    indices = []
    index = text.find(substring)
    while index != -1:
        indices.append(index)
        index = text.find(substring, index + 1)
    return indices

def starts_with_def(text):
    """ Check if the generated text start with a def or not."""
    lines = text.split('\n')
    found_comment = False

    for line in lines:
        stripped_line = line.strip()

        if not found_comment and stripped_line.startswith('#'):
            found_comment = True
            continue

        if found_comment and stripped_line:
            return stripped_line.startswith('def')

    return False

def cut_off_generated_text(text):
    # Finding the index of the pattern '\n\nprint('
    index_print = text.find('\n\nprint(')
    # Finding the index of the pattern '\n\ndef'
    index_def = text.find('\n\ndef')
    startwith_def = starts_with_def(text)
    indexes_def = find_all_indices(text, '\n\ndef')
    # Finding the minimum index among the two patterns (if found)
    index = -1
    if index_print != -1 and index_def != -1:
        # there are both 'def's and 'print()'s within the text
        if startwith_def:
            if len(indexes_def) > 1:
                index_def = indexes_def[1]
            else:
                index_def = np.inf
        index = min(index_print, index_def)
    elif index_print != -1:
        index = index_print
    elif index_def != -1:
        index = index_def

    # If any pattern is found, slicing the text accordingly
    if index != -1:
        return text[:index]
    return text
