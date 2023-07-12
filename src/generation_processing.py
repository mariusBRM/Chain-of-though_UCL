

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


