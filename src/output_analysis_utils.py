import re
from collections import Counter
from post_processing import * 
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from scipy.stats import mannwhitneyu

#######################################################################################
#                       Utils function to Analyse the output                          #
#######################################################################################

def calculate_lengths(codes_prompt, codes_context):
    lengths_prompt = [[len(code) for code in inner_list] for inner_list in codes_prompt]
    lengths_context = [[len(code) for code in inner_list] for inner_list in codes_context]
    
    return lengths_prompt, lengths_context


def get_most_common_words(codes_list1, codes_list2, topn=10):
    # Flatten the list of lists and join all the codes into a single string
    all_codes_str = ' '.join([' '.join(sublist) for sublist in codes_list1+codes_list2])

    # Use regular expressions to find all occurrences of each word
    all_words = re.findall(r'\b\w+\b', all_codes_str)

    # Use a Counter to count the occurrences of each word
    word_counter = Counter(all_words)

    # Return the top n most common words and their counts
    return word_counter.most_common(topn)

def count_number_of_eos(list_p, list_c):

    num_eos_prompt = 0
    num_eos_context = 0

    idx_eos_prompt = []
    idx_eos_context = []

    for idx in range(len(list_p)):
        for idx_prompt in range(len(list_p[idx])):
            if list_p[idx][idx_prompt].count('<|') != 0:
                num_eos_prompt+=1
                idx_eos_prompt.append([idx, idx_prompt])
            if list_c[idx][idx_prompt].count('<|') != 0:
                num_eos_context+=1
                idx_eos_context.append([idx, idx_prompt])
    
    return (num_eos_prompt, idx_eos_prompt), (num_eos_context, idx_eos_context)

def get_descriptive_stats(list_of_lists):
    # Flatten the list of lists
    flat_list = [num for sublist in list_of_lists for num in sublist]

    stats = {}

    stats['mean'] = statistics.mean(flat_list)
    stats['median'] = statistics.median(flat_list)
    try:
        stats['mode'] = statistics.mode(flat_list)
    except statistics.StatisticsError:
        stats['mode'] = 'No unique mode found'
    stats['stdev'] = statistics.stdev(flat_list)
    stats['variance'] = statistics.variance(flat_list)
    stats['min'] = min(flat_list)
    stats['max'] = max(flat_list)
    stats['range'] = stats['max'] - stats['min']

    return stats


def plot_and_test_distribution(list1, list2):
    """ Plot and test if the two generated code's length differences have a normal distribution"""

    # Flatten the lists
    flat_list1 = [num for sublist in list1 for num in sublist]
    flat_list2 = [num for sublist in list2 for num in sublist]

    # Ensure both lists have the same length for difference calculation
    min_length = min(len(flat_list1), len(flat_list2))

    # Calculate the differences
    differences = np.subtract(flat_list1[:min_length], flat_list2[:min_length])

    # Plot the distribution of differences
    sns.histplot(differences, kde=True)
    plt.title('Distribution of differences in code lengths')
    plt.xlabel('Difference in length')
    plt.ylabel('Frequency')
    plt.show()

    # Perform a normality test
    k2, p = stats.normaltest(differences)
    print("Normality test p-value:", p)
    if p < 0.05:
        print("The differences do not follow a normal distribution.")
    else:
        print("The differences follow a normal distribution.")


def perform_hypothesis_testing(list1, list2):
    """ Perform a hypothesis test to see if the difference is statistically significant!    """
    # Flatten the lists and keep only integer values
    flat_list1 = [num for sublist in list1 for num in sublist if isinstance(num, int)]
    flat_list2 = [num for sublist in list2 for num in sublist if isinstance(num, int)]

    stat, p = mannwhitneyu(flat_list1, flat_list2)

    return stat, p

def run_general_analysis(data, data_lenght_penalty = None, lenght_penalty = False):
    """ Run all analysis on prompt vs context data generated and also length """

    analysis = {
        'most_common_words_context':[],
        'most_common_words_prompt':[],
        'most_common_words_lenght_penalty':[],
        'count_eos_prompt':0,
        'count_eos_context':0,
        'count_eos_penalty':0,
        'indices_eos_prompt' : [],
        'indices_eos_context' : [],
        'indices_eos_penalty' : [],
        'statistic_prompt': [],
        'statistic_context': [],
        'statistic_penalty': []
    }

    # context and prompt post processing
    generated_code_prompt, generated_code_context = process_generated_codes(data, lenght_penalty)

    if lenght_penalty:

        # length penalty 
        _, generated_lenght_penalty= process_generated_codes(data_lenght_penalty, True)

        # get the most common words
        most_common_words_penalty = get_most_common_words(generated_lenght_penalty, [])

        # lenght of each generation : number of character
        lengths_penalty, _ = calculate_lengths(generated_lenght_penalty, generated_code_context)

        # calculate statistic:
        statistics = get_descriptive_stats(lengths_penalty)

        # counting and listing the number of EOS
        _, (count_eos_penalty_token, indices_eos_penalty_token) = count_number_of_eos(generated_code_prompt, generated_lenght_penalty)

        # adding the information
        analysis['most_common_words_lenght_penalty'] = most_common_words_penalty
        analysis['count_eos_penalty'] = count_eos_penalty_token
        analysis['indices_eos_penalty'] = indices_eos_penalty_token
        analysis['statistic_penalty'] = statistics
        
    else:
        # most common words
        most_common_words_prompt = get_most_common_words(generated_code_prompt, [])
        most_common_words_context = get_most_common_words([], generated_code_context)

        # lenghts of each generation : number of character
        lengths_prompt, lengths_context = calculate_lengths(generated_code_prompt, generated_code_context)

        # calculate statistic:
        statistics_context = get_descriptive_stats(lengths_context)
        statistics_prompt = get_descriptive_stats(lengths_prompt)

        # counting and listing the number of EOS
        (count_eos_prompt, indices_eos_prompt), (count_eos_context, indices_eos_context) = count_number_of_eos(generated_code_prompt, generated_code_context)

        # adding the information
        analysis['most_common_words_context'] = most_common_words_context
        analysis['most_common_words_prompt'] = most_common_words_prompt

        analysis['count_eos_prompt'] = count_eos_prompt
        analysis['indices_eos_prompt'] = indices_eos_prompt

        analysis['count_eos_context'] = count_eos_context
        analysis['indices_eos_context'] = indices_eos_context

        analysis['statistic_context'] = statistics_context
        analysis['statistic_prompt'] = statistics_prompt

    # returning all info
    return analysis


def run_hypothesis_testing(data, data_lenght_penalty = None, lenght_penalty = False):
    """ Hypothesis testing for the length. """
    # context and prompt post processing
    generated_code_prompt, generated_code_context = process_generated_codes(data, lenght_penalty)

    # lenghts of each generation : number of character
    lengths_prompt, lengths_context = calculate_lengths(generated_code_prompt, generated_code_context)

    if lenght_penalty:

        # length penalty generated code
        _, generated_lenght_penalty= process_generated_codes(data_lenght_penalty, True)

        # length of the length penalty generation 
        lengths_penalty, _ = calculate_lengths(generated_lenght_penalty, generated_code_context)

        # perform hypothesis testing
        statistic, p_value = perform_hypothesis_testing(lengths_penalty, lengths_context)
        plot_and_test_distribution(lengths_penalty, lengths_context)

    else:
        # perform hypothesis testing
        statistic, p_value = perform_hypothesis_testing(lengths_prompt, lengths_context)
        plot_and_test_distribution(lengths_prompt, lengths_context)
    
    return statistic, p_value



