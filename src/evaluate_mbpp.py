from santaC import *
from tokenizer import *
import sys
import pandas as pd

# eval
def eval(path_to_hub, path_to_data, path_to_save, early_stop = 3):

    print('Start to instanciate model and data...')
    # instantiate the model
    model = MySantaCoder(path_to_hub)
    # define the data
    data= pd.read_csv(path_to_data)

    mbpp_data = MyTokenizer(
        data=data,
        path_to_hub=path_to_hub
    )

    results = []
    model.eval()

    print('Start code generation...')
    for i in range(len(mbpp_data)):
        output = model(mbpp_data[i])
        result = model.decode_output(output)
        results.append(result)
        if i > early_stop:
            break
    
    data['Gen_code'] = results
    print('Save generated data ...')
    data.to_csv(path_to_save + "mbpp_generated.csv", index=False)

    return results



def main():

    #define the variables
    
    return None


if __name__=="__main__":
    sys.exit(main())