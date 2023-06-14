from santaC import *
from tokenizer import *
import sys
import pandas as pd


# make sure to add all device in needed time
def eval(test_loader, model):
    """
    This function evaluate the model on the test data
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    fin_targets=[]
    fin_outputs=[]
    fin_tests = []
    running_loss = 0.0

    with torch.no_grad():
        for _, data in enumerate(test_loader, 0):

            ids = data['input_ids'].to(device, dtype = torch.long)
            targets = data['labels'].to(device, dtype = torch.long)
            tests = data['tests'].to(device, dtype = torch.long)
            # forward
            output = model.forward(ids)
            # this is supposed to be a list of 200 sequence

            # decode and process the output
            decoded_output = model.decode_output(output)
            code_generated = model.post_generation_processing(decoded_output)

            # decode the label and the tests
            label = model.decode_output(targets)
            test_list = model.decode_output(tests)

            fin_targets.append()
            fin_outputs.append()
            fin_tests.append()

    return fin_outputs, fin_targets, running_loss/len(test_loader)

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



if __name__=="__main__":
    print('dfdsf')
