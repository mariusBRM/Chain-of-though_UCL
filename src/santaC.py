import torch
import re
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import numpy as np
import random
from torch.utils.data import DataLoader
from tokenizer import *


#################################
#       Define SantaCoder       #
#################################

class MySantaCoder(nn.Module):
    def __init__(self, generation_method, max_tokens = 128, num_sol = 1):
        super(MySantaCoder, self).__init__()
        self.checkpoint = "bigcode/santacoder"
        # self.checkpoint = model_path_to_hub
        self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.max_new_tokens = max_tokens

        if generation_method == 'GrdS':

            self.generation_config = GenerationConfig(
                num_beams = num_sol,
                num_return_sequences = num_sol,
                max_new_tokens = self.max_new_tokens,
                eos_token_id=self.model.generation_config.eos_token_id,
                bos_token_id=self.model.generation_config.bos_token_id
                )
        elif generation_method == 'SmplM' : 
     
            self.generation_config = GenerationConfig(   
                do_sample = True,  
                num_beams = num_sol,
                num_return_sequences = num_sol,
                top_p = 0.8,
                temperature = 0.95,
                max_new_tokens = self.max_new_tokens,
                eos_token_id=self.model.generation_config.eos_token_id,
                bos_token_id=self.model.generation_config.bos_token_id
                )

    def forward(self, input_ids):
        # input_ids = input_ids.unsqueeze(0)
        outputs = self.model.generate(input_ids, self.generation_config)
        return outputs

    def decode_output(self, encoded_output):
        output = self.tokenizer.decode(encoded_output)
        return output

    def post_generation_processing(self,code):
        # split it into list of blocks
        list_blocks = re.split('def |class |assert |print ', code)
        if 'init' in list_blocks[1]:
            fill_word = '\nclass '
        else:
            fill_word = '\ndef '
        # keep only the first block
        result = list_blocks[0] + fill_word + list_blocks[1]
        return result

# in case we want to fix the right generation token that we want 
def find_max_token(df):
    max_token = 0
    avg = 0
    nm_to_be_generated = 0
    for i in range(len(df)):
        
        text = len(df.iloc[i]['text'])
        code = len(df.iloc[i]['code'])
        nm_to_be_generated += 10 + text
        avg += code + text
        nm_tokens = 1.1 * (text + code)
        if  nm_tokens > max_token:
            max_token = nm_tokens
    
    return max_token, avg/len(df), nm_to_be_generated/len(df)
    
#################################
#       Define Training         #
#################################

def loss_fn(output, target):
    return nn.BCELoss()(output, target)

def validation(validation_loader, model):
    """
    This function evaluate the model on the validation data
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    fin_targets=[]
    fin_outputs=[]
    running_loss = 0.0

    with torch.no_grad():
        for _, data in enumerate(validation_loader, 0):

            ids = data['input_ids'].to(device, dtype = torch.long)
            targets = data['labels'].to(device, dtype = torch.long)
            
            # forward
            output = model.forward(ids)
            # evaluate the loss
            loss = loss_fn(output, targets)
 
            # adding to list
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(output.cpu().detach().numpy().tolist())

            # add the loss to the running loss
            running_loss+=loss.item()

    return fin_outputs, fin_targets, running_loss/len(validation_loader)

def training_model(nb_epochs, train_dataloader, val_dataloader, patience):
    """
    This function trains the model on training data
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MySantaCoder()
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)
    best_val_loss = np.inf
    # keep track of the performances
    summary = []


    for epoch in range(nb_epochs):
            # dict containing the information
        report_epoch = {
                'epoch': epoch+1,
                'training_loss': 0.0,
                'valid_loss':0.0
            }
        model.train()
        running_loss = 0.0

        for i, data in enumerate(train_dataloader, 0):

            ids = data['input_ids'].to(device, dtype = torch.long)
            labels = data['labels'].to(device, dtype = torch.long)
            
             # initialize the optimizer
            optimizer.zero_grad()
            #forward inputs
            output = model.forward(ids) 
            # define the loss
            loss = loss_fn(output, labels)
            # backpropagate
            loss.backward()
            optimizer.step()
            # add the loss to the running loss
            running_loss+=loss.item()
            
            print('\rEpoch: {}\tbatch: {}\tLoss =  {:.3f}'.format(epoch+1, i, loss), end="")

        running_loss = running_loss / len(train_dataloader)
        report_epoch['training_loss'] = running_loss
        print("\n")
        # validation
        model.eval()
        with torch.no_grad():

            outputs, targets, val_loss = validation(validation_loader=val_dataloader, model= model)

            report_epoch['valid_loss'] = val_loss
            # set to evaluate potential metrics...

            print(f"Epoch {epoch+1}: train CE loss = {running_loss}", f"|| Valid: CE loss = {val_loss}")
            

        # early-stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            dict_model = model.state_dict()
            pat = 0
        else:
            pat += 1
            print("pat ", pat)
            if pat == patience:
                print("Early Stopping: Validation Loss did not decrease for", patience, "epochs.")
                break
        
        # save model each epoch
        torch.save(model.state_dict(), 'epoch'+str(epoch+1)+'.pt')

        print("\n")
        # add performances of the epoch to the overall summary
        summary.append(report_epoch)

    torch.save(dict_model, 'Fine_Tuned_SantaCoder.pt')
    
    return summary

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__=="__name__":
    
    # load the data
    train_data, val_data = ''
    PATH_TO_HUB = 'bigcode/santacoder'

    g = torch.Generator()
    g.manual_seed(0)

    # Tokenizing datasets
    trainset = MyTokenizer(train_data, PATH_TO_HUB)
    valset = MyTokenizer(val_data, PATH_TO_HUB)

    # Create dataloader
    batch_size = 1
    num_workers = 0

    trainloader = dataloading(train_data, PATH_TO_HUB, batch_size, num_workers, g, seed_worker)
    validloader = dataloading(val_data, PATH_TO_HUB, batch_size, num_workers, g, seed_worker)

    # set training parameters ( look out how to enhance this)
    nm_epochs = 5
    patience = 2
    summary = training_model(nm_epochs, trainloader, validloader, patience)