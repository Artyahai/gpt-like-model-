from model import GptModel
from dataloader import tokenizer
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'


seq_len=100
vocab_size = len(tokenizer)
d_model = 256
h = 8
num_layers = 4
d_ff = 1024
dropout = 0.1
model = GptModel(d_model=d_model, vocab_size=vocab_size, seq_len=seq_len, h=h, d_ff=d_ff,
                   num_classes=len(tokenizer), num_layers=num_layers, dropout=dropout) # Define Model
model.load_state_dict(torch.load("gpt_model.pth", map_location=device)) # load pretrained model
model.to(device) #transpose model to device
model.eval() # put model to the inference mode 


while True: # infinite loop 
    sentence = input('Test a model: ') # sentence which we use to test the model
    decoding = tokenizer(
        sentence, 
        return_tensors='pt',
        padding = 'max_length',
        truncation=True,
        max_length = 100
    ) # tokenized it
    input_ids = decoding['input_ids'].to(device) # define input ids
    attention_mask = decoding['attention_mask'].to(device) # define attention masks
    if sentence == 'exit': # make an exit mechanism 
        break
    with torch.no_grad():
        outputs = model(input_ids)
        _, predicted = torch.max(outputs, dim=-1)
    print(predicted[0][0].item())
