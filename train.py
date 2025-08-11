from model import GptModel
import torch
from dataloader import train_dataloader, test_dataloader, validation_dataloader, tokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

seq_len=100
epochs = 3
vocab_size = len(tokenizer)
d_model = 256
h = 8
num_layers = 4
d_ff = 1024
batch_size = 8
lr = 1e-4
dropout = 0.1

model = GptModel(seq_len=seq_len, vocab_size=vocab_size, d_model=d_model, num_layers=num_layers, d_ff=d_ff, h=h, dropout=dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

def evaluate(model, dataloader):
    total_loss = 0 
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            logits = model(input_ids)
            labels = input_ids[:, 1:].contiguous()
            logits = logits[:, :-1,:].contiguous()

            loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
            total_loss +=loss.item()
        return total_loss / len(dataloader) 

total_loss_sum = 0 
for epoch in range(epochs):
    model.train()
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        optimizer.zero_grad()
        logits = model(input_ids)

        labels = input_ids[:, 1:].contiguous()
        logits = logits[:, :-1, :].contiguous()
        loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss_sum = loss.item()
        train_loss = total_loss_sum / len(train_dataloader)
        val_loss = evaluate(model, validation_dataloader)
        epoch+=0.01
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

test_loss = evaluate(model, test_dataloader)
print(f"Test Loss: {test_loss:.4f}")

torch.save(model.state_dict(), "gpt_model.pth")
