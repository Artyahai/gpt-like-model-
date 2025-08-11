from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
dataset = dataset.filter(lambda x: x["text"] is not None and x["text"].strip() != "")


def tokenize_func(example):
    return tokenizer(example['text'], truncation = True, max_length = 100, padding ='max_length')
tokenized_dataset = dataset.map(tokenize_func)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])


test_dataset = tokenized_dataset['test']
train_dataset = tokenized_dataset['train']
validation_dataset = tokenized_dataset['validation']


train_dataloader = DataLoader(train_dataset, batch_size = 8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size = 8)
validation_dataloader = DataLoader(validation_dataset, batch_size=8)

