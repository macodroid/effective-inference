import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, T5ForConditionalGeneration


class NLtoPLDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length, max_target_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_text = self.data.iloc[index, 0]
        output_text = self.data.iloc[index, 1]

        input = self.tokenizer.batch_encode_plus([input_text], max_length=self.max_input_length,
                                                     padding='max_length',
                                                     truncation=True, return_tensors="pt")
        output = self.tokenizer.batch_encode_plus([output_text], max_length=self.max_target_length,
                                                      padding='max_length', truncation=True, return_tensors="pt")

        input_ids = input['input_ids'].squeeze()
        input_mask = input["attention_mask"].squeeze()
        output_ids = output["input_ids"].squeeze()
        output_mask = output["attention_mask"].squeeze()

        return {
            "input_ids": input_ids.to(dtype=torch.long),
            "input_mask": input_mask.to(dtype=torch.long),
            "output_ids": output_ids.to(dtype=torch.long),
            "output_mask": output_mask.to(dtype=torch.long)
        }


def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids = batch['input_ids'].to(device, dtype=torch.long)
        input_mask = batch['input_mask'].to(device, dtype=torch.long)
        output_ids = batch['output_ids'].to(device, dtype=torch.long)
        y_ids = output_ids[:, :-1].contiguous()
        lm_labels = output_ids[:, 1:].clone().detach()
        lm_labels[output_ids[:, 1:] == tokenizer.pad_token_id] = -100

        outputs = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )

        loss = outputs[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch: {epoch}, Loss: {total_loss / len(loader)}")


def val_test(tokenizer, model, device, loader, max_target_length):
    model.eval()
    predictions = []
    actual = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device, dtype=torch.long)
            input_mask = batch['input_mask'].to(device, dtype=torch.long)
            y = batch['output_ids'].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=input_mask,
                max_length=max_target_length,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )

            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                     generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]

            predictions.extend(preds)
            actual.extend(target)
    return predictions, actual


train_data = pd.read_csv('data/ProblemSolutionPythonV3_TRAIN.csv').iloc[:, 1:]
test_data = pd.read_csv('data/ProblemSolutionPythonV3_TEST.csv').iloc[:, 1:]
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')
model.to(device)

train_size = 0.8
train_dataset = train_data.sample(frac=train_size)
val_dataset = train_data.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

train_ds = NLtoPLDataset(train_dataset, tokenizer, 512, 512)
val_ds = NLtoPLDataset(val_dataset, tokenizer, 512, 512)
test_ds = NLtoPLDataset(test_data, tokenizer, 512, 512)

train_loader = DataLoader(batch_size=8, dataset=train_ds, shuffle=True, num_workers=4)
val_loader = DataLoader(batch_size=1, dataset=val_ds, shuffle=False, num_workers=1)
test_loader = DataLoader(batch_size=1, dataset=test_ds, shuffle=False, num_workers=1)

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)

for epoch in range(5):
    train(epoch, tokenizer, model, device, train_loader, optimizer)
    predictions, actual = val_test(tokenizer, model, device, val_loader, 512)
    print(predictions)
    print(actual)
