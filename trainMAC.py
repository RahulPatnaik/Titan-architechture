import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

# Import MACModule from your titan package
from titan.memory_modules import MACModule

# Hyperparameters
d_model = 64
num_classes = 2
batch_size = 16
max_length = 64  # maximum token length for each sample
num_epochs = 3
learning_rate = 1e-3

# Load the SST2 dataset from GLUE
dataset = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Custom dataset class for SST2
class SST2Dataset(Dataset):
    def __init__(self, split, tokenizer, max_length):
        self.samples = dataset[split]
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]["sentence"]
        label = self.samples[idx]["label"]
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze(0)  # shape: (max_length)
        return input_ids, label

# Define a TitanClassifierMAC that uses the MAC variant.
class TitanClassifierMAC(nn.Module):
    def __init__(self, vocab_size, d_model, num_classes, max_length, persistent_len=10):
        super(TitanClassifierMAC, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Use MACModule from titan/memory_modules.py
        self.mac = MACModule(d_model, persistent_len=persistent_len)
        # Classification head: average pooled representation -> linear layer
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        x = self.embedding(input_ids)  # shape: (batch, seq_len, d_model)
        
        # For MAC, we split the sequence into two parts:
        # For example, first half as current segment and second half as historical memory.
        seq_len = x.size(1)
        half = seq_len // 2
        current_segment = x[:, :half, :]         # shape: (batch, half, d_model)
        historical_memory = x[:, half:, :]         # shape: (batch, seq_len-half, d_model)
        
        # Pass segments through MACModule.
        # MACModule concatenates persistent memory, historical memory, and current segment.
        mac_out = self.mac(current_segment, historical_memory)  # shape: (batch, persistent_len + seq_len, d_model)
        
        # Mean pooling over the token dimension
        pooled = mac_out.mean(dim=1)  # shape: (batch, d_model)
        
        # Classification logits
        logits = self.classifier(pooled)  # shape: (batch, num_classes)
        return logits

# Instantiate dataset and dataloaders
train_dataset = SST2Dataset("train", tokenizer, max_length)
val_dataset = SST2Dataset("validation", tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Instantiate the model
vocab_size = tokenizer.vocab_size
model = TitanClassifierMAC(vocab_size, d_model, num_classes, max_length)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(logits, dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += input_ids.size(0)
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item() * input_ids.size(0)
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += input_ids.size(0)
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

# Training loop
for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
    print(f"Epoch {epoch}:")
    print(f"  Train Loss: {train_loss:.4f}  Train Accuracy: {train_acc:.4f}")
    print(f"  Val Loss:   {val_loss:.4f}  Val Accuracy:   {val_acc:.4f}")

# Optionally save the trained model
torch.save(model.state_dict(), "titan_classifier_mac.pt")
