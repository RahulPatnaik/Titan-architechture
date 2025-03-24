import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from titan.memory_modules import MAGModule  # Ensure your titan modules are in PYTHONPATH

# Hyperparameters
d_model = 64
num_classes = 2
batch_size = 16
max_length = 64  # Maximum token length for each sample
num_epochs = 3
learning_rate = 1e-3
memory_loss_weight = 0.1  # Weight for the associative memory loss

# Load SST2 dataset (GLUE subset)
dataset = load_dataset("glue", "sst2")

# Initialize tokenizer (we only use its vocabulary)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Custom Dataset class
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
        attention_mask = encoding["attention_mask"].squeeze(0)  # unused here, but could be used later
        return input_ids, label

# Create training and validation datasets and loaders
train_dataset = SST2Dataset("train", tokenizer, max_length)
val_dataset = SST2Dataset("validation", tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define the Titan-based classifier model
class TitanClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_classes, max_length):
        super(TitanClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Use your MAGModule from titan/memory_modules.py
        self.titan = MAGModule(d_model=d_model)
        # Classification head: average pooled representation -> linear layer
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        x = self.embedding(input_ids)  # (batch, seq_len, d_model)
        titan_out, mem_loss = self.titan(x)  # (batch, seq_len, d_model), scalar loss per batch element aggregated later
        # Mean pooling over the sequence dimension
        x_pooled = titan_out.mean(dim=1)  # (batch, d_model)
        logits = self.classifier(x_pooled)  # (batch, num_classes)
        return logits, mem_loss

# Instantiate model, loss, and optimizer
vocab_size = tokenizer.vocab_size
model = TitanClassifier(vocab_size, d_model, num_classes, max_length)
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
        logits, mem_loss = model(input_ids)
        # Classification loss
        cls_loss = criterion(logits, labels)
        # Combine losses
        loss = cls_loss + memory_loss_weight * mem_loss
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
            logits, mem_loss = model(input_ids)
            cls_loss = criterion(logits, labels)
            loss = cls_loss + memory_loss_weight * mem_loss
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

# Save the trained model (optional)
torch.save(model.state_dict(), "titan_classifier.pt")
