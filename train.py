from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import BertTokenizer, BertConfig
from transformer import Transformer

dataset = load_dataset("wikitext", "wikitext-2-v1")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
context_size = 512


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=context_size
    )


# Tokenize the dataset
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# Convert to PyTorch format
tokenized_dataset.set_format("torch")

# Create dataloaders
train_dataloader = DataLoader(
    tokenized_dataset["train"],
    batch_size=8,
    shuffle=True
)

val_dataloader = DataLoader(
    tokenized_dataset["validation"],
    batch_size=8,
    shuffle=False
)

config = BertConfig.from_pretrained('bert-base-uncased')
transformer = Transformer(config.hidden_size, tokenizer.vocab_size, tokenizer.pad_token_id, seq_len=context_size)


def pad(token_ids, length):
    padded_ids = torch.full([length], tokenizer.pad_token_id, dtype=torch.long)
    padded_ids[:token_ids.size(0)] = token_ids
    return padded_ids


# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer = transformer.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(transformer.parameters(), lr=1e-4)


def train():
    # Training loop
    num_epochs = 3
    subset_size = len(train_dataloader) // 256

    import time
    from pathlib import Path
    import datetime

    Path("models").mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/transformer_{timestamp}.pt"

    best_loss = float('inf')
    for epoch in range(num_epochs):
        transformer.train()
        total_loss = 0

        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
            if i >= subset_size:
                break

            optimizer.zero_grad()

            # Get input_ids from batch
            input_ids = batch['input_ids']

            batch_loss = 0
            # Process each sequence in the batch individually
            for sequence in input_ids:
                # Pad sequence to context_size
                padded_sequence = pad(sequence, context_size)
                padded_sequence = padded_sequence.to(device)

                # Create target by shifting input right by 1
                target = torch.roll(padded_sequence, -1)
                target[-1] = tokenizer.pad_token_id
                target = target.to(device)

                # Forward pass
                output = transformer(padded_sequence)

                # Compute loss
                loss = criterion(output, target)
                batch_loss += loss

            # Average loss over batch
            batch_loss = batch_loss / input_ids.size(0)

            # Backward pass
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()

        avg_loss = total_loss / subset_size
        print(f'Epoch {epoch + 1} Average Loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save model state
            torch.save({
                'epoch': epoch,
                'model_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, model_path)
            print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train()
