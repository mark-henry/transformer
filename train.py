from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertConfig
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from accelerate import Accelerator
from pathlib import Path
import datetime
from transformer import Transformer

def tokenize_function(examples, tokenizer, context_size):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=context_size
    )

def pad(token_ids, length, pad_token_id):
    padded_ids = torch.full([length], pad_token_id, dtype=torch.long)
    padded_ids[:token_ids.size(0)] = token_ids
    return padded_ids

def train():
    # Initialize accelerator
    accelerator = Accelerator()

    # Dataset and tokenizer setup
    print("Loading tokenizer and dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-v1")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    context_size = 512

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, context_size),
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

    # Model setup
    config = BertConfig.from_pretrained('bert-base-uncased')
    model = Transformer(config.hidden_size, tokenizer.vocab_size, tokenizer.pad_token_id, seq_len=context_size)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Prepare with accelerator
    model, optimizer, criterion, train_dataloader = accelerator.prepare(
        model, optimizer, criterion, train_dataloader
    )

    # Training loop
    num_epochs = 3
    subset_size = len(train_dataloader)

    Path("models").mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/transformer_{timestamp}.pt"

    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
            if i >= subset_size:
                break

            optimizer.zero_grad()
            input_ids = batch['input_ids']
            batch_loss = 0

            for sequence in input_ids:
                # Move sequence to device before padding
                sequence = accelerator.prepare(sequence)

                # Pad sequence to context_size
                padded_sequence = pad(sequence, context_size, tokenizer.pad_token_id)
                padded_sequence = padded_sequence.to(sequence.device)  # ensure same device

                # Create target by shifting input right by 1
                target = torch.roll(padded_sequence, -1)
                target[-1] = tokenizer.pad_token_id
                target = target.to(padded_sequence.device)  # ensure same device

                # Forward pass
                output = model(padded_sequence)

                # Compute loss
                loss = criterion(output, target)
                batch_loss += loss

            # Average loss over batch
            batch_loss = batch_loss / input_ids.size(0)

            # Backward pass using accelerator
            accelerator.backward(batch_loss)
            optimizer.step()

            total_loss += batch_loss.item()

        avg_loss = total_loss / subset_size
        print(f'Epoch {epoch + 1} Average Loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save model state - unwrap model first
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, model_path)
            print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()