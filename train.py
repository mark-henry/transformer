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
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train a transformer model')
    parser.add_argument('--checkpoint', type=str, help='Path to existing model checkpoint')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--context-size', type=int, default=512, help='Context size for transformer')
    return parser.parse_args()

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

def load_or_create_model(checkpoint_path, hidden_size, vocab_size, pad_token_id, seq_len, learning_rate):
    """Initialize model, optimizer, and training state, optionally from checkpoint."""
    model = Transformer(hidden_size, vocab_size, pad_token_id, seq_len=seq_len)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    start_epoch = 0
    best_loss = float('inf')

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print(f"Resumed from epoch {start_epoch} with best loss {best_loss:.4f}")
    else:
        print("Starting training from scratch")

    return model, optimizer, start_epoch, best_loss

def train():
    args = parse_args()
    accelerator = Accelerator()
    dataset = load_dataset("wikitext", "wikitext-2-v1")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.context_size),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    tokenized_dataset.set_format("torch")

    train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(tokenized_dataset["validation"], batch_size=args.batch_size, shuffle=False)

    config = BertConfig.from_pretrained('bert-base-uncased')
    model, optimizer, start_epoch, best_loss = load_or_create_model(
        args.checkpoint, config.hidden_size, tokenizer.vocab_size,
        tokenizer.pad_token_id, args.context_size, args.lr
    )

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    model, optimizer, criterion, train_dataloader = accelerator.prepare(
        model, optimizer, criterion, train_dataloader
    )

    Path("models").mkdir(exist_ok=True)
    model_path = f"models/transformer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids']

            targets = torch.roll(input_ids, -1, dims=1)
            targets[:, -1] = tokenizer.pad_token_id

            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1} Average Loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, model_path)

if __name__ == "__main__":
    train()