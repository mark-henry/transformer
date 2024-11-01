import argparse
import datetime
import math
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config

from transformer import Transformer


class TokenizedDataset(Dataset):
    def __init__(self, split, context_size, dataset_path):
        self.context_size = context_size
        dataset_head, dataset_name = os.path.split(dataset_path)
        cache_name = dataset_path.replace('/', '_')
        dataset = load_dataset(dataset_head, dataset_name)[split]
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        tokenized = dataset.map(
            lambda x: tokenizer(x['text'], truncation=False),
            batched=True,
            desc=f"Tokenizing {split}",
            num_proc=os.cpu_count() - 1,
            cache_file_name=f"cache/tokenized/{cache_name}_{split}.arrow",
        )
        # Store as np.int32 instead of int64 to halve memory usage
        self.tokens = np.concatenate(tokenized['input_ids']).astype(np.int32)
        self.num_examples = len(self.tokens) - context_size

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        # Get a slice of tokens starting at idx
        # To minimize GPU footprint, create torch tensor only when needed
        return {'input_ids': torch.tensor(self.tokens[idx:idx + self.context_size], dtype=torch.long)}

def parse_args():
    parser = argparse.ArgumentParser(description='Train a transformer model')
    parser.add_argument('--checkpoint', type=str, help='Path to existing model checkpoint')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    # NVIDIA A100-SXM4-80GB context 64 batch_size 1024
    # Tesla V100-SXM2-16GB context 64 batch_size 192
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--context-size', type=int, default=512, help='Context size for transformer')
    parser.add_argument('--embedding-size', type=int, default=768, help='Model embedding size')
    parser.add_argument('--head-count', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--layer-count', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--train-embeddings', action='store_true',
                       help='Train embeddings from scratch instead of using GPT2 pretrained')
    parser.add_argument("--dataset", type=str, default="Salesforce/wikitext/wikitext-2-v1",
                       help="HF dataset path and name to train on")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                       help="Maximum norm for gradient clipping")
    return parser.parse_args()


def load_or_create_model(args, vocab_size, pad_token_id):
    """Initialize model, optimizer, and training state, optionally from checkpoint."""
    # Get pretrained embeddings if not training from scratch
    embedding = None
    if not args.train_embeddings and not args.checkpoint:
        print("Loading pretrained GPT2 embeddings...")
        gpt2_config = GPT2Config.from_pretrained('gpt2')
        gpt2_config.n_embd = args.embedding_size
        gpt2 = GPT2Model.from_pretrained('gpt2', config=gpt2_config)
        embedding = gpt2.wte

    model = Transformer(
        args.embedding_size, vocab_size, pad_token_id,
        embedding=embedding,
        seq_len=args.context_size,
        num_attention_heads=args.head_count,
        num_layers=args.layer_count
    )

    # If using pretrained embeddings, don't train them
    if not args.train_embeddings:
        model.embedding.weight.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.lr)
    start_epoch = 0
    best_loss = float('inf')

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint {args.checkpoint} not found")
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Resumed from after epoch {start_epoch} with val perplexity {math.exp(best_loss):.4f}")

    return model, optimizer, start_epoch, best_loss


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids']
            outputs = model(input_ids)
            targets = torch.roll(input_ids, -1, dims=1)
            targets[:, -1] = input_ids[:, 0]
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train():
    args = parse_args()
    accelerator = Accelerator()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    print("loading datasets...")
    train_dataset = TokenizedDataset('train', args.context_size, args.dataset)
    val_dataset = TokenizedDataset('validation', args.context_size, args.dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    model, optimizer, start_epoch, best_loss = load_or_create_model(
        args, tokenizer.vocab_size, tokenizer.pad_token_id
    )

    criterion = nn.CrossEntropyLoss()
    model, optimizer, criterion, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, criterion, train_dataloader, val_dataloader
    )

    print("Starting training.", flush=True)
    Path("models").mkdir(exist_ok=True)
    model_path = f"models/transformer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_steps = 0
        epoch_loss = 0

        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch + 1}")
        for step, batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            outputs = model(input_ids)
            # Predict next token, using first token as target for last position
            targets = torch.roll(input_ids, -1, dims=1)
            targets[:, -1] = input_ids[:, 0]

            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            accelerator.backward(loss)

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    accelerator.unwrap_model(model).parameters(),
                    max_norm=args.grad_clip
                )

            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            avg_loss = epoch_loss / epoch_steps
            progress_bar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'epoch_loss': f'{avg_loss:.3f}',
                'epoch_ppl': f'{math.exp(avg_loss):.0f}'
            })

        # Epoch completion stats
        train_loss = epoch_loss / len(train_dataloader)
        val_loss = evaluate(model, val_dataloader, criterion)
        print(f'\nEpoch {epoch + 1} Complete:')
        print(f'Train Loss: {train_loss:.4f} (ppl: {math.exp(train_loss):.0f})')
        print(f'Val Loss: {val_loss:.4f} (ppl: {math.exp(val_loss):.0f})')

        # Save if validation improves
        if val_loss < best_loss:
            best_loss = val_loss
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_loss': best_loss,
            }, model_path)
            print(f"New best model saved to {model_path}")


if __name__ == "__main__":
    train()
