import math
import os

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertConfig, AutoModel
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
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    # batch_size 150 for 80 GB 1A100.22V
    # Tesla V100-SXM2-16GB context 64 batch_size 192
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--context-size', type=int, default=512, help='Context size for transformer')
    parser.add_argument('--embedding-size', type=int, default=768, help='Model embedding size')
    parser.add_argument('--head-count', type=int, default=8, help='How many heads of attention')
    parser.add_argument('--layer-count', type=int, default=6, help='How many layers in the transformer')
    parser.add_argument('--train-embeddings', action='store_true',
                        help='Train embeddings from scratch instead of using BERT pretrained')
    parser.add_argument("--dataset", type=str, default="Salesforce/wikitext/wikitext-2-v1", help="HF dataset path and name to train on")
    return parser.parse_args()


def tokenize_function(examples, tokenizer, context_size):
    outputs = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=context_size
    )

    # Filter special tokens from each sequence
    special_tokens = [tokenizer.cls_token_id, tokenizer.sep_token_id,
                      tokenizer.mask_token_id]
    cleaned_ids = []
    for seq in outputs['input_ids']:
        cleaned = [token for token in seq if token not in special_tokens]
        # Re-pad to context_size if needed
        if len(cleaned) < context_size:
            cleaned.extend([tokenizer.pad_token_id] * (context_size - len(cleaned)))
        cleaned_ids.append(cleaned[:context_size])

    return {'input_ids': cleaned_ids}


def load_or_create_model(checkpoint_path, embedding_size, vocab_size, pad_token_id, seq_len,
                         learning_rate, head_count, layer_count, train_embeddings):
    """Initialize model, optimizer, and training state, optionally from checkpoint."""
    # Get pretrained embeddings if not training from scratch
    embedding = None
    if not train_embeddings and not checkpoint_path:
        print("Loading pretrained BERT embeddings...")
        bert = AutoModel.from_pretrained('bert-base-uncased')
        print("BERT embeddings loaded")
        embedding = bert.embeddings.word_embeddings
        embedding_size = bert.config.hidden_size

    model = Transformer(
        embedding_size, vocab_size, pad_token_id,
        embedding=embedding,
        seq_len=seq_len,
        num_attention_heads=head_count,
        num_layers=layer_count
    )

    # If using pretrained embeddings, don't train them
    if not train_embeddings:
        model.embedding.weight.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=learning_rate)
    start_epoch = 0
    best_loss = float('inf')

    if checkpoint_path:
        if not Path(checkpoint_path).exists():
            raise ValueError(f"Checkpoint {checkpoint_path} not found")
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        val_loss = checkpoint['val_loss']
        print(f"Resumed from after epoch {start_epoch} with val loss {val_loss:.4f}")

    return model, optimizer, start_epoch, best_loss


def evaluate(model, dataloader, criterion, tokenizer):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids']
            targets = torch.roll(input_ids, -1, dims=1)
            targets[:, -1] = tokenizer.pad_token_id

            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train():
    args = parse_args()
    accelerator = Accelerator()

    print("tokenizing dataset...")
    dataset_head, dataset_name = os.path.split(args.dataset)
    dataset = load_dataset(dataset_head, dataset_name)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.context_size),
        batched=True,
        remove_columns=dataset["train"].column_names,
        cache_file_names={
            "test": f"cache/tokenized/{args.dataset}/test",
            "train": f"cache/tokenized/{args.dataset}/train",
            "validation": f"cache/tokenized/{args.dataset}/validation"
        }
    )
    tokenized_dataset.set_format("torch")

    train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(tokenized_dataset["validation"], batch_size=args.batch_size, shuffle=False)
    print("dataset loaded")

    config = BertConfig.from_pretrained('bert-base-uncased')
    model, optimizer, start_epoch, best_loss = load_or_create_model(
        args.checkpoint, args.embedding_size, tokenizer.vocab_size,
        tokenizer.pad_token_id, args.context_size, args.lr,
        args.head_count, args.layer_count, args.train_embeddings
    )

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    model, optimizer, criterion, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, criterion, train_dataloader, val_dataloader
    )

    print("Starting training.", flush=True)
    Path("models").mkdir(exist_ok=True)
    model_path = f"models/transformer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"

    best_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_steps = 0
        epoch_loss = 0

        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch + 1}")
        for step, batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids']

            targets = torch.roll(input_ids, -1, dims=1)
            targets[:, -1] = tokenizer.pad_token_id

            outputs = model(input_ids)
            loss = criterion(outputs[:, :-1, :].reshape(-1, outputs.size(-1)), input_ids[:, 1:].reshape(-1))

            accelerator.backward(loss)
            optimizer.step()

            # Update running statistics
            epoch_loss += loss.item()
            epoch_steps += 1

            # Update progress bar every step
            avg_loss = epoch_loss / epoch_steps
            progress_bar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'epoch_loss': f'{avg_loss:.3f}',
                'epoch_perplexity': f'{math.exp(avg_loss):.0f}'
            })

        # Epoch completion stats
        train_loss = epoch_loss / len(train_dataloader)
        val_loss = evaluate(model, val_dataloader, criterion, tokenizer)
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
