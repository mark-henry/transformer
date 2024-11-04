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
    """Dataset that yields padded sliding windows over tokenized text documents.

    For a context size of 3 and document "a b c d", yields examples:
        inputs      targets
        [p p a] --> [p a b]
        [p a b] --> [a b c]
        [a b c] --> [b c d]
        [b c d] --> [c d e]
    where p=pad_token and e=eos_token.

    This approach ensures the model:
    1. Learns to handle short prompts via left-padded examples
    2. Sees all possible subsequences of each document
    3. Properly learns next-token prediction, not input repetition

    Memory efficient: Stores only raw tokens and computes examples on-the-fly.

    Args:
        split (str): Dataset split to use ('train' or 'validation')
        context_size (int): Size of input sequence windows
        dataset_path (str): HuggingFace dataset path (e.g. 'Salesforce/wikitext/wikitext-2-v1')

    Returns:
        Dict with keys:
            input_ids: torch.LongTensor of shape (context_size)
            target_ids: torch.LongTensor of shape (context_size)
    """

    def __init__(self, split, context_size, dataset_path):
        self.context_size = context_size
        dataset_head, dataset_name = os.path.split(dataset_path)
        dataset = load_dataset(dataset_head, dataset_name)[split]

        self.tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2",
                                                       add_prefix_space=True,
                                                       add_bos_token=True)
        self.tokenizer.pad_token = self.tokenizer.bos_token

        tokenized = dataset.map(
            lambda x: self.tokenizer(x['text'], truncation=False),
            batched=True,
            desc=f"Tokenizing {split}",
            num_proc=os.cpu_count() - 1,
            cache_file_name=f"cache/tokenized/{dataset_path.replace('/', '_')}_{split}.arrow",
        )

        # Store tokens with padding and EOS for each document
        tokens = []
        for token_ids in tokenized['input_ids']:
            padded = np.concatenate([
                np.full(context_size, self.tokenizer.pad_token_id),
                token_ids,
                [self.tokenizer.eos_token_id]
            ])
            tokens.extend(padded)

        self.tokens = np.array(tokens, dtype=np.int32)
        self.stride = context_size + 1  # Distance between start of each document

    def __len__(self):
        return (len(self.tokens) - self.context_size) // self.stride * self.stride

    def __getitem__(self, idx):
        start_idx = idx
        input_ids = self.tokens[start_idx:start_idx + self.context_size]
        target_ids = self.tokens[start_idx + 1:start_idx + self.context_size + 1]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long)
        }


def parse_args():
    parser = argparse.ArgumentParser(description='Train a transformer model')
    parser.add_argument('--checkpoint', type=str, help='Path to existing model checkpoint')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--context-size', type=int, default=128, help='Context size for transformer')
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Resumed from after epoch {start_epoch} with val perplexity {math.exp(best_loss):.4f}")

    return model, optimizer, start_epoch, best_loss


def evaluate(model, dataloader, criterion, max_batches=250):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            input_ids = batch['input_ids']
            target_ids = batch['target_ids']
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            total_loss += loss.item()
    return total_loss / min(max_batches, len(dataloader))


def train():
    args = parse_args()
    accelerator = Accelerator()
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2",
                                              add_prefix_space=True,
                                              add_bos_token=True)
    tokenizer.pad_token = tokenizer.bos_token

    print("loading datasets...")
    train_dataset = TokenizedDataset('train', args.context_size, args.dataset)
    val_dataset = TokenizedDataset('validation', args.context_size, args.dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
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
    base_path = f"models/transformer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    checkpoint_freq = 100000  # num examples between checkpoints

    for epoch in range(start_epoch, max(start_epoch + 1, args.epochs)):
        last_checkpoint = 0
        examples_processed = 0
        model.train()
        epoch_loss = 0
        epoch_steps = 0

        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch + 1}")
        for step, batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            target_ids = batch['target_ids']
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            accelerator.backward(loss)

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    accelerator.unwrap_model(model).parameters(),
                    max_norm=args.grad_clip
                )

            optimizer.step()
            examples_processed += args.batch_size

            epoch_loss += loss.item()
            epoch_steps += 1
            avg_loss = epoch_loss / epoch_steps

            progress_bar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'avg_loss': f'{avg_loss:.3f}',
                'ppl': f'{math.exp(avg_loss):.0f}',
                'examples': examples_processed
            })

            if examples_processed - last_checkpoint >= checkpoint_freq:
                checkpoint_suffix = f"{epoch}_{examples_processed}"
                checkpoint(accelerator, avg_loss, base_path, checkpoint_suffix, criterion, epoch, examples_processed,
                           model, optimizer, val_dataloader)
                last_checkpoint = examples_processed

        checkpoint(accelerator, avg_loss, base_path, f"epoch{epoch}", criterion, epoch, examples_processed,
                   model, optimizer, val_dataloader)


def checkpoint(accelerator, avg_loss, base_path, checkpoint_suffix, criterion, epoch, examples_processed, model,
               optimizer, val_dataloader):
    val_loss = evaluate(model, val_dataloader, criterion)
    print(f'\nEpoch {epoch} examples {examples_processed}:')
    print(f'Train Loss: {avg_loss:.4f} (ppl: {math.exp(avg_loss):.0f})')
    print(f'Val Loss: {val_loss:.4f} (ppl: {math.exp(val_loss):.0f})')
    unwrapped_model = accelerator.unwrap_model(model)
    checkpoint_path = f"{base_path}_{checkpoint_suffix}.pt"
    torch.save({
        'examples': examples_processed,
        'epoch': epoch,
        'model_state_dict': unwrapped_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_loss,
        'val_loss': val_loss,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}", flush=True)
    model.train()


if __name__ == "__main__":
    train()
