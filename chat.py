import torch
from transformers import AutoTokenizer
from accelerate import Accelerator
from pathlib import Path
import math

from transformer import Transformer


def pad(token_ids, length, pad_token_id):
    padded_ids = torch.full([length], pad_token_id, dtype=torch.long)
    padded_ids[:token_ids.size(0)] = token_ids
    return padded_ids


def load_model(model_path):
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Load checkpoint and configuration
    checkpoint = torch.load(model_path, weights_only=True)
    model_state = checkpoint['model_state_dict']

    # Infer embedding size from embedding weights
    embedding_weight = model_state['embedding.weight']
    vocab_size, embedding_size = embedding_weight.shape

    # Get sequence length from positional encoding
    seq_len = model_state['pe'].shape[0]

    # Count attention heads from first layer
    head_count = sum(1 for k in model_state.keys()
                     if k.startswith('decoder_layers.0.attention_heads.')
                     and k.endswith('.Q.weight'))

    # Count layers
    layer_count = sum(1 for k in model_state.keys()
                      if k.startswith('decoder_layers.')
                      and k.endswith('.layer_norm1.weight'))

    print(f"Loaded model config: embedding_size={embedding_size}, seq_len={seq_len}, heads={head_count}, "
          f"layers={layer_count}, epoch={checkpoint['epoch']} val_perplexity={math.exp(checkpoint['val_loss'])}")

    model = Transformer(
        embedding_size=embedding_size,
        vocab_size=vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        seq_len=seq_len,
        num_attention_heads=head_count,
        num_layers=layer_count
    )

    model.load_state_dict(model_state)
    model = accelerator.prepare(model)
    model.eval()

    return model, tokenizer, accelerator


def generate_text(model, tokenizer, accelerator, prompt, max_length=50, temperature=0.1):
    model.eval()
    # Get the device from the model
    device = next(model.parameters()).device

    # Move input to correct device
    prompt += ' ' + tokenizer.sep_token
    input_ids = tokenizer.encode(prompt, return_tensors='pt')[0].to(device)

    generated = []
    with torch.no_grad():
        for _ in range(max_length):
            input_ids = pad(input_ids, model.seq_len, tokenizer.pad_token_id).to(device)
            outputs = model(input_ids.unsqueeze(0))
            next_token_logits = outputs[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze()

            generated.append(next_token.item())
            input_ids = torch.cat([input_ids[1:], next_token.view(-1)])

            if len(generated) > 5 and next_token.item() == tokenizer.sep_token_id:
                break

    return tokenizer.decode(generated)


def main():
    model_files = list(Path("models").glob("transformer_*.pt"))
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading model: {latest_model}")

    model, tokenizer, accelerator = load_model(latest_model)

    print("\nChat with the model")
    while True:
        prompt = input("\nYou: ")
        response = generate_text(model, tokenizer, accelerator, prompt)
        print(f"Model: {response}")


if __name__ == "__main__":
    main()
