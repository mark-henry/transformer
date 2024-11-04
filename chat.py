import torch
from transformers import AutoTokenizer
from accelerate import Accelerator
from pathlib import Path
import math
import argparse
from transformer import Transformer


def pad_left(token_ids, length, pad_token_id):
    if len(token_ids) >= length:
        return token_ids[-length:]

    padding_length = length - len(token_ids)
    padding = torch.full([padding_length], pad_token_id, dtype=torch.long, device=token_ids.device)
    return torch.cat([padding, token_ids])


def load_model(model_path):
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained('GPT2')
    tokenizer.pad_token = tokenizer.eos_token
    # Load checkpoint and configuration
    checkpoint = torch.load(model_path, weights_only=True, map_location=accelerator.device)
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
    device = next(model.parameters()).device

    # Initialize sequence with prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')[0].to(device)
    generated = []

    # Ensure we have exactly seq_len tokens, padded on the left if necessary
    input_ids = pad_left(input_ids, model.seq_len, tokenizer.pad_token_id)

    with torch.no_grad():
        for _ in range(max_length):
            # Get model prediction
            outputs = model(input_ids.unsqueeze(0))  # Shape: (1, seq_len, vocab_size)
            next_token_logits = outputs[0, -1, :] / temperature  # Shape: (vocab_size,)

            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze()

            # Add to generated sequence
            generated.append(next_token.item())

            # Prepare input for next iteration by appending new token
            input_ids = torch.cat([input_ids[1:], next_token.view(-1)])

            # Optional: Break if we generate an end token
            if len(generated) > 5 and next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser(description='Chat with a transformer model')
    parser.add_argument('--model_path', type=str,
                        help='Path to the model checkpoint file')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for text generation (default: 0.1)')
    parser.add_argument('--max_length', type=int, default=50,
                        help='Maximum length of generated text (default: 50)')

    args = parser.parse_args()

    # If no model path is provided, use the latest model in the models directory
    if args.model_path is None:
        model_files = list(Path("models").glob("transformer_*.pt"))
        if not model_files:
            print("Error: No model files found in the models directory")
            return
        model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        print(f"No model path provided. Using latest model: {model_path}")
    else:
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"Error: Model file not found at {model_path}")
            return

    model, tokenizer, accelerator = load_model(model_path)
    print("\nChat with the model (Ctrl+C to exit)")

    try:
        while True:
            prompt = input("\nYou: ")
            response = generate_text(
                model,
                tokenizer,
                accelerator,
                prompt,
                max_length=args.max_length,
                temperature=args.temperature
            )
            print(f"Model: {response}")
    except KeyboardInterrupt:
        print("\nExiting chat...")


if __name__ == "__main__":
    main()
