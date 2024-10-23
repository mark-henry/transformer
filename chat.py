# chat.py
import torch
from transformers import AutoTokenizer
from accelerate import Accelerator
from pathlib import Path

# Import your model definition
from training import Transformer, pad, BertConfig

def load_model(model_path):
    accelerator = Accelerator()

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    config = BertConfig.from_pretrained('bert-base-uncased')

    model = Transformer(
        config.hidden_size,
        tokenizer.vocab_size,
        tokenizer.pad_token_id,
        seq_len=512
    )

    # Load the saved state
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Prepare model with accelerator
    model = accelerator.prepare(model)
    model.eval()

    return model, tokenizer, accelerator

def generate_text(model, tokenizer, accelerator, prompt, max_length=50, temperature=0.7):
    model.eval()

    # Encode the prompt and move to correct device
    input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
    input_ids = accelerator.prepare(input_ids)

    # Generate tokens one at a time
    generated = []
    with torch.no_grad():
        for _ in range(max_length):
            # Pad sequence to context size
            padded_sequence = pad(input_ids, model.seq_len)
            padded_sequence = accelerator.prepare(padded_sequence)

            # Get model predictions
            outputs = model(padded_sequence)

            # Get predictions for last token
            next_token_logits = outputs[-1, :] / temperature

            # Sample from the distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze()

            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.view(-1)])

            # Stop if we generate [SEP] token
            if next_token.item() == tokenizer.sep_token_id:
                break

    return tokenizer.decode(generated)

def main():
    # Load latest model from models directory
    model_files = list(Path("models").glob("transformer_*.pt"))
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading model: {latest_model}")

    model, tokenizer, accelerator = load_model(latest_model)

    print("Chat with the model (type 'quit' to exit)")
    while True:
        prompt = input("\nYou: ")
        if prompt.lower() == 'quit':
            break

        response = generate_text(model, tokenizer, accelerator, prompt)
        print(f"Model: {response}")

if __name__ == "__main__":
    main()