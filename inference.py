import argparse
from datasets import load_dataset
import torch

from src.generation import generate_from_series
from src.model import ForecastingModel, ModelConfig
from src.visualisation.chart import plot_series


def parse_args():
    parser = argparse.ArgumentParser(description='Inference for a Time Series Forecasting Model.')

    parser.add_argument('--model_path', type=str, default='checkpoint.pth', help='Path to the trained model.')
    parser.add_argument('--dataset', type=str, default='dataset.jsonl', help='Path to the dataset.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of transformer layers.')
    parser.add_argument('--n_head', type=int, default=2, help='Number of attention heads.')
    parser.add_argument('--n_embd', type=int, default=384, help='Embedding dimension (default: n_head * 64).')
    parser.add_argument('--vocab_size', type=int, default=1024, help='Vocabulary size.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate.')
    parser.add_argument('--block_size', type=int, default=256, help='Context window size.')

    return parser.parse_args()


def load_model(arguments, device: str = 'mps'):
    """
    Load a model that has been saved using torch.save(self.state_dict(), weights_path)
    """
    config = ModelConfig(
        block_size=arguments.block_size,
        n_layer=arguments.n_layers,
        n_head=arguments.n_head,
        n_embd=arguments.n_embd,
        vocab_size=arguments.vocab_size,
        dropout=arguments.dropout,
        ffn_hidden_size=arguments.n_embd * 4
    )

    model = ForecastingModel(config)
    model.load_state_dict(torch.load(arguments.model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()

    return model


if __name__ == '__main__':
    args = parse_args()

    model = load_model(args, device='mps' if torch.backends.mps.is_available() else 'cpu')

    series = list(load_dataset(
        "json",
        data_files=args.dataset,
        split="train",
        streaming=True,
    ).shuffle(seed=args.seed).take(1))[0]['target'][0]

    generation = generate_from_series(
        series=series,
        model=model,
        context_length=30,
        low_limit=-1000,
        high_limit=1000,
        num_bins=1024,
        pad_token_id=1024,
        max_tokens=50,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        log_file=None,
    )

    plot_series(
        y_init=generation[-512:-256],
        y_forecast=generation[-256:],
        show_figure=True
    )
