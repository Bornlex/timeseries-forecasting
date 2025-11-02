import argparse
from dataclasses import dataclass
import os
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb

from src.config import ModelConfig
from src.generation import generate_from_series
from src.model import ForecastingModel
from src.tokenizer import setup_data, tokenized_window_from_series
from src.visualisation.chart import plot_series


@dataclass
class TrainingConfig:
    lr: float
    lr_decay_iters: int
    min_lr: float
    max_iters: int
    beta1: float
    beta2: float
    eval_interval: int
    eval_iters: int
    log_interval: int
    weight_decay: float


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Time Series Forecasting Model.')

    parser.add_argument('--save', action='store_true', help='Flag to save the trained model.')
    parser.add_argument('--resume', action='store_true', help='Whether to resume training from checkpoint.')
    parser.add_argument('--save_path', type=str, default='model.pth', help='Path to save the trained model.')
    parser.add_argument('--resume_path', type=str, default='checkpoint.pth', help='Path to save the checkpointed model.')

    # --- Model Hyperparameters ---
    parser.add_argument('--n_layers', type=int, default=2, help='Number of transformer layers.')
    parser.add_argument('--n_head', type=int, default=2, help='Number of attention heads.')
    parser.add_argument('--n_embd', type=int, default=384, help='Embedding dimension (default: n_head * 64).')
    parser.add_argument('--vocab_size', type=int, default=1024, help='Vocabulary size.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate.')
    parser.add_argument('--low_limit', type=float, default=-15.0, help='Low limit for scaling.')
    parser.add_argument('--high_limit', type=float, default=15.0, help='High limit for scaling.')

    # --- Data & Training ---
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--block_size', type=int, default=256, help='Context window size.')
    parser.add_argument('--dataset', type=str, default='shakespeare', help='Name of the dataset to use.')
    parser.add_argument('--max_iters', type=int, default=5000, help='Total training iterations.')
    parser.add_argument('--num_series', type=int, default=2000, help='Number of series to train on.')

    # --- Optimizer & Scheduler ---
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate.')
    parser.add_argument('--lr_decay_iters', type=int, default=5000, help='Iterations for learning rate decay.')
    parser.add_argument('--min_lr', type=float, default=1e-4, help='Minimum learning rate (default: learning_rate / 10).')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam optimizer beta1.')
    parser.add_argument('--beta2', type=float, default=0.99, help='Adam optimizer beta2.')
    parser.add_argument('--weight_decay', type=float, default=1e-1, help='Weight decay.')

    # --- Logging & Evaluation ---
    parser.add_argument('--eval_interval', type=int, default=250, help='Iterations between evaluations.')
    parser.add_argument('--eval_iters', type=int, default=200, help='Number of iterations for evaluation.')
    parser.add_argument('--log_interval', type=int, default=10, help='Iterations between logging.')

    # --- W&B Logging ---
    parser.add_argument('--wandb_log', action='store_true', help='Flag to enable Weights & Biases logging.')
    parser.add_argument('--wandb_project', type=str, default='time-series-forecasting', help='Name of the Weights & Biases project.')

    args = parser.parse_args()

    return args


def train(
        forecasting_model: nn.Module,
        dataset: DataLoader,
        training_config: TrainingConfig,
        batch_size: int,
        block_size: int,
        low_limit: float = -15.0,
        high_limit: float = 15.0,
        num_bins: int = 1023,
        pad_token_id: int = 1023,
        checkpoint_path: str = None,
        validation_series_list: list = None
):
    device = 'mps' if torch.mps.is_available() else 'cpu'

    optimizer = torch.optim.AdamW(
        forecasting_model.parameters(),
        lr=training_config.lr,
        betas=(training_config.beta1, training_config.beta2),
        weight_decay=training_config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config.lr_decay_iters,
        eta_min=training_config.min_lr
    )
    loss_fn = nn.CrossEntropyLoss()

    forecasting_model.to(device)
    forecasting_model.train()

    for iteration in range(training_config.max_iters):
        x, y = next(iter(dataset))
        x = x.to(device)
        y = y.to(device, dtype=torch.long)

        prediction, _ = forecasting_model(x)
        b, n, c = prediction.shape

        loss = loss_fn(prediction.view(b * n, c), y.view(b * n))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(forecasting_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        wandb.log({"train/loss": loss.item(), "iteration": iteration, "lr": optimizer.param_groups[0]['lr']})

        if iteration % 10 == 0:
            print(f'[{iteration}|{training_config.max_iters}] loss : {loss.item():.4f}')

            if checkpoint_path is not None:
                forecasting_model.save_weights(checkpoint_path)

        if iteration % 100 == 0 and validation_series_list:
            forecasting_model.eval()

            with torch.no_grad():
                x_eval, y_eval = next(iter(dataset))
                x_eval = x_eval.to(device)

                generation = generate_from_series(
                    series=random.choice(validation_series_list),
                    model=forecasting_model,
                    context_length=block_size,
                    low_limit=low_limit,
                    high_limit=high_limit,
                    num_bins=num_bins,
                    pad_token_id=pad_token_id,
                )

                plot_series(
                    y_init=x_eval[0].cpu().numpy(),
                    y_forecast=generation,
                )

            forecasting_model.train()

    return forecasting_model


if __name__ == '__main__':
    arguments = parse_args()

    conf = TrainingConfig(
        lr=arguments.learning_rate,
        lr_decay_iters=arguments.lr_decay_iters,
        min_lr=arguments.min_lr,
        max_iters=arguments.max_iters,
        beta1=arguments.beta1,
        beta2=arguments.beta2,
        eval_interval=arguments.eval_interval,
        eval_iters=arguments.eval_iters,
        log_interval=arguments.log_interval,
        weight_decay=arguments.weight_decay
    )
    model_config = ModelConfig(
        block_size=arguments.block_size,
        n_layer=arguments.n_layers,
        n_head=arguments.n_head,
        n_embd=arguments.n_embd,
        vocab_size=arguments.vocab_size,
        dropout=arguments.dropout,
        ffn_hidden_size=arguments.n_embd * 4
    )
    model = ForecastingModel(model_config)

    if arguments.resume and os.path.isfile(arguments.resume_path):
        model.load_state_dict(torch.load(arguments.resume_path))

    run = wandb.init(
        project=arguments.wandb_project,
        name=f'l{arguments.n_layers}-h{arguments.n_head}-d{arguments.n_embd}',
        config={
            "learning_rate": conf.lr,
        },
    )

    train_loader, validation_list = setup_data(
        num_series=arguments.num_series,
        context_length=arguments.block_size,
        horizon=1,
        num_bins=arguments.vocab_size - 1,
        batch_size=arguments.batch_size
    )

    model = train(
        model,
        train_loader,
        conf,
        arguments.batch_size,
        arguments.block_size,
        checkpoint_path=arguments.resume_path,
        low_limit=arguments.low_limit,
        high_limit=arguments.high_limit,
        num_bins=arguments.vocab_size - 1,
        pad_token_id=arguments.vocab_size - 1,
        validation_series_list=validation_list
    )

    if arguments.save:
        model.save_weights(arguments.save_path)
