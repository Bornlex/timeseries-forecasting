import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Optional, Sequence, Tuple


def simple_plot_series(y: Sequence[float]) -> None:
    sns.set_theme(style="whitegrid")

    y = np.asarray(y)
    x = np.arange(len(y))

    plt.figure()
    plt.plot(x, y, label="series", color='blue', linewidth=2)
    plt.grid(True)
    plt.legend()
    plt.show()


def simple_subplot(series_list: Sequence[Sequence[float]], labels: Sequence[str]) -> None:
    sns.set_theme(style="whitegrid")

    num_series = len(series_list)
    if num_series == 0:
        return

    ncols = int(np.ceil(np.sqrt(num_series)))
    nrows = int(np.ceil(num_series / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    for i, (y, label) in enumerate(zip(series_list, labels)):
        ax = axes_flat[i]
        y = np.asarray(y)
        x = np.arange(len(y))

        ax.plot(x, y, label=label, color='blue', linewidth=2)
        ax.grid(True)
        ax.legend()

    for ax in axes_flat[num_series:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_series(
        y_init: Sequence[float],
        y_forecast: Sequence[float],
        filename: Optional[str] = None,
        x_init: Optional[Sequence[float]] = None,
        x_forecast: Optional[Sequence[float]] = None,
        forecast_start: Optional[int] = None,
        show_figure: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    sns.set_theme(style="whitegrid")

    y_init = np.asarray(y_init)
    y_forecast = np.asarray(y_forecast)

    if x_init is None:
        x_init = np.arange(len(y_init))
    else:
        x_init = np.asarray(x_init)

    if x_forecast is None:
        if forecast_start is None:
            forecast_start = len(x_init)
        x_forecast = np.arange(forecast_start, forecast_start + len(y_forecast))
    else:
        x_forecast = np.asarray(x_forecast)

    fig, ax = plt.subplots()

    ax.plot(x_init, y_init, label="observed", color='blue', linewidth=2)
    ax.plot(x_forecast, y_forecast, label="forecast", color='red', linestyle='--', linewidth=2)

    ax.grid(True)
    ax.legend()

    if filename:
        fig.savefig(filename, bbox_inches="tight")

    if show_figure:
        plt.show()

    return fig, ax
