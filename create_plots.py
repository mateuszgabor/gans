import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch


def load_training_details(training_details_path):
    training_details_path = Path(training_details_path)

    if not training_details_path.exists():
        raise FileNotFoundError(f"File not found: {training_details_path}")

    with open(training_details_path, "r") as file:
        return torch.load(file)


def create_plots(train_details_path):
    path = Path(train_details_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    training_details = torch.load(path, weights_only=False)
    output_path = Path(f"{path.parent.parent}/plots")
    output_path.mkdir(parents=True, exist_ok=True)

    epochs = training_details["epoch"]
    d_losses = training_details["d_losses"]
    g_losses = training_details["g_losses"]
    d_x = training_details["d_x"]
    d_g_z1 = training_details["d_g_z1"]
    d_g_z2 = training_details["d_g_z2"]
    sns.set_theme(style="whitegrid")

    plt.figure()
    plt.plot(epochs, d_losses, "-o", label="Discriminator Loss")
    plt.plot(epochs, g_losses, "-o", label="Generator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Discriminator and Generator Losses Over Epochs")
    plt.legend()
    plt.savefig(f"{output_path}/losses_{path.stem}.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, d_x, "-o", label="D(x)")
    plt.plot(epochs, d_g_z1, "-o", label="D(G(z1))")
    plt.plot(epochs, d_g_z2, "-o", label="D(G(z2))")
    plt.xlabel("Epochs")
    plt.ylabel("Discriminator Output")
    plt.title("Discriminator Outputs Over Epochs")
    plt.legend()
    plt.savefig(f"{output_path}/disc_outputs_{path.stem}.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str, required=True)
    args = parser.parse_args()

    create_plots(args.path)
