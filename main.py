import argparse
from pathlib import Path

import torch
import torch.optim as optim
import torchvision.utils as vutils
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA

from models import Discriminator, Generator
from utils import epoch

IMG_SIZE = 64
NORMALIZE = (0.5, 0.5, 0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", type=float, default=2e-4, required=False)
    parser.add_argument("-epochs", type=int, default=30, required=False)
    parser.add_argument("-bs", type=int, default=64, required=False)
    parser.add_argument("-b_1", type=float, default=0.5, required=False)
    parser.add_argument("-b_2", type=float, default=0.999, required=False)
    parser.add_argument("-nc", type=int, default=3, required=False)
    parser.add_argument("-nf", type=int, default=64, required=False)
    parser.add_argument("-nz", type=int, default=100, required=False)
    parser.add_argument("-workers", type=int, default=4, required=False)
    args = parser.parse_args()

    lr = args.lr
    num_epochs = args.epochs
    bs = args.bs
    b_1 = args.b_1
    b_2 = args.b_2
    nc = args.nc
    nf = args.nf
    nz = args.nc
    workers = args.workers

    p = Path(__file__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.CenterCrop(IMG_SIZE),
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE, NORMALIZE),
        ]
    )

    train_dataset = CelebA(".", split="train", download=True, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )

    net_d = Discriminator(nc, nf).to(device)
    net_g = Generator(nz, nc, nf).to(device)
    opt_d = optim.Adam(net_d.parameters(), lr=lr, betas=(b_1, b_2))
    opt_g = optim.Adam(net_g.parameters(), lr=lr, betas=(b_1, b_2))
    criterion = nn.BCELoss()

    real_label = 1
    fake_label = 0

    for i in range(1, num_epochs + 1):
        d_loss, g_loss, real_probs, fake_probs = epoch(
            train_loader,
            net_d,
            net_g,
            device,
            criterion,
            opt_d,
            opt_g,
            real_label,
            fake_label,
            bs,
            nz,
        )
        print(
            f"Epoch: {i} | D loss: {d_loss:.4f}, G loss: {g_loss:.4f} | "
            + f"Real probs: {real_probs:.4f}, Fake probs: {fake_probs:.4f}"
        )


if __name__ == "__main__":
    main()
