import argparse
import json
from pathlib import Path

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA
from torchvision.utils import save_image

from utils import epoch, get_network, save_checkpoint

IMG_SIZE = 64
NORMALIZE = (0.5, 0.5, 0.5)
PATHS = {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-net", type=str, default="dcgan", required=False)
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

    net_name = args.net
    lr = args.lr
    num_epochs = args.epochs
    bs = args.bs
    b_1 = args.b_1
    b_2 = args.b_2
    nc = args.nc
    nf = args.nf
    nz = args.nz
    workers = args.workers

    p = Path(__file__)
    checkpoint_path = f"{p.parent}/checkpoints"
    generated_path = f"{p.parent}/generated/{net_name}"
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    Path(generated_path).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
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

    net_d, net_g = get_network(net_name, nc, nf, nz, device)
    print(net_d)
    print(net_g)
    opt_d = optim.AdamW(net_d.parameters(), lr=lr, betas=(b_1, b_2))
    opt_g = optim.AdamW(net_g.parameters(), lr=lr, betas=(b_1, b_2))
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    d_losses = []
    g_losses = []
    for i in range(1, num_epochs + 1):
        d_loss, g_loss, d_x, d_g_z1, d_g_z2 = epoch(
            train_loader,
            net_d,
            net_g,
            device,
            criterion,
            opt_d,
            opt_g,
            nz,
        )
        print(
            f"Epoch: {i} | D loss: {d_loss:.4f}, G loss: {g_loss:.4f} | "
            + f"D(x): {d_x:.4f}, D(G(z)): {d_g_z1:.4f} / {d_g_z2:.4f}"
        )
        d_losses.append(d_loss)
        g_losses.append(g_loss)

        with torch.no_grad():
            fake = net_g(fixed_noise)
            save_image(fake, f"{generated_path}/{i}.png", normalize=True)

    with open(f"{checkpoint_path}/{net_name}_d_losses.json", "w") as f:
        json.dump(d_losses, f)

    with open(f"{checkpoint_path}/{net_name}_g_losses.json", "w") as f:
        json.dump(g_losses, f)

    save_checkpoint(
        {
            "epoch": i,
            "net_d": net_d.state_dict(),
            "net_g": net_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "opt_g": opt_g.state_dict(),
        },
        f"{checkpoint_path}/{net_name}.pth.tar",
    )


if __name__ == "__main__":
    main()
