import torch
from tqdm import tqdm

from models import (
    Discriminator,
    Generator,
    SaDiscriminator,
    SaGenerator,
    SpectralDiscriminator,
)

REAL_LABEL = 1
FAKE_LABEL = 0


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TrainingLogger:
    def __init__(self):
        self.details = {
            "epoch": [],
            "d_losses": [],
            "g_losses": [],
            "d_x": [],
            "d_g_z1": [],
            "d_g_z2": [],
            "times": [],
        }

    def log_epoch(self, epoch_num, d_loss, g_loss, d_x, d_g_z1, d_g_z2, time):
        self.details["epoch"].append(epoch_num)
        self.details["d_losses"].append(d_loss)
        self.details["g_losses"].append(g_loss)
        self.details["d_x"].append(d_x)
        self.details["d_g_z1"].append(d_g_z1)
        self.details["d_g_z2"].append(d_g_z2)
        self.details["times"].append(time)

    def save(self, filepath):
        torch.save(self.details, filepath)


def get_network(net, nc, nf, nz, device):
    match net:
        case "dcgan":
            net_d = Discriminator(nc, nf)
            net_g = Generator(nz, nc, nf)
        case "spectral_dcgan":
            net_d = SpectralDiscriminator(nc, nf)
            net_g = Generator(nz, nc, nf)
        case "sagan":
            net_d = SaDiscriminator(nc, nf)
            net_g = SaGenerator(nz, nc, nf)
        case _:
            raise NotImplementedError(
                f"Network of name '{net}' currently is not implemented"
            )

    net_d = net_d.to(device)
    net_g = net_g.to(device)
    return net_d, net_g


def epoch(
    loader,
    net_d,
    net_g,
    device,
    criterion,
    opt_d,
    opt_g,
    nz,
):
    losses_d = AverageMeter()
    losses_g = AverageMeter()
    d_x_meter = AverageMeter()
    d_g_z1_meter = AverageMeter()
    d_g_z2_meter = AverageMeter()

    for real_imgs, _ in tqdm(loader, leave=False):
        b_size = real_imgs.size(0)
        net_d.zero_grad(set_to_none=True)
        real_imgs = real_imgs.to(device)
        labels = torch.full((b_size,), REAL_LABEL, dtype=torch.float, device=device)
        outputs = net_d(real_imgs).view(-1)
        loss_d_real = criterion(outputs, labels)
        loss_d_real.backward()
        d_x = outputs.mean()

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake_imgs = net_g(noise)
        labels.fill_(FAKE_LABEL)

        outputs = net_d(fake_imgs.detach()).view(-1)
        loss_d_fake = criterion(outputs, labels)
        loss_d_fake.backward()
        d_g_z1 = outputs.mean()
        loss_d = loss_d_real + loss_d_fake
        opt_d.step()

        net_g.zero_grad(set_to_none=True)
        labels.fill_(REAL_LABEL)
        outputs = net_d(fake_imgs).view(-1)
        loss_g = criterion(outputs, labels)
        loss_g.backward()
        d_g_z2 = outputs.mean()
        opt_g.step()

        losses_d.update(loss_d.item(), b_size)
        losses_g.update(loss_g.item(), b_size)
        d_x_meter.update(d_x.item(), b_size)
        d_g_z1_meter.update(d_g_z1.item(), b_size)
        d_g_z2_meter.update(d_g_z2.item(), b_size)

    return (
        losses_d.avg,
        losses_g.avg,
        d_x_meter.avg,
        d_g_z1_meter.avg,
        d_g_z2_meter.avg,
    )
