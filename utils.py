import torch
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def epoch(
    loader, net_d, net_g, device, criterion, opt_d, opt_g, r_label, f_label, bs, nz
):
    losses_d = AverageMeter()
    losses_g = AverageMeter()
    real_probs = AverageMeter()
    fake_probs = AverageMeter()

    for real_imgs, _ in tqdm(loader, leave=False):
        net_d.zero_grad(set_to_none=True)
        real_imgs = real_imgs.to(device)
        labels = torch.full((bs,), r_label, dtype=torch.float, device=device)
        outputs = net_d(real_imgs).view(-1)
        loss_d_real = criterion(outputs, labels)
        loss_d_real.backward()
        d_x = outputs.mean().item()

        noise = torch.randn(bs, nz, 1, 1, device=device)
        fake_imgs = net_g(noise)
        labels.fill_(f_label)

        outputs = net_d(fake_imgs.detach()).view(-1)
        loss_d_fake = criterion(outputs, labels)
        loss_d_fake.backward()
        d_g_z1 = outputs.mean().item()
        loss_d = loss_d_real + loss_d_fake
        opt_d.step()

        net_g.zero_grad(set_to_none=True)
        labels.fill_(r_label)
        outputs = net_d(fake_imgs).view(-1)
        loss_g = criterion(outputs, labels)
        loss_g.backward()
        d_g_z2 = outputs.mean().item()
        opt_g.step()

        losses_d.update(loss_d.item(), bs)
        losses_g.update(loss_g.item(), bs)
        real_probs.update(d_x, bs)
        fake_probs.update(d_g_z1 / d_g_z2, bs)

    return losses_d.avg, losses_g.avg, real_probs.avg, fake_probs.avg
