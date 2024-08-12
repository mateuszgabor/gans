import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.inception import inception_v3
from tqdm import tqdm

from models import Generator


def load_generator_from_checkpoint(checkpoint_path, nz, nc, nf, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    net_g = Generator(nz, nc, nf).to(device)

    try:
        net_g.load_state_dict(checkpoint["net_g"])
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        print("Check the model architecture and the parameters used during training.")
        raise e

    return net_g


def generate_images(generator, nz, num_images, batch_size, device):
    generator.eval()
    generated_images = []
    noise = torch.randn(num_images, nz, 1, 1, device=device)

    with torch.no_grad():
        for i in range(0, num_images, batch_size):
            batch_noise = noise[i : i + batch_size]
            fake_images = generator(batch_noise)
            generated_images.append(fake_images)

    generated_images = torch.cat(generated_images, dim=0)
    return generated_images


def inception_score(images, device, batch_size=32, splits=10):
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    transform_resize = transforms.Resize((299, 299))
    preds = []

    for i in tqdm(range(0, len(images), batch_size), desc="Computing Inception Score"):
        batch = images[i : i + batch_size]
        batch = transform_resize(batch)
        with torch.no_grad():
            batch_preds = inception_model(batch)

            if isinstance(batch_preds, tuple):
                batch_preds = batch_preds[0]

            preds.append(F.softmax(batch_preds, dim=1).cpu())

    preds = torch.cat(preds, dim=0)
    split_scores = []

    for k in range(splits):
        part = preds[
            k * (preds.size(0) // splits) : (k + 1) * (preds.size(0) // splits), :
        ]
        py = torch.mean(part, dim=0)
        scores = []
        for i in range(part.size(0)):
            pyx = part[i, :]
            scores.append(F.kl_div(pyx.log(), py, reduction="batchmean").exp().item())
        split_scores.append(torch.mean(torch.tensor(scores)).item())

    return torch.mean(torch.tensor(split_scores)), torch.std(torch.tensor(split_scores))


def main():
    checkpoint_path = "checkpoints/dcgan.pth.tar"
    num_images = 30000
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nz = 100
    nc = 3
    nf = 64

    net_g = load_generator_from_checkpoint(checkpoint_path, nz, nc, nf, device)

    generated_images = generate_images(net_g, nz, num_images, batch_size, device)

    mean_is, std_is = inception_score(generated_images, device, batch_size)
    print(f"Inception Score: {mean_is:.4f} ± {std_is:.4f}")


if __name__ == "__main__":
    main()
