# experiments/mnist/01_mnist_gan.py

import os
import sys

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from src.config import load_config, merge_configs
from src.utils import set_seed, plot_images, create_noises, make_sample_path
from src.datasets.mnist import load_mnist
from src.models.gan import Generator, Discriminator, VanillaGAN
from src.training.trainer import train


def main():
    config = load_config("configs/mnist.yaml")
    config = merge_configs(config, load_config("configs/gan.yaml"))

    seed = config['seed']
    set_seed(seed)

    #################################################################
    print("\n>> Loading dataset:")
    #################################################################
    data_dir = config['path']['data_dir']
    output_dir = config['path']['output_dir']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']

    train_loader = load_mnist(
        data_dir=data_dir,
        split='train',
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    #################################################################
    print("\n>> Building model:")
    #################################################################
    img_size = config['data']['img_size']
    in_channels = config['data']['in_channels']
    out_channels = config['data']['out_channels']
    latent_dim = config['model']['latent_dim']

    generator = Generator(img_size=img_size, latent_dim=latent_dim, out_channels=out_channels)
    discriminator = Discriminator(img_size=img_size, in_channels=in_channels)
    gan = VanillaGAN(generator, discriminator)

    #################################################################
    print("\n>> Training model:")
    #################################################################
    max_epochs = config['train']['max_epochs']
    num_samples = config['train']['num_samples']
    sample_interval = config['train']['sample_interval']

    noises = create_noises(num_samples, latent_dim)
    for epoch in range(1, max_epochs + 1):
        train_results = train(gan, train_loader)
        print(f"[{epoch:>2}/{max_epochs}] {train_results['info']}")

        if epoch % sample_interval == 0:
            # images, labels = gan.predict(noises, return_labels=True)
            # labels = [label >=0.5 for label in labels]
            # sample_path = make_sample_path(output_dir, __file__, epoch)
            # plot_images(*images, titles=labels, save_path=sample_path, ncols=10)

            images = gan.predict(noises)
            sample_path = make_sample_path(output_dir, __file__, epoch)
            plot_images(*images, save_path=sample_path, ncols=10)



if __name__ == "__main__":
    main()
