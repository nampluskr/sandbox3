# tests/test_03_gan_model.py

import pytest
import torch
import torch.nn as nn

from src.models.gan import Generator, Discriminator, VanillaGAN


@pytest.fixture
def latent_dim():
    return 100


@pytest.fixture
def batch_size():
    return 8


@pytest.fixture
def img_size():
    return 32


@pytest.fixture
def out_channels():
    return 1


@pytest.fixture
def base():
    return 64


def test_generator_initialization(latent_dim, img_size, out_channels, base):
    """Generator 초기화 및 구조 검증"""
    gen = Generator(img_size=img_size, latent_dim=latent_dim, out_channels=out_channels, base=base)

    assert gen.latent_dim == latent_dim
    assert gen.img_size == img_size
    assert gen.out_channels == out_channels


def test_generator_forward_pass(batch_size, latent_dim, img_size, out_channels):
    """Generator 정방향 전파 및 출력 크기 검증"""
    gen = Generator(img_size=img_size, latent_dim=latent_dim, out_channels=out_channels)
    z = torch.randn(batch_size, latent_dim)

    with torch.no_grad():
        output = gen(z)

    assert output.shape == (batch_size, out_channels, img_size, img_size)
    assert output.min() >= -1.0
    assert output.max() <= 1.0


def test_generator_output_range(batch_size, latent_dim):
    """Generator 출력이 Tanh로 [-1, 1] 범위에 있는지 검증"""
    gen = Generator(img_size=32, latent_dim=latent_dim, out_channels=1)
    z = torch.randn(batch_size, latent_dim)

    with torch.no_grad():
        output = gen(z)

    assert torch.all(output >= -1.0) and torch.all(output <= 1.0)


def test_discriminator_initialization(img_size, out_channels, base):
    """Discriminator 초기화 검증"""
    disc = Discriminator(img_size=img_size, in_channels=out_channels, base=base)

    assert disc.img_size == img_size


def test_discriminator_forward_pass(batch_size, img_size, out_channels):
    """Discriminator 정방향 전파 및 출력 크기 검증"""
    disc = Discriminator(img_size=img_size, in_channels=out_channels)
    x = torch.randn(batch_size, out_channels, img_size, img_size)

    with torch.no_grad():
        output = disc(x)

    assert output.shape == (batch_size,)  # (B,)
    assert output.dtype == torch.float32


def test_discriminator_output_range(batch_size, img_size):
    """Discriminator 출력이 로짓 범위에 있는지 (이상치 없음)"""
    disc = Discriminator(img_size=img_size, in_channels=1)
    x = torch.randn(batch_size, 1, img_size, img_size)

    with torch.no_grad():
        output = disc(x)

    # 로짓이 과도하게 크지 않아야 함
    assert not torch.any(torch.isnan(output))
    assert not torch.any(torch.isinf(output))


# -------------------------
# VanillaGAN 테스트
# -------------------------

def test_vanillagan_initialization(latent_dim, img_size, out_channels):
    """VanillaGAN 초기화 및 장치 이동 검증"""
    gen = Generator(img_size=img_size, latent_dim=latent_dim, out_channels=out_channels)
    disc = Discriminator(img_size=img_size, in_channels=out_channels)

    model = VanillaGAN(generator=gen, discriminator=disc, latent_dim=latent_dim, device='cpu')

    assert model.device == torch.device('cpu')
    assert model.latent_dim == latent_dim
    assert model.global_epoch == 0


def test_vanillagan_forward(batch_size, latent_dim):
    """VanillaGAN에서 가짜 이미지 생성 테스트"""
    gen = Generator(img_size=32, latent_dim=latent_dim, out_channels=1)
    disc = Discriminator(img_size=32, in_channels=1)
    model = VanillaGAN(gen, disc, latent_dim=latent_dim, device='cpu')

    with torch.no_grad():
        fake_images = model(batch_size=batch_size)

    assert fake_images.shape == (batch_size, 1, 32, 32)
    assert fake_images.min() >= -1.0
    assert fake_images.max() <= 1.0


def test_vanillagan_train_step(batch_size, latent_dim):
    """train_step이 정상적으로 손실을 반환하는지 검증"""
    gen = Generator(img_size=32, latent_dim=latent_dim, out_channels=1)
    disc = Discriminator(img_size=32, in_channels=1)
    model = VanillaGAN(gen, disc, latent_dim=latent_dim, device='cpu')

    # 더미 배치 생성
    batch = {
        'image': torch.randn(batch_size, 1, 32, 32)
    }

    model.train()
    results = model.train_step(batch)

    assert 'g_loss' in results
    assert 'd_loss' in results
    assert 'batch_size' in results
    assert results['batch_size'] == batch_size
    assert isinstance(results['g_loss'], float)
    assert isinstance(results['d_loss'], float)


# def test_vanillagan_eval_step(batch_size, latent_dim):
#     """eval_step이 정상 작동하는지 검증"""
#     gen = Generator(img_size=32, latent_dim=latent_dim, out_channels=1)
#     disc = Discriminator(img_size=32, in_channels=1)
#     model = VanillaGAN(gen, disc, latent_dim=latent_dim, device='cpu')

#     batch = {
#         'image': torch.randn(batch_size, 1, 32, 32)
#     }

#     with torch.no_grad():
#         results = model.eval_step(batch)

#     assert 'g_loss' in results
#     assert 'd_loss' in results
#     assert results['batch_size'] == batch_size


# -------------------------
# 초기화 테스트
# -------------------------

def test_init_weights_conv2d():
    """init_weights가 Conv2d 가중치를 올바르게 초기화하는지"""
    conv = nn.Conv2d(3, 64, 4)
    old_weight = conv.weight.clone().detach()

    init_weights(conv)
    new_weight = conv.weight

    assert not torch.equal(old_weight, new_weight)
    assert torch.isclose(new_weight.mean(), torch.tensor(0.0), atol=0.03)


def test_init_weights_batchnorm():
    """init_weights가 BatchNorm2d를 올바르게 초기화하는지"""
    bn = nn.BatchNorm2d(64)
    old_weight = bn.weight.clone().detach()
    old_bias = bn.bias.clone().detach()

    init_weights(bn)

    assert not torch.equal(old_weight, bn.weight)
    assert not torch.equal(old_bias, bn.bias)
    assert torch.isclose(bn.weight.mean(), torch.tensor(1.0), atol=0.03)
    assert torch.isclose(bn.bias.mean(), torch.tensor(0.0), atol=1e-6)


def test_vanillagan_applies_init_weights(latent_dim):
    """VanillaGAN이 Generator와 Discriminator에 init_weights 적용하는지"""
    gen = Generator(img_size=32, latent_dim=latent_dim, out_channels=1)
    disc = Discriminator(img_size=32, in_channels=1)

    # 초기 상태 확인
    first_conv_weight_before = gen.initial[0].weight.clone().detach()

    # VanillaGAN 생성 → init_weights 적용
    model = VanillaGAN(gen, disc, latent_dim=latent_dim, device='cpu')

    first_conv_weight_after = model.generator.initial[0].weight

    # 가중치가 변경되었는지 확인
    assert not torch.equal(first_conv_weight_before, first_conv_weight_after)
    assert torch.isclose(first_conv_weight_after.mean(), torch.tensor(0.0), atol=0.03)
