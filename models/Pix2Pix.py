import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.image.fid import FrechetInceptionDistance

from models.Generator import GeneratorUNet
from models.Discriminator import Discriminator


class Pix2Pix(pl.LightningModule):
    def __init__(self,
                 batch_size: int,
                 patch,
                 lambda_pixel: int = 100,
                 lr: float = 2e-4,
                 beta1: float = 0.5,
                 beta2: float = 0.999,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.patch = patch
        self.lambda_pixel = lambda_pixel
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.generator = GeneratorUNet()
        self.discriminator = Discriminator()

        self.metric = FrechetInceptionDistance(feature=64)

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx, optimizer_idx):
        a, b = batch
        ba_si = a.size(0)

        # patch label
        real_label = torch.ones(ba_si, *self.patch, requires_grad=False)
        fake_label = torch.zeros(ba_si, *self.patch, requires_grad=False)

        fake_b = self.generator(a)

        # generator
        if optimizer_idx == 0:
            out_dis = self.discriminator(fake_b, b).detach().cpu()  # 가짜 이미지 식별

            gen_loss = F.binary_cross_entropy_with_logits(out_dis, real_label)
            pixel_loss = F.l1_loss(fake_b, b)

            g_loss = gen_loss + self.lambda_pixel * pixel_loss

            # fid calc
            self.metric.update(b.type(torch.uint8), real=True)
            self.metric.update(fake_b.type(torch.uint8), real=False)
            fid = self.metric.compute()

            self.log("fid", fid, prog_bar=True)
            self.log("g_loss", g_loss, prog_bar=True)
            return g_loss.requires_grad_(True)

        # discriminator
        if optimizer_idx == 1:
            out_dis = self.discriminator(b, a).detach().cpu()  # 진짜 이미지 식별
            real_loss = F.binary_cross_entropy_with_logits(out_dis, real_label)

            out_dis = self.discriminator(fake_b.detach(), a).detach().cpu()  # 가짜 이미지 식별
            fake_loss = F.binary_cross_entropy_with_logits(out_dis, fake_label)

            d_loss = (real_loss + fake_loss) / 2.
            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss.requires_grad_(True)

    def validation_step(self, batch, batch_idx):
        a, _ = batch
        fake_b = self.generator(a)
        return fake_b

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(),
                                 lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(),
                                 lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2))

        scheduler_g = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_g, 10, 2, 0.0001, -1)
        scheduler_d = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_d, 10, 2, 0.0001, -1)

        return (
            {"optimizer": opt_g, "lr_scheduler": scheduler_g, "monitor": "g_loss"},
            {"optimizer": opt_d, "lr_scheduler": scheduler_d, "monitor": "d_loss"}
        )
