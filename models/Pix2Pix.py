import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.autograd import Variable
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from models.Generator import GeneratorUNet
from models.Discriminator import Discriminator


class Pix2Pix(pl.LightningModule):
    def __init__(self,
                 batch_size: int,
                 patch,
                 size=256,
                 cuda=True,
                 half=True,
                 lambda_pixel: int = 100,
                 lr: float = 2e-4,
                 beta1: float = 0.5,
                 beta2: float = 0.999,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.patch = patch
        self.size = size
        self.lambda_pixel = lambda_pixel
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        # generator & discriminator
        self.generator = GeneratorUNet()
        self.discriminator = Discriminator()

        # metric: fid
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()

        # Inputs & targets memory allocation
        self.ttype = torch.Tensor
        if cuda:
            if half:
                self.ttype = torch.cuda.HalfTensor
            else:
                self.ttype = torch.cuda.FloatTensor
        else:
            self.ttype = torch.Tensor

        self.input_A = self.ttype(batch_size, 3, size, size)
        self.input_B = self.ttype(batch_size, 3, size, size)
        self.target_real = torch.ones(batch_size, *self.patch, requires_grad=False).type(self.ttype)
        self.target_fake = torch.zeros(batch_size, *self.patch, requires_grad=False).type(self.ttype)

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx, optimizer_idx):
        a, b = batch
        cur_batch_size = a.size(0)

        self.input_A = self.ttype(cur_batch_size, 3, self.size, self.size)
        self.input_B = self.ttype(cur_batch_size, 3, self.size, self.size)
        real_a = Variable(self.input_A.copy_(a)).type(self.ttype)
        real_b = Variable(self.input_B.copy_(b)).type(self.ttype)

        fake_b = self.generator(real_a)

        # patch label
        self.target_real = torch.ones(cur_batch_size, *self.patch, requires_grad=False).type(self.ttype)
        self.target_fake = torch.zeros(cur_batch_size, *self.patch, requires_grad=False).type(self.ttype)

        # generator
        if optimizer_idx == 0:
            out_dis = self.discriminator(fake_b, real_b)  # 가짜 이미지 식별

            # print(out_dis.dtype)
            # print(self.target_real.dtype)

            gen_loss = self.bce(out_dis, self.target_real)
            pixel_loss = self.l1(fake_b, real_b)

            g_loss = gen_loss + self.lambda_pixel * pixel_loss

            # fid calc
            lpips = self.lpips(real_b, fake_b)

            self.log("lpips", lpips, prog_bar=True)
            self.log("g_loss", g_loss, prog_bar=True)
            return g_loss.requires_grad_(True)

        # discriminator
        if optimizer_idx == 1:
            out_dis = self.discriminator(real_b, real_a)  # 진짜 이미지 식별
            real_loss = self.bce(out_dis, self.target_real)

            out_dis = self.discriminator(fake_b, real_a)  # 가짜 이미지 식별
            fake_loss = self.bce(out_dis, self.target_fake)

            d_loss = (real_loss + fake_loss) / 2.
            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss.requires_grad_(True)

    def validation_step(self, batch, batch_idx):
        a, _ = batch
        cur_batch_size = a.size(0)

        self.input_A = self.ttype(cur_batch_size, 3, self.size, self.size)
        self.input_B = self.ttype(cur_batch_size, 3, self.size, self.size)
        real_a = Variable(self.input_A.copy_(a)).type(self.ttype)

        fake_b = self.generator(real_a)
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
