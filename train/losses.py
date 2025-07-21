import torch
import torch.nn as nn

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            # print("last_layer.requires_grad: ", last_layer.requires_grad)
            # print("nll_loss grad_fn: ", nll_loss.grad_fn)
            # print("g_loss grad_fn: ", g_loss.grad_fn)
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
            # print("nll_grads:", nll_grads)
            # print("g_grads:", g_grads)
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        # rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        rec_loss = torch.abs(inputs - reconstructions)
        if self.perceptual_weight > 0:
            # p_loss = self.perceptual_loss(inputs[:,:3].contiguous(), reconstructions[:,:3].contiguous())
            # p_loss = self.perceptual_loss(inputs, reconstructions)
            p_loss = self.perceptual_loss(inputs[:,:3], reconstructions[:,:3])
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if global_step >= self.discriminator_iter_start:
                if cond is None:
                    assert not self.disc_conditional
                    # logits_fake = self.discriminator(reconstructions.contiguous())
                    logits_fake = self.discriminator(reconstructions)
                else:
                    assert self.disc_conditional
                    # logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
                    logits_fake = self.discriminator(torch.cat((reconstructions, cond), dim=1))
                g_loss = -torch.mean(logits_fake)

                if self.disc_factor > 0.0:
                    try:
                        # d_weight = torch.tensor(1.0)
                        d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                    except RuntimeError:
                        assert not self.training
                        d_weight = torch.tensor(0.0)
                else:
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)
                g_loss = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss
            # loss = weighted_nll_loss + self.kl_weight * kl_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.clone().detach().mean(), "{}/nll_loss".format(split): nll_loss.clone().detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.clone().detach().mean(),
                   "{}/d_weight".format(split): d_weight.clone().detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.clone().detach().mean(),
                   }
            # log = {}
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

class RGBAVAELoss(nn.Module):
    Eb: torch.Tensor
    Eb2: torch.Tensor
    def __init__(
        self, 
        custom_Eb=None, custom_Eb2=None, reduce_mean=False,
        use_naive_mse=False, use_patchgan=False, use_lpips=False,
    ):
        super().__init__()
        if custom_Eb is None:
            custom_Eb = [-0.0357, -0.0811, -0.1797]
        if custom_Eb2 is None:
            custom_Eb2 = [0.3163, 0.3060, 0.3634]
        self.register_buffer("Eb", torch.tensor(custom_Eb).reshape(1, 3, 1, 1), persistent=False)
        self.register_buffer("Eb2", torch.tensor(custom_Eb2).reshape(1, 3, 1, 1), persistent=False)
        self.reduce_mean = reduce_mean
        self.use_naive_mse = use_naive_mse
        self.use_patchgan = use_patchgan
        self.use_lpips = use_lpips
        if self.use_lpips:
            self.lpips = LPIPS().eval()
        if self.use_patchgan:
            self.discriminator = NLayerDiscriminator(
                input_nc=4,
                n_layers=3,
                use_actnorm=False
            ).apply(weights_init)

    def reduce_loss(self, value):
        if self.reduce_mean:
            return torch.mean(value)
        else:
            return torch.sum(value) / value.shape[0]

    def reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor):
        if self.use_naive_mse:
            return self.reduce_loss((pred - target) ** 2)
        target_rgb = target[:, :3]
        target_a = (target[:, 3:] + 1) / 2
        pred_rgb = pred[:, :3]
        pred_a = (pred[:, 3:] + 1) / 2
        rgba_diff = target_rgb * target_a - pred_rgb * pred_a
        alpha_diff = target_a - pred_a
        # print(rgba_diff.shape, alpha_diff.shape, self.Eb.shape, self.Eb2.shape)
        loss = rgba_diff ** 2 - 2 * self.Eb * rgba_diff * alpha_diff + self.Eb2 * alpha_diff ** 2
        return self.reduce_loss(loss)

    def perceptual_loss(self, pred: torch.Tensor, target: torch.Tensor):
        target_rgb = target[:, :3]
        target_a = (target[:, 3:] + 1) / 2
        pred_rgb = pred[:, :3]
        pred_a = (pred[:, 3:] + 1) / 2
        loss_black = self.lpips(target_rgb * target_a, pred_rgb * pred_a)
        loss_white = self.lpips(target_rgb * target_a + (1 - target_a), pred_rgb * pred_a + (1 - pred_a))
        loss = (loss_black + loss_white) / 2
        return self.reduce_loss(loss)
    
    def kl_loss(self, pred_posterior: DiagonalGaussianDistribution, ref: DiagonalGaussianDistribution = None):
        loss = pred_posterior.kl(ref)
        return self.reduce_loss(loss)
    
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        # print("last_layer.requires_grad: ", last_layer.requires_grad)
        # print("nll_loss grad_fn: ", nll_loss.grad_fn)
        # print("g_loss grad_fn: ", g_loss.grad_fn)
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        # print("nll_grads:", nll_grads)
        # print("g_grads:", g_grads)
            
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        # d_weight = d_weight * self.discriminator_weight
        return d_weight
    
    def generator_loss(self, rec_loss, reconstructions, last_layer):
        self.discriminator.requires_grad_(False)
        logits_fake = self.discriminator(reconstructions)
        g_loss = -torch.mean(logits_fake)
        d_weight = self.calculate_adaptive_weight(rec_loss, g_loss, last_layer=last_layer)
        return g_loss * d_weight
    
    def discriminator_loss(self, inputs, reconstructions):
        self.discriminator.requires_grad_(True)
        logits_real = self.discriminator(inputs.contiguous().detach())
        logits_fake = self.discriminator(reconstructions.contiguous().detach())
        return hinge_d_loss(logits_real, logits_fake)
    