from __future__ import annotations

import math

import torch
from torch import nn


def extract(values: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return values.gather(0, t).reshape(t.shape[0], *((1,) * (len(shape) - 1)))


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    return (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-5, 0.999).float()


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps: int = 1000, beta_schedule: str = "cosine"):
        super().__init__()
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1.0 - alpha_bar).sqrt())

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        noise = torch.randn_like(z0) if noise is None else noise
        return extract(self.sqrt_alpha_bar, t, z0.shape) * z0 + extract(self.sqrt_one_minus_alpha_bar, t, z0.shape) * noise

    def predict_x0(self, z_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (z_t - extract(self.sqrt_one_minus_alpha_bar, t, z_t.shape) * eps) / extract(self.sqrt_alpha_bar, t, z_t.shape).clamp_min(1e-8)

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        z_li: torch.Tensor,
        z_raw: torch.Tensor,
        mask: torch.Tensor,
        steps: int = 50,
        eta: float = 0.0,
        start_timestep: int | None = None,
    ) -> torch.Tensor:
        device = z_li.device
        batch = z_li.shape[0]
        start = self.timesteps - 1 if start_timestep is None else min(max(1, int(start_timestep)), self.timesteps - 1)
        sequence = torch.linspace(start, 0, steps, device=device).long().unique_consecutive()
        initial_noise = torch.randn_like(z_li)
        initial_t = torch.full((batch,), start, device=device, dtype=torch.long)
        z = self.q_sample(z_li, initial_t, initial_noise) if start_timestep is not None else initial_noise
        known_noise = torch.randn_like(z_raw)
        for i, time in enumerate(sequence):
            t = torch.full((batch,), int(time.item()), device=device, dtype=torch.long)
            eps = model(z, t, z_li, z_raw, mask)
            x0 = self.predict_x0(z, t, eps).clamp(-4.0, 4.0)
            if i == len(sequence) - 1:
                z = x0
                break
            next_time = sequence[i + 1]
            a_t = self.alpha_bar[time]
            a_next = self.alpha_bar[next_time]
            sigma = eta * torch.sqrt(((1 - a_next) / (1 - a_t)) * (1 - a_t / a_next)).clamp_min(0)
            direction = torch.sqrt((1 - a_next - sigma.square()).clamp_min(0)) * eps
            z = torch.sqrt(a_next) * x0 + direction + sigma * torch.randn_like(z)
            known_t = torch.full((batch,), int(next_time.item()), device=device, dtype=torch.long)
            known = self.q_sample(z_raw, known_t, known_noise)
            latent_mask = mask
            if latent_mask.shape[-2:] != z.shape[-2:]:
                latent_mask = torch.nn.functional.interpolate(
                    latent_mask, z.shape[-2:], mode="bilinear", align_corners=False
                )
            z = latent_mask * z + (1.0 - latent_mask) * known
        return z

    @torch.no_grad()
    def ddim_sample_residual(
        self,
        model: nn.Module,
        z_li: torch.Tensor,
        z_raw: torch.Tensor,
        mask: torch.Tensor,
        li_delta: torch.Tensor,
        steps: int = 50,
        eta: float = 0.0,
        start_timestep: int | None = 500,
        support_threshold: float = 0.10,
    ) -> torch.Tensor:
        device = z_li.device
        batch = z_li.shape[0]
        start = self.timesteps - 1 if start_timestep is None else min(
            max(1, int(start_timestep)), self.timesteps - 1
        )
        sequence = torch.linspace(start, 0, steps, device=device).long().unique_consecutive()
        prior = z_raw - z_li
        initial_t = torch.full((batch,), start, device=device, dtype=torch.long)
        residual = self.q_sample(prior, initial_t, torch.randn_like(prior))
        soft_support = mask
        if soft_support.shape[-2:] != residual.shape[-2:]:
            soft_support = torch.nn.functional.interpolate(
                soft_support, residual.shape[-2:], mode="bilinear", align_corners=False
            )
        hard_support = (soft_support >= float(support_threshold)).to(residual.dtype)
        residual = residual * hard_support

        for i, time in enumerate(sequence):
            t = torch.full((batch,), int(time.item()), device=device, dtype=torch.long)
            eps = model(residual, t, z_li, z_raw, soft_support, li_delta)
            residual0 = self.predict_x0(residual, t, eps).clamp(-4.0, 4.0)
            if i == len(sequence) - 1:
                residual = residual0 * hard_support
                break
            next_time = sequence[i + 1]
            a_t = self.alpha_bar[time]
            a_next = self.alpha_bar[next_time]
            sigma = eta * torch.sqrt(
                ((1 - a_next) / (1 - a_t)) * (1 - a_t / a_next)
            ).clamp_min(0)
            direction = torch.sqrt((1 - a_next - sigma.square()).clamp_min(0)) * eps
            residual = (
                torch.sqrt(a_next) * residual0
                + direction
                + sigma * torch.randn_like(residual)
            )
            residual = residual * hard_support
        return residual
