"""Numerical action-space solvers ported from Alpamayo's public utilities."""

from __future__ import annotations

import logging

import einops
import torch

from .rotation import round_2pi_torch, so3_to_yaw_torch

logger = logging.getLogger(__name__)


def unwrap_angle(phi: torch.Tensor) -> torch.Tensor:
    """Unwrap the last dimension so diffs stay in (-pi, pi]."""

    d = torch.diff(phi, dim=-1)
    d = round_2pi_torch(d)
    return torch.cat([phi[..., :1], phi[..., :1] + torch.cumsum(d, dim=-1)], dim=-1)


def first_order_D(
    N: int,
    lead_shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    D = torch.zeros(*lead_shape, N - 1, N, dtype=dtype, device=device)
    rows = torch.arange(N - 1, device=device)
    D[..., rows, rows] = -1.0
    D[..., rows, rows + 1] = 1.0
    return D


def second_order_D(
    N: int,
    lead_shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    D = torch.zeros(*lead_shape, max(N - 2, 0), N, dtype=dtype, device=device)
    rows = torch.arange(max(N - 2, 0), device=device)
    D[..., rows, rows] = -1.0
    D[..., rows, rows + 1] = 2.0
    D[..., rows, rows + 2] = -1.0
    return D


def third_order_D(
    N: int,
    lead_shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    D = torch.zeros(*lead_shape, max(N - 3, 0), N, dtype=dtype, device=device)
    rows = torch.arange(max(N - 3, 0), device=device)
    D[..., rows, rows] = -1.0
    D[..., rows, rows + 1] = 3.0
    D[..., rows, rows + 2] = -3.0
    D[..., rows, rows + 3] = 1.0
    return D


@torch.amp.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
@torch._dynamo.disable()
def construct_DTD(
    N: int,
    lead: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    w_smooth1: float | torch.Tensor | None = None,
    w_smooth2: float | torch.Tensor | None = None,
    w_smooth3: float | torch.Tensor | None = None,
    lam: float = 1e-3,
    dt: float = 1.0,
) -> torch.Tensor:
    DTD = torch.zeros(*lead, N, N, dtype=dtype, device=device)
    if w_smooth1 is not None:
        lam_1 = lam / dt**2
        if isinstance(w_smooth1, float):
            w_smooth1_tensor = torch.full(
                (*lead, max(N - 1, 0)), w_smooth1, dtype=dtype, device=device
            )
        else:
            w_smooth1_tensor = w_smooth1
        D1 = first_order_D(N, lead, device=device, dtype=dtype)
        DTD += lam_1 * einops.einsum(
            D1 * w_smooth1_tensor.unsqueeze(-1), D1, "... i j, ... i k -> ... j k"
        )

    if w_smooth2 is not None:
        lam_2 = lam / dt**4
        if isinstance(w_smooth2, float):
            w_smooth2_tensor = torch.full(
                (*lead, max(N - 2, 0)), w_smooth2, dtype=dtype, device=device
            )
        else:
            w_smooth2_tensor = w_smooth2
        D2 = second_order_D(N, lead, device=device, dtype=dtype)
        DTD += lam_2 * einops.einsum(
            D2 * w_smooth2_tensor.unsqueeze(-1), D2, "... i j, ... i k -> ... j k"
        )

    if w_smooth3 is not None:
        lam_3 = lam / dt**6
        if isinstance(w_smooth3, float):
            w_smooth3_tensor = torch.full(
                (*lead, max(N - 3, 0)), w_smooth3, dtype=dtype, device=device
            )
        else:
            w_smooth3_tensor = w_smooth3
        D3 = third_order_D(N, lead, device=device, dtype=dtype)
        DTD += lam_3 * einops.einsum(
            D3 * w_smooth3_tensor.unsqueeze(-1), D3, "... i j, ... i k -> ... j k"
        )

    return DTD


@torch.amp.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
@torch._dynamo.disable()
def solve_single_constraint(
    x_init: torch.Tensor,
    x_target: torch.Tensor,
    w_data: torch.Tensor | None = None,
    w_smooth1: float | torch.Tensor | None = None,
    w_smooth2: float | torch.Tensor | None = None,
    w_smooth3: float | torch.Tensor | None = None,
    lam: float = 1e-3,
    ridge: float = 0.0,
    dt: float = 1.0,
) -> torch.Tensor:
    device, dtype = x_target.device, x_target.dtype
    *lead, N = x_target.shape
    if N <= 0:
        raise ValueError("x_target must have a positive last-dimension length N.")
    if w_data is None:
        w_data = torch.ones_like(x_target)
    x_init = torch.as_tensor(x_init, dtype=dtype, device=device)

    A_data = torch.eye(N, dtype=dtype, device=device).expand(*lead, N, N)
    Aw_data = A_data * w_data.unsqueeze(-1)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
        rhs = einops.einsum(Aw_data, x_target, "... i j, ... i -> ... j")

    DTD = construct_DTD(
        N + 1,
        lead,
        device=device,
        dtype=dtype,
        w_smooth1=w_smooth1,
        w_smooth2=w_smooth2,
        w_smooth3=w_smooth3,
        lam=lam,
        dt=dt,
    )
    rhs -= DTD[..., 1:, 0] * x_init.unsqueeze(-1)

    ridge_term = ridge * torch.eye(N, dtype=dtype, device=device).expand(*lead, N, N)
    lhs = ATA + DTD[..., 1:, 1:] + ridge_term

    L = torch.linalg.cholesky(lhs)
    x = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
    return torch.cat([x_init.unsqueeze(-1), x], dim=-1)


@torch.amp.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
@torch._dynamo.disable()
def solve_xs_eq_y(
    s: torch.Tensor,
    y: torch.Tensor,
    w_data: torch.Tensor | None = None,
    w_smooth1: float | torch.Tensor | None = None,
    w_smooth2: float | torch.Tensor | None = None,
    w_smooth3: float | torch.Tensor | None = None,
    lam: float = 1e-3,
    ridge: float = 0.0,
    dt: float = 1.0,
) -> torch.Tensor:
    device, dtype = y.device, y.dtype
    *lead, N = y.shape
    if w_data is None:
        w_data = torch.ones_like(y)
    if w_data.shape != y.shape:
        raise ValueError("w_data must have the same shape as y")

    A_data = torch.diag_embed(s)
    Aw_data = A_data * w_data.unsqueeze(-1)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
        rhs = einops.einsum(Aw_data, y, "... i j, ... i -> ... j")

    DTD = construct_DTD(
        N,
        lead,
        device=device,
        dtype=dtype,
        w_smooth1=w_smooth1,
        w_smooth2=w_smooth2,
        w_smooth3=w_smooth3,
        lam=lam,
        dt=dt,
    )

    L = None
    while L is None:
        try:
            ridge_term = ridge * torch.eye(N, dtype=dtype, device=device).expand(*lead, N, N)
            lhs = ATA + DTD + ridge_term
            if rhs.dtype != lhs.dtype:
                rhs = rhs.to(lhs.dtype)
            L = torch.linalg.cholesky(lhs)
        except RuntimeError as exc:
            logger.error("Error in cholesky decomposition: %s", exc, exc_info=True)
            ridge *= 10
            logger.warning("Resolving singularity using ridge %s", ridge)

    return torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
@torch._dynamo.disable()
def dxy_theta_to_v_without_v0(
    dxy: torch.Tensor,
    theta: torch.Tensor,
    dt: float = 1.0,
    v_lambda: float = 1e-4,
    v_ridge: float = 1e-4,
) -> torch.Tensor:
    *lead, N, _ = dxy.shape
    device, dtype = dxy.device, dxy.dtype
    g = 2 / dt * dxy
    w = torch.ones_like(dxy[..., 0])

    A_data = torch.zeros(*lead, 2 * N, N + 1, dtype=dtype, device=device)
    b_data = g.flatten(start_dim=-2)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_rows = 2 * torch.arange(N, device=device)
    sin_rows = 2 * torch.arange(N, device=device) + 1
    cols = torch.arange(N, device=device)
    A_data[..., cos_rows, cols] = cos_theta[..., :-1]
    A_data[..., cos_rows, cols + 1] = cos_theta[..., 1:]
    A_data[..., sin_rows, cols] = sin_theta[..., :-1]
    A_data[..., sin_rows, cols + 1] = sin_theta[..., 1:]
    Aw_data = A_data * torch.repeat_interleave(w, 2, dim=-1).unsqueeze(-1)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
        rhs = einops.einsum(Aw_data, b_data, "... i j, ... i -> ... j")

    DTD = construct_DTD(
        N + 1,
        lead,
        device=device,
        dtype=dtype,
        w_smooth1=None,
        w_smooth2=None,
        w_smooth3=1.0,
        lam=v_lambda,
        dt=dt,
    )
    ridge_term = v_ridge * torch.eye(N + 1, dtype=dtype, device=device).expand(
        *lead, N + 1, N + 1
    )
    lhs = ATA + DTD + ridge_term
    L = torch.linalg.cholesky(lhs)
    return torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
@torch._dynamo.disable()
def dxy_theta_to_v(
    dxy: torch.Tensor,
    theta: torch.Tensor,
    v0: torch.Tensor,
    dt: float = 1.0,
    v_lambda: float = 1e-4,
    v_ridge: float = 1e-4,
) -> torch.Tensor:
    *lead, N, _ = dxy.shape
    device, dtype = dxy.device, dxy.dtype
    g = 2 / dt * dxy
    w = torch.ones_like(dxy[..., 0])

    A_data = torch.zeros(*lead, 2 * N, N + 1, dtype=dtype, device=device)
    b_data = g.flatten(start_dim=-2)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_rows = 2 * torch.arange(N, device=device)
    sin_rows = 2 * torch.arange(N, device=device) + 1
    cols = torch.arange(N, device=device)
    A_data[..., cos_rows, cols] = cos_theta[..., :-1]
    A_data[..., cos_rows, cols + 1] = cos_theta[..., 1:]
    A_data[..., sin_rows, cols] = sin_theta[..., :-1]
    A_data[..., sin_rows, cols + 1] = sin_theta[..., 1:]
    Aw_data = A_data * torch.repeat_interleave(w, 2, dim=-1).unsqueeze(-1)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
        rhs = einops.einsum(Aw_data[..., :, 1:], b_data, "... i j, ... i -> ... j")
    rhs -= ATA[..., 1:, 0] * v0.unsqueeze(-1)

    DTD = construct_DTD(
        N + 1,
        lead,
        device=device,
        dtype=dtype,
        w_smooth1=None,
        w_smooth2=None,
        w_smooth3=1.0,
        lam=v_lambda,
        dt=dt,
    )
    rhs -= DTD[..., 1:, 0] * v0.unsqueeze(-1)

    ridge_term = v_ridge * torch.eye(N, dtype=dtype, device=device).expand(*lead, N, N)
    lhs = ATA[..., 1:, 1:] + DTD[..., 1:, 1:] + ridge_term

    L = torch.linalg.cholesky(lhs)
    y = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
    return torch.cat([v0.unsqueeze(-1), y], dim=-1)


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
@torch._dynamo.disable()
def theta_smooth(
    traj_future_rot: torch.Tensor,
    dt: float = 1.0,
    theta_lambda: float = 1e-4,
    theta_ridge: float = 1e-4,
) -> torch.Tensor:
    theta = so3_to_yaw_torch(traj_future_rot)
    theta = unwrap_angle(theta)
    theta_init = torch.zeros_like(theta[..., 0])
    return solve_single_constraint(
        x_init=theta_init,
        x_target=theta,
        w_smooth1=None,
        w_smooth2=None,
        w_smooth3=1.0,
        dt=dt,
        lam=theta_lambda,
        ridge=theta_ridge,
    )
