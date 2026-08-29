from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch


BOTTLENECK_FILENAME = "nla_bottleneck.pt"
BOTTLENECK_PROTOCOL_FILENAME = "nla_bottleneck_protocol.json"
IMPLEMENTATION_VERSION = "d16_soft_bottleneck_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def unit_normalize(values: torch.Tensor) -> torch.Tensor:
    if values.ndim not in (1, 2):
        raise ValueError(f"Expected one vector or a matrix, got {tuple(values.shape)}")
    return values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def canonicalize_basis_signs(basis: torch.Tensor) -> torch.Tensor:
    """Fix the otherwise arbitrary sign of each PCA basis vector."""

    if basis.ndim != 2:
        raise ValueError("PCA basis must be a matrix")
    result = basis.clone()
    pivots = result.abs().argmax(dim=1)
    signs = result[torch.arange(len(result)), pivots].sign()
    signs[signs == 0] = 1
    return result * signs.unsqueeze(1)


def fit_source_balanced_pca(
    ddxplus: torch.Tensor,
    direct: torch.Tensor,
    *,
    d_z: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit an equal-source-weight PCA on unit-normalized train activations."""

    if ddxplus.ndim != 2 or direct.ndim != 2:
        raise ValueError("PCA sources must be activation matrices")
    if ddxplus.shape[1] != direct.shape[1]:
        raise ValueError("PCA sources have different activation dimensions")
    if not 0 < d_z < ddxplus.shape[1]:
        raise ValueError(f"d_z must be in [1, {ddxplus.shape[1] - 1}]")
    if not len(ddxplus) or not len(direct):
        raise ValueError("PCA sources cannot be empty")

    ddx = unit_normalize(ddxplus.to(dtype=torch.float64, device="cpu"))
    drt = unit_normalize(direct.to(dtype=torch.float64, device="cpu"))
    mean = 0.5 * ddx.mean(dim=0) + 0.5 * drt.mean(dim=0)
    ddx_centered = ddx - mean
    direct_centered = drt - mean
    covariance = 0.5 * (ddx_centered.T @ ddx_centered) / len(ddx_centered)
    covariance += 0.5 * (direct_centered.T @ direct_centered) / len(direct_centered)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)[:d_z]
    basis = canonicalize_basis_signs(eigenvectors[:, order].T.contiguous())
    return mean, basis, eigenvalues[order]


class NlaBottleneckProjector(torch.nn.Module):
    """The D16 no-skip activation bottleneck retained at inference."""

    def __init__(self, mean: torch.Tensor, basis: torch.Tensor) -> None:
        super().__init__()
        if mean.ndim != 1 or basis.ndim != 2:
            raise ValueError("Expected mean[d_model] and basis[d_z,d_model]")
        if basis.shape[1] != mean.numel():
            raise ValueError("PCA mean and basis dimensions differ")
        self.d_model = int(mean.numel())
        self.d_z = int(basis.shape[0])
        self.register_buffer("mean", mean.detach().float().clone())
        self.down = torch.nn.Linear(self.d_model, self.d_z, bias=False)
        self.up = torch.nn.Linear(self.d_z, self.d_model, bias=False)
        with torch.no_grad():
            self.down.weight.copy_(basis.float())
            self.up.weight.copy_(basis.T.float())

    def encode(self, activation: torch.Tensor) -> torch.Tensor:
        normalized = unit_normalize(activation.float())
        return self.down(normalized - self.mean)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.mean + self.up(latent)

    def forward(self, activation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(activation)
        return self.decode(latent), latent

    def reconstruct_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decode(latent)


def reconstruction_metrics(
    projector: NlaBottleneckProjector, activations: torch.Tensor
) -> dict[str, float]:
    with torch.no_grad():
        source = unit_normalize(activations.float())
        reconstructed, _ = projector(source)
        cosine = torch.nn.functional.cosine_similarity(source, reconstructed, dim=-1)
        centered = source - projector.mean
        residual = source - reconstructed
        denominator = centered.square().sum().clamp_min(1e-12)
        retained_variance = 1.0 - float(residual.square().sum() / denominator)
    return {
        "n": int(len(source)),
        "mean_reconstruction_cosine": float(cosine.mean()),
        "min_reconstruction_cosine": float(cosine.min()),
        "retained_variance": retained_variance,
    }


def save_bottleneck(
    path: str | Path,
    projector: NlaBottleneckProjector,
    *,
    metadata: dict[str, Any],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "d_model": projector.d_model,
            "d_z": projector.d_z,
            "state_dict": {
                key: value.detach().cpu() for key, value in projector.state_dict().items()
            },
            "metadata": metadata,
        },
        output,
    )


def load_bottleneck(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    require_gate_passed: bool = True,
) -> tuple[NlaBottleneckProjector, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("schema_version") or -1) != 1:
        raise ValueError(f"Unsupported bottleneck schema in {path}")
    state = payload["state_dict"]
    mean = state["mean"]
    down = state["down.weight"]
    projector = NlaBottleneckProjector(mean, down)
    projector.load_state_dict(state, strict=True)
    metadata = dict(payload.get("metadata") or {})
    if require_gate_passed and not bool(metadata.get("pca_sanity_gate_passed")):
        raise ValueError(f"Bottleneck PCA sanity gate did not pass: {path}")
    projector.to(device)
    return projector, metadata
