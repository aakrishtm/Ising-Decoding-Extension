"""3D CNN pre-decoder for the dual-rail 7-channel tensor contract."""

from __future__ import annotations

import torch
import torch.nn as nn


class DualRailCNN3DPreDecoder(nn.Module):
    """Small fully convolutional 3D pre-decoder.

    Input shape: ``(B, 7, T, H, W)``.
    Output shape: ``(B, 4, T, H, W)`` logits.
    """

    def __init__(
        self,
        *,
        in_channels: int = 7,
        out_channels: int = 4,
        hidden_channels: int = 128,
        depth: int = 4,
        kernel_size: int = 3,
        dropout_p: float = 0.05,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth!r}")
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd for same-padding, got {kernel_size!r}")

        layers: list[nn.Module] = []
        current_channels = int(in_channels)
        padding = int(kernel_size) // 2
        for _ in range(int(depth) - 1):
            layers.extend(
                [
                    nn.Conv3d(
                        in_channels=current_channels,
                        out_channels=int(hidden_channels),
                        kernel_size=int(kernel_size),
                        padding=padding,
                    ),
                    nn.GELU(approximate="tanh"),
                    nn.Dropout3d(p=float(dropout_p)),
                ]
            )
            current_channels = int(hidden_channels)
        layers.append(
            nn.Conv3d(
                in_channels=current_channels,
                out_channels=int(out_channels),
                kernel_size=int(kernel_size),
                padding=padding,
            )
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model_from_config(config: dict) -> DualRailCNN3DPreDecoder:
    model_cfg = dict(config.get("model", {}))
    return DualRailCNN3DPreDecoder(
        in_channels=int(model_cfg.get("in_channels", 7)),
        out_channels=int(model_cfg.get("out_channels", 4)),
        hidden_channels=int(model_cfg.get("hidden_channels", 128)),
        depth=int(model_cfg.get("depth", 4)),
        kernel_size=int(model_cfg.get("kernel_size", 3)),
        dropout_p=float(model_cfg.get("dropout_p", 0.05)),
    )

