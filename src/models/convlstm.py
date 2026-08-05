"""Full-resolution ConvLSTM encoder for the controlled recurrent-cell comparison."""
from __future__ import annotations

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    """A 3x3 ConvLSTM cell preserving the input spatial grid."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_gate, forget_gate, output_gate, candidate = self.gates(torch.cat((x, hidden), 1)).chunk(4, 1)
        cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * torch.tanh(candidate)
        hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
        return hidden, cell


class ConvLSTM(nn.Module):
    """Unrolled ConvLSTM returning only the final full-resolution hidden state."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.cell = ConvLSTMCell(input_channels, hidden_channels)
        self.hidden_channels = hidden_channels

    def forward_stateful(
        self,
        sequence: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_sequence: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Advance a site-specific state; callers reset it only at a boundary."""
        batch, _, _, height, width = sequence.shape
        if state is None:
            hidden = sequence.new_zeros((batch, self.hidden_channels, height, width))
            cell = sequence.new_zeros((batch, self.hidden_channels, height, width))
        else:
            hidden, cell = state
        states = []
        for step in sequence.unbind(1):
            hidden, cell = self.cell(step, hidden, cell)
            if return_sequence:
                states.append(hidden)
        output = torch.stack(states, 1) if return_sequence else hidden
        return output, (hidden, cell)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.forward_stateful(sequence)
        return hidden
