import torch


class ExphormerAttention(torch.nn.Module):
    def __init__(self) -> None:
        pass

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        expand_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
