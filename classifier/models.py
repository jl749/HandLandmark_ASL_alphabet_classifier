import torch
import torch.nn as nn
import torch.nn.functional as F


def pos_encodings(max_length: int, hidden_size: int) -> torch.Tensor:
    """
    max_length: L
    hidden_size: H
    :return: (L, H)
    """
    positions = torch.arange(max_length).view(-1, 1)  # scalar -> (L, 1)
    freqs = 0.0001**(torch.arange(hidden_size)[::2] / hidden_size).view(1, -1)  # (1, H/2)
    encodings = torch.zeros(size=(max_length, hidden_size))  # (L, H)

    # fill in the pairs by broadcast-multiplying freqs to positions; sin=odd_index cos=even_index
    encodings[:, ::2] = torch.sin(freqs * positions)  # (1, 50) * (10, 1) --> (10, 50)
    encodings[:, 1::2] = torch.cos(freqs * positions)

    return encodings

# pos_encodings(10, 42)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

