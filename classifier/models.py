import torch
import torch.nn as nn
import torch.nn.functional as F


def pos_encodings(max_length: int, hidden_size: int) -> torch.Tensor:  # TODO: test relative encoding too
    """
    max_length: L, 420 (7 * 60)
    hidden_size: 9 (x1, y1, z1, x2, y2, z2, x3, y3, z3)
    :return: (L, H)
    """
    # max_frames = 60
    # max_length += max_frames - 1  # PADS between frames

    positions = torch.arange(max_length).unsqueeze(-1)  # (L, 1)
    freqs = 0.0001**(torch.arange(hidden_size)[::2] / hidden_size).unsqueeze(0)  # (1, H/2)
    encodings = torch.zeros(size=(max_length, hidden_size))  # (L, H)

    # fill in the pairs by broadcast-multiplying freqs to positions; sin=odd_index cos=even_index
    encodings[:, ::2] = torch.sin(freqs * positions)  # (1, 50) * (10, 1) --> (10, 50)
    encodings[:, 1::2] = torch.cos(freqs * positions)

    return encodings


class Classifier(nn.Module):  # stacked up encoder TODO: XLNET, decoders + permutation?
    def __init__(self):
        super(Classifier, self).__init__()


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 max_length: int,
                 encoding_size: int,
                 heads: int,
                 # masked: bool = False  TODO: XLNET, decoders + permutation?
                 ):
        super(MultiHeadAttention, self).__init__()
        if encoding_size % heads != 0:
            raise ValueError("encoding_size(E) must be divisible by # of heads")
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.encoding_size = encoding_size
        self.heads = heads

        self.linear_q = nn.Linear(hidden_size, encoding_size)
        self.linear_k = nn.Linear(hidden_size, encoding_size)
        self.linear_v = nn.Linear(hidden_size, encoding_size)

        self.linear_o = nn.Linear(encoding_size, hidden_size)
        self.bn = nn.BatchNorm2d(hidden_size,)

    def forward(self, q, k, v):
        N, _, _ = q.size()

        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        q = q.reshape(N, self.max_length, self.heads, self.encoding_size // self.heads)  # (N, L, heads, head_size)
        k = k.reshape(N, self.max_length, self.heads, self.encoding_size // self.heads)  # (N, L, heads, head_size)
        v = v.reshape(N, self.max_length, self.heads, self.encoding_size // self.heads)  # (N, L, heads, head_size)
        q = q.permute(0, 2, 1, 3)  # (N, heads, L, head_size)
        k = k.permute(0, 2, 1, 3)  # (N, heads, L, head_size)
        v = v.permute(0, 2, 1, 3)  # (N, heads, L, head_size)

        # TODO: scaled_dot_product_attention


class EncoderBlock(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 max_length: int,
                 heads: int,
                 ffn_size: int,
                 dropout: float,
                 activation=nn.SiLU()):  # TODO: experiment with different activation functions
        """

        :param hidden_size: H, default 9
        :param max_length: L, default 420
        :param heads: # of heads for multi-head attention
        :param ffn_size: linear out_feature size
        :param dropout: ffn dropout rate
        """
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(hidden_size, max_length, heads)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            activation,
            nn.Linear(ffn_size, hidden_size),
            nn.Dropout(dropout),
            nn.BatchNorm2d(hidden_size)  # originally LayerNorm
        )

    def forward(self, x):
        # --> attention --> ffn -->
        return self.ffn(self.attention(x))
