"""SingleBigBirdLayer --> BigBirdAttention.

config
    attention type = block_sparse
    layers = 1
    n_heads = 8
    dim_hidden = 56
    dropout = 0.0
    attn_dropout = 0.0
    layer_norm = False
    batchNorm = True

call:
    self_attn = BigBirdAttention(...)
    h_attn = self_attn(h_dense, mask)
    result = h_attn[mask]

"""
import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import Linear


def pad_to_block_size(x: Tensor, mask: Tensor, block_size: int = 2):
    r"""A helper function to pad tokens and mask to work
    with implementation of BigBird block-sparse attention.
    """
    batch_size, seq_length, embed_dim = x.shape
    padding_len = (block_size - seq_length % block_size) % block_size
    if padding_len > 0:
        x_padding = torch.zeros((batch_size, padding_len, embed_dim),
                                dtype=torch.float, device=x.device)
        x = torch.cat([x, x_padding], dim=-2)
        mask = F.pad(mask, (0, padding_len),
                     value=False)  # no attention on the padding tokens
    return padding_len, x, mask


def create_masks_for_block_sparse_attn(mask: Tensor, block_size: int = 2):
    batch_size, seq_length = mask.size()
    assert (
        seq_length %
        block_size == 0), f"""Sequence length must be multiple of block size,
    but sequence length is {seq_length}, while block size is {block_size}."""

    def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
        r"""Create 3D attention mask from a 2D tensor mask.

        Args:
            from_blocked_mask: 2D Tensor of shape [batch_size,
            from_seq_length//from_block_size, from_block_size].
            to_blocked_mask: int32 Tensor of shape [batch_size,
            to_seq_length//to_block_size, to_block_size].

        Returns:
            float Tensor of shape [
                batch_size, 1, from_seq_length // from_block_size - 4,
                from_block_size, 3 * to_block_size].
        """
        exp_blocked_to_pad = torch.cat([
            to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2],
            to_blocked_mask[:, 3:-1]
        ], dim=2)
        band_mask = torch.einsum("blq,blk->blqk", from_blocked_mask[:, 2:-2],
                                 exp_blocked_to_pad)
        band_mask.unsqueeze_(1)
        return band_mask

    blocked_encoder_mask = mask.view(batch_size, seq_length // block_size,
                                     block_size)
    band_mask = create_band_mask_from_inputs(blocked_encoder_mask,
                                             blocked_encoder_mask)

    from_mask = mask.view(batch_size, 1, seq_length, 1)
    to_mask = mask.view(batch_size, 1, 1, seq_length)

    return blocked_encoder_mask, band_mask, from_mask, to_mask


class BigBirdOutput(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.dim_hidden, config.dim_hidden)
        self.LayerNorm = torch.nn.LayerNorm(config.dim_hidden,
                                            eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BigBirdAttention(torch.nn.Module):
    def __init__(
        self,
        max_seqlen,
        hidden_dim,
        n_heads,
        n_random_blocks,
        block_size,
        qkv_bias: bool = False,
    ):

        assert hidden_dim % n_heads == 0
        self.max_seqlen = max_seqlen
        self.num_attention_heads = n_heads
        self.num_random_blocks = n_random_blocks
        self.block_size = block_size
        self.attention_head_size = hidden_dim // n_heads
        self.all_head_size = hidden_dim

        self.q = Linear(hidden_dim, hidden_dim, bias=qkv_bias)
        self.k = Linear(hidden_dim, hidden_dim, bias=qkv_bias)
        self.v = Linear(hidden_dim, hidden_dim, bias=qkv_bias)

        self.output = BigBirdOutput()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def bigbird_block_sparse_attention(self):
        pass

    def forward(
        self,
        hidden_states,
        band_mask=None,
        from_mask=None,
        to_mask=None,
        from_blocked_mask=None,
        to_blocked_mask=None,
        output_attentions=None,
    ):

        batch_size, seqlen, _ = hidden_states.size()
        assert seqlen % self.block_size == 0
        to_seq_length = from_seq_length = seqlen
        to_block_size = from_block_size = self.block_size

        query_layer = self.transpose_for_scores(
            self.query(hidden_states))  # [bs, H, N, d]
        key_layer = self.transpose_for_scores(
            self.key(hidden_states))  # [bs, H, N, d]
        value_layer = self.transpose_for_scores(
            self.value(hidden_states))  # [bs, H, N, d]

        context_layer, attention_probs = self.bigbird_block_sparse_attention(
            query_layer,
            key_layer,
            value_layer,
            band_mask,
            from_mask,
            to_mask,
            from_blocked_mask,
            to_blocked_mask,
            self.num_attention_heads,
            self.num_random_blocks,
            self.attention_head_size,
            from_block_size,
            to_block_size,
            batch_size,
            from_seq_length,
            to_seq_length,
            seed=self.seed,
            plan_from_length=None,
            plan_num_rand_blocks=None,
            output_attentions=output_attentions,
        )

        context_layer = context_layer.contiguous().view(
            batch_size, from_seq_length, -1)

        outputs = (context_layer,
                   attention_probs) if output_attentions else (context_layer, )
        return outputs


class BigBirdEncoder(torch.nn.Module):
    def __init__(self, block_size, *args, **kwargs) -> None:
        self.block_size = block_size
        self.attention = BigBirdAttention(args)

    def forward(
        self,
        x,
        mask,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        band_mask=None,
        from_mask=None,
        to_mask=None,
        blocked_encoder_mask=None,
        return_dict=False,
    ):
        padding_len, x, mask = pad_to_block_size(x, mask, self.block_size)
        mask_lst = create_masks_for_block_sparse_attn(mask, self.block_size)
        blocked_encoder_mask, band_mask, from_mask, to_mask = mask_lst
        layer_head_mask = None

        self_attention_outputs = self.attention(
            x,
            attention_mask,
            layer_head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=None,
            output_attentions=output_attentions,
            band_mask=band_mask.float(),
            from_mask=from_mask.float(),
            to_mask=to_mask.float(),
            from_blocked_mask=blocked_encoder_mask.float(),
            to_blocked_mask=blocked_encoder_mask.float(),
        )[0]

        return self_attention_outputs[mask]

    def __repr__(self):
        return super().__repr__()
