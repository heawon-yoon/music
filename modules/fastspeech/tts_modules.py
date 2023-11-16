import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.commons.common_layers import SinusoidalPositionalEmbedding, EncSALayer, BatchNorm1dTBC
from modules.commons.espnet_positional_embedding import RelPositionalEncoding

DEFAULT_MAX_SOURCE_POSITIONS = 2000
DEFAULT_MAX_TARGET_POSITIONS = 2000


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, dropout, kernel_size=None, padding='SAME', act='gelu', num_heads=2, norm='ln'):
        super().__init__()
        self.op = EncSALayer(
            hidden_size, num_heads, dropout=dropout,
            attention_dropout=0.0, relu_dropout=dropout,
            kernel_size=kernel_size,
            padding=padding,
            norm=norm, act=act
        )

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)


######################
# fastspeech modules
######################
class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


class DurationPredictor(torch.nn.Module):
    """Duration predictor module.
    This is a module of duration predictor described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The duration predictor predicts a duration of each frame in log domain from the hidden embeddings of encoder.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    Note:
        The calculation domain of outputs is different between in `forward` and in `inference`. In `forward`,
        the outputs are calculated in log domain but in `inference`, those are calculated in linear domain.
    """

    def __init__(self, in_dims, n_layers=2, n_chans=384, kernel_size=3,
                 dropout_rate=0.1, offset=1.0, padding='SAME', dur_loss_type='mse'):
        """Initialize duration predictor module.
        Args:
            in_dims (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.
        """
        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = in_dims if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.nn.ConstantPad1d(((kernel_size - 1) // 2, (kernel_size - 1) // 2)
                                       if padding == 'SAME'
                                       else (kernel_size - 1, 0), 0),
                torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )]

        self.loss_type = dur_loss_type
        if self.loss_type in ['mse', 'huber']:
            self.out_dims = 1
        # elif hparams['dur_loss_type'] == 'mog':
        #     out_dims = 15
        # elif hparams['dur_loss_type'] == 'crf':
        #     out_dims = 32
        #     from torchcrf import CRF
        #     self.crf = CRF(out_dims, batch_first=True)
        else:
            raise NotImplementedError()
        self.linear = torch.nn.Linear(n_chans, self.out_dims)

    def out2dur(self, xs):
        if self.loss_type in ['mse', 'huber']:
            # NOTE: calculate loss in log domain
            dur = xs.squeeze(-1).exp() - self.offset  # (B, Tmax)
        # elif hparams['dur_loss_type'] == 'crf':
        #     dur = torch.LongTensor(self.crf.decode(xs)).cuda()
        else:
            raise NotImplementedError()
        return dur

    def forward(self, xs, x_masks=None, infer=True):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (BoolTensor, optional): Batch of masks indicating padded part (B, Tmax).
            infer (bool): Whether inference
        Returns:
            (train) FloatTensor, (infer) LongTensor: Batch of predicted durations in linear domain (B, Tmax).
        """
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        masks = 1 - x_masks.float()
        masks_ = masks[:, None, :]
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)
            if x_masks is not None:
                xs = xs * masks_
        xs = self.linear(xs.transpose(1, -1))  # [B, T, C]
        xs = xs * masks[:, :, None]  # (B, T, C)

        dur_pred = self.out2dur(xs)
        if infer:
            dur_pred = dur_pred.clamp(min=0.)  # avoid negative value
        return dur_pred


class VariancePredictor(torch.nn.Module):
    def __init__(self, vmin, vmax, in_dims,
                 n_layers=5, n_chans=512, kernel_size=5,
                 dropout_rate=0.1, padding='SAME'):
        """Initialize variance predictor module.
        Args:
            in_dims (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
        """
        super(VariancePredictor, self).__init__()

        self.vmin = vmin
        self.vmax = vmax
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = in_dims if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.nn.ConstantPad1d(((kernel_size - 1) // 2, (kernel_size - 1) // 2)
                                       if padding == 'SAME'
                                       else (kernel_size - 1, 0), 0),
                torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )]
        self.linear = torch.nn.Linear(n_chans, 1)
        self.embed_positions = SinusoidalPositionalEmbedding(in_dims, 0, init_size=4096)
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))

    def out2value(self, xs):
        return (xs + 1) / 2 * (self.vmax - self.vmin) + self.vmin

    def forward(self, xs, infer=True):
        """
        :param xs: [B, T, H]
        :param infer: whether inference
        :return: [B, T]
        """
        positions = self.pos_embed_alpha * self.embed_positions(xs[..., 0])
        xs = xs + positions
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)
        xs = self.linear(xs.transpose(1, -1)).squeeze(-1)  # (B, Tmax)
        if infer:
            xs = self.out2value(xs)
        return xs


class PitchPredictor(torch.nn.Module):
    def __init__(self, vmin, vmax, num_bins, deviation,
                 in_dims, n_layers=5, n_chans=384, kernel_size=5,
                 dropout_rate=0.1, padding='SAME'):
        """Initialize pitch predictor module.
        Args:
            in_dims (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
        """
        super(PitchPredictor, self).__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.interval = (vmax - vmin) / (num_bins - 1)  # align with centers of bins
        self.sigma = deviation / self.interval
        self.register_buffer('x', torch.arange(num_bins).float().reshape(1, 1, -1))  # [1, 1, N]

        self.base_pitch_embed = torch.nn.Linear(1, in_dims)
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = in_dims if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.nn.ConstantPad1d(((kernel_size - 1) // 2, (kernel_size - 1) // 2)
                                       if padding == 'SAME'
                                       else (kernel_size - 1, 0), 0),
                torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )]
        self.linear = torch.nn.Linear(n_chans, num_bins)
        self.embed_positions = SinusoidalPositionalEmbedding(in_dims, 0, init_size=4096)
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))

    def bins_to_values(self, bins):
        return bins * self.interval + self.vmin

    def out2pitch(self, probs):
        logits = probs.sigmoid()  # [B, T, N]
        # return logits
        # logits_sum = logits.sum(dim=2)  # [B, T]
        bins = torch.sum(self.x * logits, dim=2) / torch.sum(logits, dim=2)  # [B, T]
        pitch = self.bins_to_values(bins)
        # uv = logits_sum / (self.sigma * math.sqrt(2 * math.pi)) < 0.3
        # pitch[uv] = torch.nan
        return pitch

    def forward(self, xs, base):
        """
        :param xs: [B, T, H]
        :param base: [B, T]
        :return: [B, T, N]
        """
        xs = xs + self.base_pitch_embed(base[..., None])
        positions = self.pos_embed_alpha * self.embed_positions(xs[..., 0])
        xs = xs + positions
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)
        xs = self.linear(xs.transpose(1, -1))  # (B, Tmax, H)
        return self.out2pitch(xs) + base, xs


class RhythmRegulator(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, ph_dur, ph2word, word_dur):
        """
        Example (no batch dim version):
            1. ph_dur = [4,2,3,2]
            2. word_dur = [3,4,2], ph2word = [1,2,2,3]
            3. word_dur_in = [4,5,2]
            4. alpha_w = [0.75,0.8,1], alpha_ph = [0.75,0.8,0.8,1]
            5. ph_dur_out = [3,1.6,2.4,2]
        :param ph_dur: [B, T_ph]
        :param ph2word: [B, T_ph]
        :param word_dur: [B, T_w]
        """
        ph_dur = ph_dur.float() * (ph2word > 0)
        word_dur = word_dur.float()
        word_dur_in = ph_dur.new_zeros(ph_dur.shape[0], ph2word.max() + 1).scatter_add(
            1, ph2word, ph_dur
        )[:, 1:]  # [B, T_ph] => [B, T_w]
        alpha_w = word_dur / word_dur_in.clamp(min=self.eps)  # avoid dividing by zero
        alpha_ph = torch.gather(F.pad(alpha_w, [1, 0]), 1, ph2word)  # [B, T_w] => [B, T_ph]
        ph_dur_out = ph_dur * alpha_ph
        return ph_dur_out.round().long()


class LengthRegulator(torch.nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, dur, dur_padding=None, alpha=None):
        """
        Example (no batch dim version):
            1. dur = [2,2,3]
            2. token_idx = [[1],[2],[3]], dur_cumsum = [2,4,7], dur_cumsum_prev = [0,2,4]
            3. token_mask = [[1,1,0,0,0,0,0],
                             [0,0,1,1,0,0,0],
                             [0,0,0,0,1,1,1]]
            4. token_idx * token_mask = [[1,1,0,0,0,0,0],
                                         [0,0,2,2,0,0,0],
                                         [0,0,0,0,3,3,3]]
            5. (token_idx * token_mask).sum(0) = [1,1,2,2,3,3,3]

        :param dur: Batch of durations of each frame (B, T_txt)
        :param dur_padding: Batch of padding of each frame (B, T_txt)
        :param alpha: duration rescale coefficient
        :return:
            mel2ph (B, T_speech)
        """
        assert alpha is None or alpha > 0
        if alpha is not None:
            dur = torch.round(dur.float() * alpha).long()
        if dur_padding is not None:
            dur = dur * (1 - dur_padding.long())
        token_idx = torch.arange(1, dur.shape[1] + 1)[None, :, None].to(dur.device)
        dur_cumsum = torch.cumsum(dur, 1)
        dur_cumsum_prev = F.pad(dur_cumsum, [1, -1], mode='constant', value=0)

        pos_idx = torch.arange(dur.sum(-1).max())[None, None].to(dur.device)
        token_mask = (pos_idx >= dur_cumsum_prev[:, :, None]) & (pos_idx < dur_cumsum[:, :, None])
        mel2ph = (token_idx * token_mask.long()).sum(1)
        return mel2ph


class StretchRegulator(torch.nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, mel2ph, dur=None):
        """
        Example (no batch dim version):
            1. dur = [2,4,3]
            2. mel2ph = [1,1,2,2,2,2,3,3,3]
            3. mel2dur = [2,2,4,4,4,4,3,3,3]
            4. bound_mask = [0,1,0,0,0,1,0,0,1]
            5. 1 - bound_mask * mel2dur = [1,-1,1,1,1,-3,1,1,-2] => pad => [0,1,-1,1,1,1,-3,1,1]
            6. stretch_denorm = [0,1,0,1,2,3,0,1,2]

        :param dur: Batch of durations of each frame (B, T_txt)
        :param mel2ph: Batch of mel2ph (B, T_speech)
        :return:
            stretch (B, T_speech)
        """
        if dur is None:
            dur = mel2ph_to_dur(mel2ph, mel2ph.max())
        dur = F.pad(dur, [1, 0], value=1)  # Avoid dividing by zero
        mel2dur = torch.gather(dur, 1, mel2ph)
        bound_mask = torch.gt(mel2ph[:, 1:], mel2ph[:, :-1])
        bound_mask = F.pad(bound_mask, [0, 1], mode='constant', value=True)
        stretch_delta = 1 - bound_mask * mel2dur
        stretch_delta = F.pad(stretch_delta, [1, -1], mode='constant', value=0)
        stretch_denorm = torch.cumsum(stretch_delta, dim=1)
        stretch = stretch_denorm / mel2dur
        return stretch * (mel2ph > 0)


def mel2ph_to_dur(mel2ph, T_txt, max_dur=None):
    B, _ = mel2ph.shape
    dur = mel2ph.new_zeros(B, T_txt + 1).scatter_add(1, mel2ph, torch.ones_like(mel2ph))
    dur = dur[:, 1:]
    if max_dur is not None:
        dur = dur.clamp(max=max_dur)
    return dur


class FastSpeech2Encoder(nn.Module):
    def __init__(self, embed_tokens, hidden_size, num_layers,
                 ffn_kernel_size=9, ffn_padding='SAME', ffn_act='gelu',
                 dropout=None, num_heads=2, use_last_norm=True, norm='ln',
                 use_pos_embed=True, rel_pos=True):
        super().__init__()
        self.num_layers = num_layers
        embed_dim = self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_pos_embed = use_pos_embed
        self.use_last_norm = use_last_norm

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                self.hidden_size, self.dropout,
                kernel_size=ffn_kernel_size, padding=ffn_padding, act=ffn_act,
                num_heads=num_heads
            )
            for _ in range(self.num_layers)
        ])
        if self.use_last_norm:
            if norm == 'ln':
                self.layer_norm = nn.LayerNorm(embed_dim)
            elif norm == 'bn':
                self.layer_norm = BatchNorm1dTBC(embed_dim)
        else:
            self.layer_norm = None

        self.embed_tokens = embed_tokens  # redundant, but have to persist for compatibility with old checkpoints
        self.embed_scale = math.sqrt(hidden_size)
        self.padding_idx = 0
        self.rel_pos = rel_pos
        if self.rel_pos:
            self.embed_positions = RelPositionalEncoding(hidden_size, dropout_rate=0.0)
        else:
            self.embed_positions = SinusoidalPositionalEmbedding(
                hidden_size, self.padding_idx, init_size=DEFAULT_MAX_TARGET_POSITIONS,
            )

    def forward_embedding(self, main_embed, extra_embed=None, padding_mask=None):
        # embed tokens and positions
        x = self.embed_scale * main_embed
        if extra_embed is not None:
            x = x + extra_embed
        if self.use_pos_embed:
            if self.rel_pos:
                x = self.embed_positions(x)
            else:
                positions = self.embed_positions(~padding_mask)
                x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, main_embed, extra_embed, padding_mask, attn_mask=None, return_hiddens=False):
        x = self.forward_embedding(main_embed, extra_embed, padding_mask=padding_mask)  # [B, T, H]
        nonpadding_mask_TB = 1 - padding_mask.transpose(0, 1).float()[:, :, None]  # [T, B, 1]

        # NOTICE:
        # The following codes are commented out because
        # `self.use_pos_embed` is always False in the older versions,
        # and this argument did not compat with `hparams['use_pos_embed']`,
        # which defaults to True. The new version fixed this inconsistency,
        # resulting in temporary removal of pos_embed_alpha, which has actually
        # never been used before.

        # if self.use_pos_embed:
        #     positions = self.pos_embed_alpha * self.embed_positions(x[..., 0])
        #     x = x + positions
        #     x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1) * nonpadding_mask_TB
        hiddens = []
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=padding_mask, attn_mask=attn_mask) * nonpadding_mask_TB
            hiddens.append(x)
        if self.use_last_norm:
            x = self.layer_norm(x) * nonpadding_mask_TB
        if return_hiddens:
            x = torch.stack(hiddens, 0)  # [L, T, B, C]
            x = x.transpose(1, 2)  # [L, B, T, C]
        else:
            x = x.transpose(0, 1)  # [B, T, C]
        return x
