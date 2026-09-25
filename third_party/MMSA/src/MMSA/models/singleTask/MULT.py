"""
From: https://github.com/yaohungt/Multimodal-Transformer
Paper: Multimodal Transformer for Unaligned Multimodal Language Sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..subNets import BertTextEncoder
from ..subNets.transformers_encoder.transformer import TransformerEncoder

__all__ = ['MULT']

class MULT(nn.Module):
    def __init__(self, args):
        super(MULT, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers, pretrained=args.pretrained)
        self.use_bert = args.use_bert
        # Mult Model Initialization.
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        self.mask_padding = getattr(args, 'mask_padding', False)
        self.mask_unavailable = getattr(args, 'mask_unavailable', False)

        combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
        
        output_dim = args.num_classes if args.train_mode == "classification" else 1
        
        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # 2. Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')
    
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
    
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        # Projection layers
        # print("combined_dim: shape")
        # print(combined_dim)
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        # TODO: Replace with nn.TransformerEncoder
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    @staticmethod
    def _pool_valid(hidden, content_mask):
        """Pool the actual utterance positions, excluding CLS/SEP and padding.

        The original MulT uses the last element, which is often padding in the
        contest's 50-step aligned files.  Keep that behavior when no mask is
        supplied so upstream MMSA callers are unaffected.
        """
        if content_mask is None:
            return hidden[-1]
        mask = content_mask.transpose(0, 1).unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=0) / mask.sum(dim=0).clamp_min(1.0)

    def forward(self, text, audio, video, content_mask=None, availability_masks=None):
        # Cross-modal attention must mask keys according to the *source*
        # modality. A single content mask hides padding but still exposes
        # zero-filled gaps as attention keys.
        if self.mask_padding and content_mask is not None:
            source_masks = {}
            for name in ('text', 'audio', 'vision'):
                available = (availability_masks.get(name, content_mask)
                             if self.mask_unavailable and availability_masks is not None
                             else content_mask)
                available = available.to(dtype=torch.bool) & content_mask
                key_mask = ~available.clone()
                # The attention implementation cannot softmax an all-masked
                # source. Keep one zero-input placeholder for that rare case.
                empty = ~available.any(dim=1)
                key_mask[empty, 0] = False
                source_masks[name] = key_mask
            memory_mask = ~content_mask
        else:
            source_masks = {'text': None, 'audio': None, 'vision': None}
            memory_mask = None
        if self.use_bert:
            text = self.text_model(text)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)
        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        # (V,A) --> L
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a, key_padding_mask=source_masks['audio'])    # Dimension (L, N, d_l)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v, key_padding_mask=source_masks['vision'])    # Dimension (L, N, d_l)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls, key_padding_mask=memory_mask)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = self._pool_valid(h_ls, content_mask)

        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l, key_padding_mask=source_masks['text'])
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v, key_padding_mask=source_masks['vision'])
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as, key_padding_mask=memory_mask)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = self._pool_valid(h_as, content_mask)
        
        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l, key_padding_mask=source_masks['text'])
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a, key_padding_mask=source_masks['audio'])
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = self.trans_v_mem(h_vs, key_padding_mask=memory_mask)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = self._pool_valid(h_vs, content_mask)
    
        last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        res = {
            'Feature_t': last_h_l,
            'Feature_a': last_h_a,
            'Feature_v': last_h_v,
            'Feature_f': last_hs,
            'Feature_fused': last_hs_proj,
            'Sequence_t': h_ls.transpose(0, 1),
            'Sequence_a': h_as.transpose(0, 1),
            'Sequence_v': h_vs.transpose(0, 1),
            'M': output
        }
        return res
