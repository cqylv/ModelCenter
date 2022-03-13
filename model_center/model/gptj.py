# coding=utf-8
import torch
from ..layer import Encoder, Embedding, Projection, RotaryEmbedding
from .basemodel import BaseModel
from .config import GPTjConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

class GPTj(BaseModel):

    _CONFIG_TYPE = GPTjConfig

    def __init__(self, config: GPTjConfig):
        
        super().__init__()

        self.encoder = Encoder(
            num_layers = config.num_layers,
            dim_model = config.dim_model, 
            dim_ff = config.dim_ff,
            num_heads = config.num_heads,
            dim_head = config.dim_head,
            dtype = config.dtype, 
            int8 = config.int8,
            norm_eps = config.norm_eps, 
            norm_init_var = config.norm_init_var,
            norm_bias = config.norm_bias,
            att_init_mean = config.att_init_mean, 
            att_init_std = config.att_init_std,
            att_bias = config.att_bias,
            att_mask_value = float(config.att_mask_value),
            pos_bias_type = config.pos_bias_type,
            ffn_init_mean = config.ffn_init_mean, 
            ffn_init_std = config.ffn_init_std,
            ffn_bias = config.ffn_bias,
            ffn_activate_fn = config.ffn_activate_fn,
            length_scale = config.length_scale,
            attn_scale = config.attn_scale,
            dropout_p = config.dropout_p,
            parallel_ffn = True,
        )

        self.input_embedding = Embedding(
            vocab_size = config.vocab_size, 
            embedding_size = config.dim_model,
            length_scale = config.length_scale,
            dtype = config.dtype,
            int8 = config.int8,
            init_mean = config.emb_init_mean,
            init_std = config.emb_init_std,
        )

        self.position_bias = RotaryEmbedding(
            rotary_dim = config.pos_rotary_dim,
        )
        
        self.tied = config.tied
        self.cls_head = config.cls_head
        if self.cls_head:
            self.cls_projection = Projection(
                dim_out = self.cls_head,
                dim_in = config.dim_model,
                length_scale = config.length_scale,
                dtype = config.dtype,
                int8 = config.int8,
                init_mean = config.proj_init_mean,
                init_std = config.proj_init_std,
                bias = config.proj_bias,
            )
        if not self.tied:
            self.output_projection = Projection(
                dim_out = config.vocab_size,
                dim_in = config.dim_model,
                length_scale = config.length_scale,
                dtype = config.dtype,
                int8 = config.int8,
                init_mean = config.proj_init_mean,
                init_std = config.proj_init_std,
                bias = config.proj_bias,
            )

    def forward(self, 
                input_ids=None, # (batch, seqlen)
                length=None, # (batch)
                attention_mask=None,
                token_type_ids=None, #unused
                position_ids=None, #unused
                head_mask=None, #unused
                inputs_embeds=None,
                output_attentions=None, #unused
                output_hidden_states=None, #unused
                return_dict=True,
                return_logits = False,
    ):
        assert input_ids is not None or inputs_embeds is not None

        if input_ids is not None:
            batch = input_ids.size(0)
            seq_length = input_ids.size(1)
            device = input_ids.device
        else:
            batch = inputs_embeds.size(0)
            seq_length = inputs_embeds.size(1)
            device = inputs_embeds.device

        with torch.no_grad():
            if attention_mask is not None:
                attention_mask = attention_mask.to(torch.bool)
            else:
                attention_mask = torch.arange(seq_length, device=device)[None, :].repeat(batch, 1) < length[:, None]
            directional_mask_2d = torch.arange(seq_length, device=device).view(-1, 1) <= torch.arange(seq_length, device=device)
            attention_mask = attention_mask.view(batch, seq_length, 1) & directional_mask_2d.view(1, seq_length, seq_length)

        if inputs_embeds is None:
            hidden_states = self.input_embedding(input_ids)
        else:
            hidden_states = inputs_embeds

        hidden_states = self.encoder(hidden_states, attention_mask, self.position_bias)

        if self.cls_head:
            logits = self.cls_projection(hidden_states)
        elif self.tied:
            logits = self.input_embedding.projection(hidden_states)
        elif not self.tied:
            logits = self.output_projection(hidden_states)

        if return_logits:
            return logits

        if not return_dict:
            return tuple(hidden_states, None, None, None, None)
        else:
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
                cross_attentions=None,
            )