import torch
import torch.nn as nn
from layers.Transformer_EncDec import TDecoder, TDecoderLayer, TEncoder, TEncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding

class vTransformer(nn.Module):
	"""
	Vanilla Transformer with O(L^2) complexity
	"""
	def __init__(self, configs):
		super(vTransformer, self).__init__()
		self.pred_len = configs.pred_len
		self.output_attention = configs.output_attention

		# Embedding
		# learned emb as in patch tst
		self.encProjection = torch.nn.Linear(configs.enc_in, configs.d_model)
		self.decProjection = torch.nn.Linear(configs.dec_in, configs.d_model)
		self.encEmbedding = torch.nn.Parameter(torch.rand(configs.seq_len, configs.d_model), requires_grad=True)
		self.decEmbedding = torch.nn.Parameter(torch.rand(configs.seq_len, configs.d_model), requires_grad=True)
		
		# Encoder
		self.encoder = TEncoder(
			[
				TEncoderLayer(
					AttentionLayer(
						FullAttention(False, configs.factor, attention_dropout=configs.dropout,
									  output_attention=configs.output_attention), configs.d_model, configs.n_heads),
					configs.d_model,
					configs.d_ff,
					dropout=configs.dropout,
					activation=configs.activation
				) for l in range(configs.e_layers)
			],
			norm_layer=torch.nn.LayerNorm(configs.d_model)
		)
		# Decoder
		self.decoder = TDecoder(
			[
				TDecoderLayer(
					AttentionLayer(
						FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
						configs.d_model, configs.n_heads),
					AttentionLayer(
						FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
						configs.d_model, configs.n_heads),
					configs.d_model,
					configs.d_ff,
					dropout=configs.dropout,
					activation=configs.activation,
				)
				for l in range(configs.d_layers)
			],
			norm_layer=torch.nn.LayerNorm(configs.d_model),
			projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
		)

	def forward(self, x_enc, _, __, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
		x_dec = torch.cat([x_enc[:, -horizons:, :], torch.zeros_like(x_enc[:, -horizons:, :]).float()], dim=1).float().to(device)
		enc_out = self.encProjection(x_enc) + self.encEmbedding
		enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

		dec_out = self.decProjection(x_dec) + self.decEmbedding 
		dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

		if self.output_attention:
			return dec_out[:, -self.pred_len:, :][:, :, 0]
		else:
			return dec_out[:, -self.pred_len:, :][:, :, 0]