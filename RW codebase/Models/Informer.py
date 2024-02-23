import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.Informer_EncDec import IDecoder, IDecoderLayer, IEncoder, IEncoderLayer,IConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer,ProbAttention
from layers.Embed import DataEmbedding


class Informer(nn.Module):
	"""
	Informer with Propspare attention in O(LlogL) complexity
	"""
	def __init__(self, configs):
		super(Informer, self).__init__()
		self.pred_len = configs.pred_len
		self.output_attention = configs.output_attention

		# Embedding
		#self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
		#                                   configs.dropout)
		#self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
		#                                   configs.dropout)
		
		# learned emb as in patch tst
		self.encProjection = torch.nn.Linear(configs.enc_in, configs.d_model)
		self.decProjection = torch.nn.Linear(configs.dec_in, configs.d_model)
		self.encEmbedding = torch.nn.Parameter(torch.rand(configs.seq_len, configs.d_model), requires_grad=True)
		self.decEmbedding = torch.nn.Parameter(torch.rand(configs.seq_len, configs.d_model), requires_grad=True)
		
		# Encoder
		self.encoder = IEncoder(
			[
				IEncoderLayer(
					AttentionLayer(
						ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
									  output_attention=configs.output_attention),
						configs.d_model, configs.n_heads),
					configs.d_model,
					configs.d_ff,
					dropout=configs.dropout,
					activation=configs.activation
				) for l in range(configs.e_layers)
			],
			[
				IConvLayer(
					configs.d_model
				) for l in range(configs.e_layers - 1)
			] if configs.distil else None,
			norm_layer=torch.nn.LayerNorm(configs.d_model)
		)
		# Decoder
		self.decoder = IDecoder(
			[
				IDecoderLayer(
					AttentionLayer(
						ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
						configs.d_model, configs.n_heads),
					AttentionLayer(
						ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
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

	def forward(self, x_enc, _, __):
		
		x_dec = torch.cat([x_enc[:, -horizons:, :], torch.zeros_like(x_enc[:, -horizons:, :]).float()], dim=1).float().to(device)

		enc_out = self.encProjection(x_enc) + self.encEmbedding #self.enc_embedding(x_enc, x_mark_enc) # batch, length, channel
		enc_out, attns = self.encoder(enc_out, attn_mask=None)

		dec_out = self.decProjection(x_dec) + self.decEmbedding  #self.dec_embedding(x_dec, x_mark_dec)
		dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

		if self.output_attention:
			return dec_out[:, -self.pred_len:, :][:, :, 0]
		else:
			return dec_out[:, -self.pred_len:, :][:, :, 0]