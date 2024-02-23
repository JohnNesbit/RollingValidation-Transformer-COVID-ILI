import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from layers.Embed import DataEmbedding_wo_pos
from layers.MultiWaveletCorrelation import MultiWaveletTransform,MultiWaveletCross
from layers.FourierCorrelation import FourierBlock,FourierCrossAttention
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FEDFormer(nn.Module):
	"""
	FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
	"""
	def __init__(self, configs):
		super(FEDFormer, self).__init__()
		self.version = configs.version
		self.mode_select = configs.mode_select
		self.modes = configs.modes
		self.seq_len = configs.seq_len
		self.label_len = configs.label_len
		self.pred_len = configs.pred_len
		self.output_attention = configs.output_attention

		# Decomp
		kernel_size = configs.moving_avg
		if isinstance(kernel_size, list):
			self.decomp = series_decomp_multi(kernel_size)
		else:
			self.decomp = series_decomp(kernel_size)

		# Embedding
		# The series-wise connection inherently contains the sequential information.
		# Thus, we can discard the position embedding of transformers.
		# learned emb as in patch tst
		self.encProjection = torch.nn.Linear(configs.enc_in, configs.d_model)
		self.decProjection = torch.nn.Linear(configs.dec_in, configs.d_model)
		self.encEmbedding = torch.nn.Parameter(torch.rand(configs.seq_len, configs.d_model), requires_grad=True)
		self.decEmbedding = torch.nn.Parameter(torch.rand(configs.seq_len, configs.d_model), requires_grad=True)

		if configs.version == 'Wavelets':
			encoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=configs.L, base=configs.base)
			decoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=configs.L, base=configs.base)
			decoder_cross_att = MultiWaveletCross(in_channels=configs.d_model,
												  out_channels=configs.d_model,
												  seq_len_q=self.seq_len // 2 + self.pred_len,
												  seq_len_kv=self.seq_len,
												  modes=configs.modes,
												  ich=configs.d_model,
												  base=configs.base,
												  activation=configs.cross_activation)
		else:
			encoder_self_att = FourierBlock(in_channels=configs.d_model,
											out_channels=configs.d_model,
											seq_len=self.seq_len,
											modes=configs.modes,
											mode_select_method=configs.mode_select)
			decoder_self_att = FourierBlock(in_channels=configs.d_model,
											out_channels=configs.d_model,
											seq_len=self.seq_len//2+self.pred_len,
											modes=configs.modes,
											mode_select_method=configs.mode_select)
			decoder_cross_att = FourierCrossAttention(in_channels=configs.d_model,
													  out_channels=configs.d_model,
													  seq_len_q=self.seq_len//2+self.pred_len,
													  seq_len_kv=self.seq_len,
													  modes=configs.modes,
													  mode_select_method=configs.mode_select)
		# Encoder
		enc_modes = int(min(configs.modes, configs.seq_len//2))
		dec_modes = int(min(configs.modes, (configs.seq_len//2+configs.pred_len)//2))
		print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

		self.encoder = Encoder(
			[
				EncoderLayer(
					AutoCorrelationLayer(
						encoder_self_att,
						configs.d_model, configs.n_heads),

					configs.d_model,
					configs.d_ff,
					moving_avg=configs.moving_avg,
					dropout=configs.dropout,
					activation=configs.activation
				) for l in range(configs.e_layers)
			],
			norm_layer=Automy_Layernorm(configs.d_model)
		)
		# Decoder
		self.decoder = Decoder(
			[
				DecoderLayer(
					AutoCorrelationLayer(
						decoder_self_att,
						configs.d_model, configs.n_heads),
					AutoCorrelationLayer(
						decoder_cross_att,
						configs.d_model, configs.n_heads),
					configs.d_model,
					configs.c_out,
					configs.d_ff,
					moving_avg=configs.moving_avg,
					dropout=configs.dropout,
					activation=configs.activation,
				)
				for l in range(configs.d_layers)
			],
			norm_layer=my_Layernorm(configs.d_model),
			projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
		)

	def forward(self, x_enc, _, __, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
		x_dec = torch.cat([x_enc[:, -horizons:, :], torch.zeros_like(x_enc[:, -horizons:, :]).float()], dim=1).float().to(device)
		# decomp init
		mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
		zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(device)  # cuda()
		seasonal_init, trend_init = self.decomp(x_enc)
		# decoder input
		trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
		#print("init", trend_init.shape)
		seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
		# enc
		enc_out = self.encProjection(x_enc) + self.encEmbedding
		enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
		# dec
		dec_out = self.decProjection(x_dec) + self.decEmbedding 
		seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
												 trend=trend_init)
		# final
		dec_out = trend_part + seasonal_part

		if self.output_attention:
			return dec_out[:, -self.pred_len:, :][:, :, 0]
		else:
			return dec_out[:, -self.pred_len:, :][:, :, 0]  # [B, L, D]