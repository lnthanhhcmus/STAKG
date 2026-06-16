import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import TransformerConv
from .modules.module_visual import VisualModel, VisualConfig
from .modules.until_module import LayerNorm
from collections import namedtuple
import logging
logger = logging.getLogger(__name__)


Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

def pad_mask(src, r2l_trg, trg, pad_idx, video_mask):
    if not isinstance(src, tuple): 
        src_vid_mask = (video_mask != pad_idx).unsqueeze(1)
        enc_src_mask = src_vid_mask
        dec_src_mask = src_vid_mask
        src_mask = (enc_src_mask, dec_src_mask)  
    elif len(src) == 3:
        src_vid_mask = (video_mask != pad_idx).unsqueeze(1)
        src_object_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)
        src_rel_mask = (src[2][:, :, 0] != pad_idx).unsqueeze(1)

        enc_src_mask = (src_vid_mask, src_object_mask, src_rel_mask)
        dec_src_mask = src_vid_mask
        src_mask = (enc_src_mask, dec_src_mask)
    elif len(src) == 2:
        src_vid_mask = (video_mask != pad_idx).unsqueeze(1)
        src_rel_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)

        enc_src_mask = (src_vid_mask, src_rel_mask)
        dec_src_mask = src_vid_mask
        src_mask = (enc_src_mask, dec_src_mask)        


    if trg is not None:
        if isinstance(src_mask, tuple):
            trg_mask = (trg != pad_idx).unsqueeze(1) & subsequent_mask(trg.size(1)).type_as(src_vid_mask.data)
            r2l_pad_mask = (r2l_trg != pad_idx).unsqueeze(1).type_as(src_vid_mask.data)
            r2l_trg_mask = r2l_pad_mask & subsequent_mask(r2l_trg.size(1)).type_as(src_vid_mask.data)
            return src_mask, r2l_pad_mask, r2l_trg_mask, trg_mask
        else:
            trg_mask = (trg != pad_idx).unsqueeze(1) & subsequent_mask(trg.size(1)).type_as(src_mask.data)
            r2l_pad_mask = (r2l_trg != pad_idx).unsqueeze(1).type_as(src_mask.data)
            r2l_trg_mask = r2l_pad_mask & subsequent_mask(r2l_trg.size(1)).type_as(src_mask.data)
            return src_mask, r2l_pad_mask, r2l_trg_mask, trg_mask

    else:
        return src_mask

def subsequent_mask(size):
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return (torch.from_numpy(mask) == 0).cuda()

def self_attention(query, key, value, dropout=None, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask.cuda()
        scores = scores.masked_fill(mask == 0, -1e9)
    self_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        self_attn = dropout(self_attn)
    return torch.matmul(self_attn, value), self_attn
  
class FeatEmbedding(nn.Module):
    def __init__(self, d_feat, d_model, dropout):
        super(FeatEmbedding, self).__init__()
        self.video_embeddings = nn.Sequential(
            LayerNorm(d_feat),
            nn.Linear(d_feat, d_model))

    def forward(self, x):
        return self.video_embeddings(x)

class NormalizeVideo(nn.Module):
    def __init__(self, video_dim):
        super(NormalizeVideo, self).__init__()
        self.visual_norm2d = LayerNorm(video_dim)

    def forward(self, video):
        video = torch.as_tensor(video).float()
        video = video.view(-1, video.shape[-2], video.shape[-1])
        video = self.visual_norm2d(video)
        return video
        
class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TextEmbedding, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module): # adjust max_len
    def __init__(self, dim, dropout, max_len=5000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.drop_out = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):

        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.drop_out(emb)
        return emb


class MultiHeadAttention(nn.Module):
    def __init__(self, head, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert (d_model % head == 0)
        self.d_k = d_model // head
        self.head = head
        self.d_model = d_model
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.attn = None
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batch = query.size(0)

        query = self.linear_query(query).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 32, 64]
        key = self.linear_key(key).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]
        value = self.linear_value(value).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]

        x, self.attn = self_attention(query, key, value, dropout=self.dropout, mask=mask)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)

        return self.linear_out(x)
 

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.LayerNorm = LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.LayerNorm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.LayerNorm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return self.dropout(self.LayerNorm(x + sublayer(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, attn, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer_connection[0](x, lambda x: self.attn(x, x, x, mask))
        return self.sublayer_connection[1](x, self.feed_forward)


class EncoderLayerNoAttention(nn.Module):
    def __init__(self, size, attn, feed_forward, dropout=0.1):
        super(EncoderLayerNoAttention, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        return self.sublayer_connection[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    def __init__(self, size, attn, feed_forward, sublayer_num, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(size, dropout), sublayer_num)

    def forward(self, x, memory, src_mask, trg_mask, r2l_memory=None, r2l_trg_mask=None):
        x = self.sublayer_connection[0](x, lambda x: self.attn(x, x, x, trg_mask))
        x = self.sublayer_connection[1](x, lambda x: self.attn(x, memory, memory, src_mask))

        if r2l_memory is not None:
            x = self.sublayer_connection[-2](x, lambda x: self.attn(x, r2l_memory, r2l_memory, r2l_trg_mask))

        return self.sublayer_connection[-1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, n, encoder_layer):
        super(Encoder, self).__init__()
        self.encoder_layer = clones(encoder_layer, n)

    def forward(self, x, src_mask):
        for layer in self.encoder_layer:
            x = layer(x, src_mask)
        return x


class R2L_Decoder(nn.Module):
    def __init__(self, n, decoder_layer):
        super(R2L_Decoder, self).__init__()
        self.decoder_layer = clones(decoder_layer, n)

    def forward(self, x, memory, src_mask, r2l_trg_mask):
        for layer in self.decoder_layer:
            x = layer(x, memory, src_mask, r2l_trg_mask)
        return x


class L2R_Decoder(nn.Module):
    def __init__(self, n, decoder_layer):
        super(L2R_Decoder, self).__init__()
        self.decoder_layer = clones(decoder_layer, n)

    def forward(self, x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask):
        for layer in self.decoder_layer:
            x = layer(x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask)
        return x

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)
    
class TransC(nn.Module):
    def __init__(self, node_feat_dim, d_model, edge_dim, heads=4, project_edge_dim=None, more_skip=True, last_average=False, beta=True):
        super().__init__()
        self.lp = nn.Linear(node_feat_dim, d_model)
        self.more_skip = more_skip
        self.project_edge_dim = project_edge_dim
        if self.project_edge_dim is not None:
            self.lp_edge_attr = nn.Linear(edge_dim, project_edge_dim)
            edge_dim = project_edge_dim
        
        self.conv1 = TransformerConv(d_model, int(d_model/heads), heads, edge_dim=edge_dim, aggr='mean', beta=beta)
        
        self.conv2 = TransformerConv(d_model, int(d_model/heads), heads, edge_dim=edge_dim, aggr='mean', beta=beta)
        
        if last_average:
            self.conv3 = TransformerConv(d_model, d_model, heads, concat=False, edge_dim=edge_dim, aggr='mean', beta=beta)
        else:
            self.conv3 = TransformerConv(d_model, int(d_model/heads), heads, edge_dim=edge_dim, aggr='mean', beta=beta)

    def forward(self, data):
        x = self.lp(data.x)
        if self.project_edge_dim is not None:
            e = F.relu(self.lp_edge_attr(data.edge_attr))
        else:
            e = data.edge_attr
        if self.more_skip:
            x = F.relu(x + self.conv1(x, data.edge_index, e))
            x = F.relu(x + self.conv2(x, data.edge_index, e))
            x = F.relu(x + self.conv3(x, data.edge_index, e))
        else:
            x = F.relu(self.conv1(x, data.edge_index, e))
            x = F.relu(self.conv2(x, data.edge_index, e))
            x = F.relu(self.conv3(x, data.edge_index, e))
        return x
    
class STGraphEncoder(nn.Module):
    def __init__(self, visual_model_name, heads_type, state_dict=None, cache_dir=None, type_vocab_size=2, task_config=None):
        super(STGraphEncoder, self).__init__()
        
        if task_config and not hasattr(task_config, "local_rank"):
            task_config.__dict__["local_rank"] = 0
        elif task_config and task_config.local_rank == -1:
            task_config.local_rank = 0
        self.task_config = task_config
        
        self.visual_config, _ = VisualConfig.get_config(
            visual_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        assert self.task_config.max_frames <= self.visual_config.max_position_embeddings
        self.visual_config = update_attr("visual_config", self.visual_config, "num_hidden_layers",
                                         self.task_config, "visual_num_hidden_layers")
        self.visual_config = update_attr("visual_config", self.visual_config, "num_attention_heads",
                                         self.task_config, heads_type)
        # self.visual_config = update_attr("visual_config", self.visual_config, "hidden_size",
        #                                  self.task_config, "hidden_size")
        # self.visual_config = update_attr("visual_config", self.visual_config, "vocab_size",
        #                                  self.task_config, "vocab_size")

        self.visual = VisualModel(self.visual_config)
        self.normalize_video = NormalizeVideo(task_config.video_dim)
        
        self.check = False
        if self.visual_config.hidden_size != task_config.d_model:
            self.check = True
            self.lp = nn.Linear(self.visual_config.hidden_size, task_config.d_model)

        # Áp dụng trọng số pretrain nếu có
        if state_dict is not None:
            self.init_preweight(state_dict)

        self.apply(self.init_weights)

    
