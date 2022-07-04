import logging

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        # self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        # x = self.layer_norm(x)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn)  # [n * b, l_q, l_k]
        attn = self.dropout(attn)  # [n * b, l_v, d]

        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)

        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        # output = self.layer_norm(output)

        return output, attn


class MapBasedMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.wq_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wk_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wv_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.weight_map = nn.Linear(2 * d_k, 1, bias=False)

        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()

        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.wq_node_transform(q).view(sz_b, len_q, n_head, d_k)

        k = self.wk_node_transform(k).view(sz_b, len_k, n_head, d_k)

        v = self.wv_node_transform(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        q = torch.unsqueeze(q, dim=2)  # [(n*b), lq, 1, dk]
        q = q.expand(q.shape[0], q.shape[1], len_k, q.shape[3])  # [(n*b), lq, lk, dk]

        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        k = torch.unsqueeze(k, dim=1)  # [(n*b), 1, lk, dk]
        k = k.expand(k.shape[0], len_q, k.shape[2], k.shape[3])  # [(n*b), lq, lk, dk]

        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x lq x lk

        ## Map based Attention
        # output, attn = self.attention(q, k, v, mask=mask)
        q_k = torch.cat([q, k], dim=3)  # [(n*b), lq, lk, dk * 2]
        attn = self.weight_map(q_k).squeeze(dim=3)  # [(n*b), lq, lk]

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn)  # [n * b, l_q, l_k]
        attn = self.dropout(attn)  # [n * b, l_q, l_k]

        # [n * b, l_q, l_k] * [n * b, l_v, d_v] >> [n * b, l_q, d_v]
        output = torch.bmm(attn, v)

        output = output.view(n_head, sz_b, len_q, d_v)

        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.act(self.fc(output)))
        output = self.layer_norm(output + residual)

        return output, attn


def expand_last_dim(x, num):
    view_size = list(x.size()) + [1]
    expand_size = list(x.size()) + [num]
    return x.view(view_size).expand(expand_size)


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        # init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])

        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())

        # self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)

        # torch.nn.init.xavier_normal_(self.dense.weight)

    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)

        return harmonic  # self.dense(harmonic)


class PosEncode(torch.nn.Module):
    def __init__(self, expand_dim, seq_len):
        super().__init__()

        self.pos_embeddings = nn.Embedding(num_embeddings=seq_len, embedding_dim=expand_dim)

    def forward(self, ts):
        # ts: [N, L]
        order = ts.argsort()
        ts_emb = self.pos_embeddings(order)
        return ts_emb


class EmptyEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super().__init__()
        self.expand_dim = expand_dim

    def forward(self, ts):
        out = torch.zeros_like(ts).float()
        out = torch.unsqueeze(out, dim=-1)
        out = out.expand(out.shape[0], out.shape[1], self.expand_dim)
        return out


class LSTMPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim, time_dim):
        super(LSTMPool, self).__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.edge_dim = edge_dim

        self.att_dim = feat_dim + edge_dim + time_dim

        self.act = torch.nn.ReLU()

        self.lstm = torch.nn.LSTM(input_size=self.att_dim,
                                  hidden_size=self.feat_dim,
                                  num_layers=1,
                                  batch_first=True)
        self.merger = MergeLayer(feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        # seq [B, N, D]
        # mask [B, N]
        seq_x = torch.cat([seq, seq_e, seq_t], dim=2)

        _, (hn, _) = self.lstm(seq_x)

        hn = hn[-1, :, :]  # hn.squeeze(dim=0)

        out = self.merger.forward(hn, src)
        return out, None


class MeanPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim):
        super(MeanPool, self).__init__()
        self.edge_dim = edge_dim
        self.feat_dim = feat_dim
        self.act = torch.nn.ReLU()
        self.merger = MergeLayer(edge_dim + feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        # seq [B, N, D]
        # mask [B, N]
        src_x = src
        seq_x = torch.cat([seq, seq_e], dim=2)  # [B, N, De + D]
        hn = seq_x.mean(dim=1)  # [B, De + D]
        output = self.merger(hn, src_x)
        return output, None


class AttnModel(torch.nn.Module):
    """Attention based temporal layers
    """

    def __init__(self, feat_dim, edge_dim, time_dim,
                 attn_mode='prod', n_head=2, dropout=0.1):
        """
        args:
          feat_dim: dim for the node features
          edge_dim: dim for the temporal edge features
          time_dim: dim for the time encoding
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          dropout: probability of dropping a neural.
        """
        super(AttnModel, self).__init__()

        self.feat_dim = feat_dim
        self.time_dim = time_dim

        self.edge_in_dim = (feat_dim + edge_dim + time_dim)
        self.model_dim = self.edge_in_dim
        # self.edge_fc = torch.nn.Linear(self.edge_in_dim, self.feat_dim, bias=False)

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)

        # self.act = torch.nn.ReLU()

        assert (self.model_dim % n_head == 0)
        self.logger = logging.getLogger(__name__)
        self.attn_mode = attn_mode

        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadAttention(n_head,
                                                        d_model=self.model_dim,
                                                        d_k=self.model_dim // n_head,
                                                        d_v=self.model_dim // n_head,
                                                        dropout=dropout)
            self.logger.info('Using scaled prod attention')

        elif attn_mode == 'map':
            self.multi_head_target = MapBasedMultiHeadAttention(n_head,
                                                                d_model=self.model_dim,
                                                                d_k=self.model_dim // n_head,
                                                                d_v=self.model_dim // n_head,
                                                                dropout=dropout)
            self.logger.info('Using map based attention')
        else:
            raise ValueError('attn_mode can only be prod or map')

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        """"Attention based temporal attention forward pass
        args:
          src: float Tensor of shape [B, D]
          src_t: float Tensor of shape [B, Dt], Dt == D
          seq: float Tensor of shape [B, N, D]
          seq_t: float Tensor of shape [B, N, Dt]
          seq_e: float Tensor of shape [B, N, De], De == D
          mask: boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.
        returns:
          output, weight
          output: float Tensor of shape [B, D]
          weight: float Tensor of shape [B, N]
        """

        src_ext = torch.unsqueeze(src, dim=1)  # src [B, 1, D]
        src_e_ph = torch.zeros_like(src_ext)
        q = torch.cat([src_ext, src_e_ph, src_t], dim=2)  # [B, 1, D + De + Dt] -> [B, 1, D]
        k = torch.cat([seq, seq_e, seq_t], dim=2)  # [B, 1, D + De + Dt] -> [B, 1, D]

        mask = torch.unsqueeze(mask, dim=2)  # mask [B, N, 1]
        mask = mask.permute([0, 2, 1])  # mask [B, 1, N]

        # # target-attention
        output, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask)  # output: [B, 1, D + Dt], attn: [B, 1, N]
        output = output[:, 0, :]
        attn = attn[:, 0, :]

        output = self.merger(output, src)
        return output, attn


class tDFS(torch.nn.Module):
    def __init__(self, n_feat, e_feat, attn_mode='prod', use_time='time', bfs_method='attn', path_agg='attn', paths_agg='mean',
                 num_layers=3, n_head=4, dropout=0.1, seq_len=None, alpha=None, ngh_finder=None):
        super(tDFS, self).__init__()

        self.attn_mode = attn_mode
        self.use_time = use_time
        self.bfs_method = bfs_method
        self.num_layers = num_layers
        self.n_head = n_head
        self.dropout = dropout
        self.seq_len = seq_len
        self.alpha = alpha

        self.ngh_finder = ngh_finder
        self.logger = logging.getLogger(__name__)
        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
        self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)))
        self.edge_raw_embed = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=True)
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)

        self.feat_dim = self.n_feat_th.shape[1]

        self.n_feat_dim = self.feat_dim
        self.e_feat_dim = self.feat_dim
        self.model_dim = self.feat_dim

        # bfs aggs
        self.bfs_attn_list = self.get_attn_list(self.bfs_method, num_layers)

        # dfs aggs
        self.path_agg = self.get_attn_list(path_agg, 1)[0]
        self.do_paths_attn = paths_agg == 'attn'
        if self.do_paths_attn:
            self.paths_agg = MultiHeadAttention(n_head,
                                                d_model=self.feat_dim,
                                                d_k=self.feat_dim // n_head,
                                                d_v=self.feat_dim // n_head,
                                                dropout=dropout)

        if use_time == 'time':
            self.logger.info('Using time encoding')
            self.time_encoder = TimeEncode(expand_dim=self.n_feat_th.shape[1])
        elif use_time == 'pos':
            assert (seq_len is not None)
            self.logger.info('Using positional encoding')
            self.time_encoder = PosEncode(expand_dim=self.n_feat_th.shape[1], seq_len=seq_len)
        elif use_time == 'empty':
            self.logger.info('Using empty encoding')
            self.time_encoder = EmptyEncode(expand_dim=self.n_feat_th.shape[1])
        else:
            raise ValueError('invalid time option!')

        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1)

    def get_attn_list(self, method, num_layers):
        if method == 'attn':
            attn_list = torch.nn.ModuleList([AttnModel(self.feat_dim,
                                                       self.feat_dim,
                                                       self.feat_dim,
                                                       attn_mode=self.attn_mode,
                                                       n_head=self.n_head,
                                                       dropout=self.dropout) for _ in range(num_layers)])
        elif method == 'lstm':
            attn_list = torch.nn.ModuleList([LSTMPool(self.feat_dim,
                                                      self.feat_dim,
                                                      self.feat_dim) for _ in range(num_layers)])
        elif method == 'mean':
            attn_list = torch.nn.ModuleList([MeanPool(self.feat_dim,
                                                      self.feat_dim) for _ in range(num_layers)])
        else:
            raise ValueError('invalid method value, use attn, lstm, or mean')

        return attn_list

    def forward(self, src_idx_l, target_idx_l, cut_time_l):

        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers)

        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)

        return score

    def contrast(self, src_idx_l, target_idx_l, background_idx_l, cut_time_l):
        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers)
        background_embed = self.tem_conv(background_idx_l, cut_time_l, self.num_layers)
        pos_score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        neg_score = self.affinity_score(src_embed, background_embed).squeeze(dim=-1)
        return pos_score.sigmoid(), neg_score.sigmoid()

    def tem_conv(self, src_idx_l, cut_time_l, curr_layers):
        assert (curr_layers >= 0)

        device = self.n_feat_th.device

        batch_size = len(src_idx_l)

        # src_node_batch_th = torch.from_numpy(src_idx_l).long().to(device)
        # cut_time_l_th = torch.from_numpy(cut_time_l).float().to(device)
        src_node_batch_th = src_idx_l
        cut_time_l_th = cut_time_l

        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)
        # query node always has the start time -> time span == 0
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th))
        src_node_feat = self.node_raw_embed(src_node_batch_th)

        if curr_layers == 0:
            return src_node_feat, None

        src_node_conv_feat, _ = self.tem_conv(src_idx_l, cut_time_l, curr_layers=curr_layers - 1)

        src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor(
                                                                                        src_idx_l.cpu().numpy(),
                                                                                        cut_time_l.cpu().numpy(),
                                                                                        num_neighbors=self.seq_len)

        src_ngh_node_batch = torch.from_numpy(src_ngh_node_batch).long().to(device)
        src_ngh_eidx_batch = torch.from_numpy(src_ngh_eidx_batch).long().to(device)
        src_ngh_t_batch = torch.from_numpy(src_ngh_t_batch).float().to(device)

        src_ngh_t_batch_th = cut_time_l.unsqueeze(-1) - src_ngh_t_batch

        # get previous layer's node features
        src_ngh_node_batch_flat = src_ngh_node_batch.flatten()  # reshape(batch_size, -1)
        src_ngh_t_batch_flat = src_ngh_t_batch.flatten()  # reshape(batch_size, -1)
        src_ngh_node_conv_feat, cache = self.tem_conv(src_ngh_node_batch_flat,
                                                      src_ngh_t_batch_flat,
                                                      curr_layers=curr_layers - 1)
        src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, self.seq_len, -1)

        # get edge time features and node features
        src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)
        src_ngn_edge_feat = self.edge_raw_embed(src_ngh_eidx_batch)

        # attention aggregation
        mask = src_ngh_node_batch == 0

        # BFS aggregation
        attn_m = self.bfs_attn_list[curr_layers - 1]
        bfs, _ = attn_m(src_node_conv_feat,
                        src_node_t_embed,
                        src_ngh_feat,
                        src_ngh_t_embed,
                        src_ngn_edge_feat,
                        mask)

        if cache is None:  # create first paths cache
            cache = [src_ngh_feat.unsqueeze(2), src_ngh_t_batch.unsqueeze(2), src_ngn_edge_feat.unsqueeze(2), mask.unsqueeze(2)]
        else:  # explode existing paths cache
            old_cache = [x.view(*src_ngh_feat.size()[:2], *x.size()[1:]) for x in cache]
            new_cache = [src_ngh_feat.unsqueeze(2), src_ngh_t_batch.unsqueeze(2), src_ngn_edge_feat.unsqueeze(2), mask.unsqueeze(2)]
            new_cache = [x.unsqueeze(2).repeat(1, 1, old_cache[0].size(2), *([1] * (x.dim()-2))) for i, x in enumerate(new_cache)]
            cache = [torch.cat([old, new], dim=3) for old, new in zip(old_cache, new_cache)]
            cache = [x.view(x.size(0), -1, *x.size()[3:]) for x in cache]

        # DFS aggregation
        if curr_layers == self.num_layers:  # if returned back to root call of the recursive function
            src_ngh_feat, src_ngh_t_batch, src_ngn_edge_feat, mask = cache

            src_ngh_t_batch_th = (cut_time_l.unsqueeze(-1).unsqueeze(-1) - src_ngh_t_batch).view(-1, src_ngh_t_batch.size(-1))
            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)

            dfs, _ = self.path_agg(src_node_conv_feat.unsqueeze(1).repeat(1, src_ngh_feat.size(1), 1).view(-1, *src_node_conv_feat.size()[1:]),
                                   src_node_t_embed.unsqueeze(1).repeat(1, src_ngh_feat.size(1), 1, 1).view(-1, *src_node_t_embed.size()[1:]),
                                   src_ngh_feat.view(-1, *src_ngh_feat.size()[2:]),
                                   src_ngh_t_embed,
                                   src_ngn_edge_feat.view(-1, *src_ngn_edge_feat.size()[2:]),
                                   mask.view(-1, *mask.size()[2:]))
            dfs = dfs.view(*src_ngh_feat.size()[:2], -1)

            if self.do_paths_attn:
                path_mask = mask.all(dim=-1)
                dfs, _ = self.paths_agg(q=bfs.unsqueeze(1), k=dfs, v=dfs, mask=path_mask.unsqueeze(1))
                dfs = dfs.squeeze()
            else: #  mean
                dfs = dfs.mean(dim=1)

            # merge the two representations
            return self.alpha * bfs + (1 - self.alpha) * dfs

        # if not root call of the recursive function, return back just the BFS representation and the path cache
        else:
            return bfs, cache
