from statistics import mean
import torch as t
from torch import nn
import torch.nn.functional as F
from params import args
from Utils.utils import *
import numpy as np
import scipy
import torch_sparse as ts
from data_handler import temHandler

init = nn.init.xavier_uniform_
uniform_init = nn.init.uniform_


import networkx as nx



class SpanningTree(nn.Module):
    def __init__(self, adj):
        super(SpanningTree, self).__init__()
        self.old_adj = adj

    def to_graph_list(self, adj):
        self.graph_list = []
        rows, cols, vals = adj.coo()    
        for i in range(rows.shape[0]):
            r = rows[i].item()
            c = cols[i].item()
            v = vals[i].item()
            if r <= c:
                self.graph_list.append((r,c,v))
        return self.graph_list

    def to_sparse_adj(self, shape, T):
        rows = []
        cols = []
        vals = []

        for tup in T:
            rows.append(tup[0])
            cols.append(tup[1])
            vals.append(tup[2]['weight'])

            rows.append(tup[1])
            cols.append(tup[0])
            vals.append(tup[2]['weight'])
        rows = t.tensor(rows)
        cols = t.tensor(cols)
        vals = t.tensor(vals)

        return ts.SparseTensor(row=rows, col=cols, value = vals, sparse_sizes= shape).cuda()

    def forward(self, adj):
        if adj == self.old_adj:
            return self.new_adj

        self.old_adj = adj
        self.to_graph_list(adj)
        G = nx.Graph()
        G.add_weighted_edges_from(self.graph_list)  
        T = nx.minimum_spanning_tree(G)    
        self.new_adj =  self.to_sparse_adj(adj.sizes(), T.edges(data=True))

        return self.new_adj



class SpAdjDropEdge(nn.Module):
    def __init__(self):
        super(SpAdjDropEdge, self).__init__()

    def forward(self, adj, keepRate):
        if keepRate == 1.0:
            return adj

        row, col, val = adj.coo()
        edgeNum = val.size()

        mask = ((t.rand(edgeNum) + keepRate).floor()).type(t.bool)
        newVals = val[mask] / keepRate  #  v1
        # newVals = val[mask]   #  v2

        newRow = row[mask]
        newCol = col[mask]

        return ts.SparseTensor(row=newRow, col=newCol, value = newVals, sparse_sizes= adj.sizes())


class HGNNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=False, act=None):
        super(HGNNLayer, self).__init__()
        # self.act = nn.LeakyReLU(negative_slope=args.leaky)
        self.W1 = nn.Parameter(t.eye(in_feat, out_feat).cuda() )
        self.bias1 = nn.Parameter(t.zeros( 1, out_feat).cuda() )

        self.W2 = nn.Parameter(t.eye(out_feat, in_feat).cuda())
        self.bias2 = nn.Parameter(t.zeros( 1, in_feat).cuda())


        if act == 'identity' or act is None:
            self.act = None
        elif act == 'leaky':
            self.act = nn.LeakyReLU(negative_slope=args.leaky)
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'relu6':
            self.act = nn.ReLU6()
        else:
            raise Exception('Error')

    def forward(self, embeds):
        # if self.act is None:
        #     # return self.linear(embeds)
        #     return  embeds @ self.W 
        out1 = self.act(  embeds @ self.W1 + self.bias1 )
        out2 = self.act(  out1 @ self.W2 + self.bias2  )
        return out2


    
    
class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()
		# self.act = nn.LeakyReLU(negative_slope=args.leaky)

	def forward(self, adj, embeds):
		# return (t.spmm(adj, embeds))
		return adj.matmul(embeds)


# ===========================================================================
# Shared utility functions for influence encoders
# ===========================================================================

def graph_propagate(adj, embeds, num_layers):
    """Multi-hop sparse graph propagation.  Returns final-layer embeddings."""
    x = embeds
    for _ in range(num_layers):
        x = adj.matmul(x)
    return x


def compute_influence_signal(ini_embeds, ts_drp_adj, ts_pk_adj, num_layers):
    """
    influence_signal = H_bar - E_bar_weak
    H_bar  : propagated embeddings over the influence (dropped-edge) graph
    E_bar_weak : propagated embeddings over the residual (kept-edge) graph
    """
    H_bar = graph_propagate(ts_drp_adj, ini_embeds, num_layers)
    E_bar_weak = graph_propagate(ts_pk_adj, ini_embeds, num_layers)
    return H_bar - E_bar_weak


def mean_pool_sparse(adj, embeds):
    """Mean-pool over edge endpoints of a sparse adjacency."""
    row, col, _ = adj.coo()
    if row.numel() == 0:
        return t.zeros(1, embeds.size(1), device=embeds.device)
    edge_feats = (embeds[row] + embeds[col]) / 2.0
    return edge_feats.mean(dim=0, keepdim=True)


def max_pool_sparse(adj, embeds):
    """Max-pool over edge endpoints of a sparse adjacency."""
    row, col, _ = adj.coo()
    if row.numel() == 0:
        return t.zeros(1, embeds.size(1), device=embeds.device)
    edge_feats = (embeds[row] + embeds[col]) / 2.0
    return edge_feats.max(dim=0, keepdim=True)[0]


def _freeze_base_model(model):
    """Freeze the base recommender's parameters."""
    if hasattr(model, 'uEmbeds') and hasattr(model, 'iEmbeds'):
        model.uEmbeds.detach()
        model.uEmbeds.requires_grad = False
        model.iEmbeds.detach()
        model.iEmbeds.requires_grad = False
    else:
        model.ini_embeds.detach()
        model.ini_embeds.requires_grad = False


def _compute_ssl_loss(gcnEmbedsLst, hyperEmbedsLst, ancs, poss):
    """Contrastive SSL loss between GCN layers and hyperplane-augmented layers."""
    ssl = 0
    for i in range(len(gcnEmbedsLst)):
        e1 = gcnEmbedsLst[i].detach()
        e2 = hyperEmbedsLst[i]
        ssl += contrastLoss(e1[:args.user], e2[:args.user], t.unique(ancs), args.hyper_temp) \
             + contrastLoss(e1[args.user:], e2[args.user:], t.unique(poss), args.hyper_temp)
    return ssl


def _base_forward_components(edge_embeds, gcnLayer, edgeDropper,
                             ts_drp_adj, ts_pk_adj,
                             ini_embeds, fnl_embeds, withdraw_rate):
    """Shared forward components: GNN layers, edge propagation, withdrawal,
    and influence signal.  Returns a dict of intermediate tensors."""
    lats = [edge_embeds]
    gnnLats, hyperLats = [], []
    for _ in range(args.gnn_layer):
        temEmbeds = gcnLayer(edgeDropper(ts_drp_adj, 1.0), lats[-1])
        hyperemb = gcnLayer(edgeDropper(ts_drp_adj, 0.95), lats[-1])
        gnnLats.append(temEmbeds)
        hyperLats.append(hyperemb)
        lats.append(temEmbeds)
    edge_embed = sum(lats)

    edges_embeddings = [edge_embed]
    for _ in range(args.unlearn_layer):
        edges_embeddings.append(ts_pk_adj.matmul(edges_embeddings[-1]))

    withdraw = [fnl_embeds * withdraw_rate]
    for _ in range(args.gnn_layer):
        withdraw.append(ts_drp_adj.matmul(withdraw[-1]))

    influence_sig = compute_influence_signal(ini_embeds, ts_drp_adj, ts_pk_adj,
                                             args.gnn_layer)
    return dict(
        edge_out=edges_embeddings[-1],
        withdraw=withdraw[-1],
        gnnLats=gnnLats,
        hyperLats=hyperLats,
        influence_signal=influence_sig,
    )


def _sparse_softmax(scores, row, N):
    """Per-source-node softmax of edge scores for sparse graphs."""
    scores = t.clamp(scores, min=-20.0, max=20.0)
    exp_s = t.exp(scores)
    denom = t.zeros(N, device=scores.device)
    denom.scatter_add_(0, row, exp_s)
    return exp_s / (denom[row] + 1e-10)


class LocalGraphAttention(nn.Module):
    """Multi-head GAT-style attention restricted to edges of a sparse graph."""

    def __init__(self, embed_dim, num_heads):
        super(LocalGraphAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        self.W_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.W_out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, embeds, adj):
        """
        embeds : (N, D)
        adj    : torch_sparse.SparseTensor defining the neighbourhood
        Returns: (N, D)  attended embeddings
        """
        N, D = embeds.shape
        H, d = self.num_heads, self.head_dim
        row, col, _ = adj.coo()

        if row.numel() == 0:
            return t.zeros_like(embeds)

        qkv = self.W_qkv(embeds)              # (N, 3D)
        Q, K, V = qkv.split(D, dim=-1)        # each (N, D)
        Q = Q.view(N, H, d)
        K = K.view(N, H, d)
        V = V.view(N, H, d)

        q_src = Q[row]                         # (E, H, d)
        k_dst = K[col]                         # (E, H, d)
        v_dst = V[col]                         # (E, H, d)

        attn = (q_src * k_dst).sum(-1) / (d ** 0.5)   # (E, H)
        attn = self.leaky(attn)

        out_heads = []
        for h in range(H):
            w = _sparse_softmax(attn[:, h], row, N)    # (E,)
            wv = v_dst[:, h, :] * w.unsqueeze(-1)      # (E, d)
            agg = t.zeros(N, d, device=embeds.device)
            agg.scatter_add_(0, row.unsqueeze(-1).expand_as(wv), wv)
            out_heads.append(agg)

        return self.W_out(t.cat(out_heads, dim=-1))    # (N, D)


class GraphUnlearning(nn.Module):
    def __init__(self, handler, model, ini_embeds, fnl_embeds):
        super(GraphUnlearning, self).__init__()        
        edges_num = handler.ts_ori_adj.nnz()
        self.edge_embeds1 = nn.Parameter(t.zeros(args.user+ args.item, args.latdim).cuda())
        # self.edge_embeds2 = nn.Parameter(t.zeros(args.user+ args.item, args.latdim).cuda())

        self.mlp_layers = nn.Sequential(*[FeedForwardLayer(args.latdim, args.latdim, act=args.act) for i in range(args.layer_mlp)])
        # self.layer_norm = nn.LayerNorm(args.latdim)
        self.act = nn.LeakyReLU(negative_slope=args.leaky)

        if args.withdraw_rate_init == 1:
            self.withdraw_rate = nn.Parameter(t.ones(args.user+args.item, 1) * args.lr * 2)            
        else:
            self.withdraw_rate = nn.Parameter(t.zeros(args.user+args.item, 1) * args.lr * 10)

        self.edgeDropper = SpAdjDropEdge()
        self.gcnLayer = GCNLayer()

        self.ini_embeds = ini_embeds.detach()
        self.fnl_embeds = fnl_embeds.detach()

        self.ini_embeds.requires_grad = False
        self.fnl_embeds.requires_grad = False
        self.model  = model


        if hasattr(self.model, "uEmbeds") and  hasattr(self.model, "iEmbeds"):  
            self.model.uEmbeds.detach()
            self.model.uEmbeds.requires_grad = False    
            self.model.iEmbeds.detach()
            self.model.iEmbeds.requires_grad = False                
        else:
            self.model.ini_embeds.detach()
            self.model.ini_embeds.requires_grad = False            


    def forward(self, ori_adj, ts_pk_adj ,mask, ts_drp_adj):        
        lats = [self.edge_embeds1 ]
        gnnLats = []        
        hyperLats = []           

        for _ in range(args.gnn_layer):            
            temEmbeds = self.gcnLayer(self.edgeDropper(ts_drp_adj, 1.0), lats[-1])
            hyperemb = self.gcnLayer(self.edgeDropper(ts_drp_adj, 0.95), lats[-1])

            gnnLats.append(temEmbeds)
            hyperLats.append(hyperemb)
            lats.append(  temEmbeds )
        edge_embed = sum(lats)
                
        edges_embeddings = [edge_embed]
        for _ in range(args.unlearn_layer):
            edges_embeddings.append(ts_pk_adj.matmul(edges_embeddings[-1]))
        
        withdraw =  [ self.fnl_embeds * self.withdraw_rate ]        
        for _ in range(args.gnn_layer):
            withdraw.append(ts_drp_adj.matmul(withdraw[-1]))

        delta_emb = - args.overall_withdraw_rate* withdraw[-1] +  edges_embeddings[-1]
        
        for i, layer in enumerate(self.mlp_layers):
            delta_emb = layer(delta_emb)

        tuned_emb = self.ini_embeds + delta_emb

        return tuned_emb, gnnLats, hyperLats  

    def outforward(self, ori_adj, ts_pk_adj ,mask, ts_drp_adj):
        self.model.training = False
        tuned_emb , _ , _ = self.forward(ori_adj, ts_pk_adj ,mask, ts_drp_adj)
        out_emb = self.model.forward(ts_pk_adj, tuned_emb, keepRate=1.0)        
        usr_embeds, itm_embeds = out_emb[:2]
        
        return usr_embeds, itm_embeds
    

    def out_all_layer(self, ori_adj, ts_pk_adj ,mask, ts_drp_adj, layer=0):
        self.model.training = False
        tuned_emb, _, _ = self.forward(ori_adj, ts_pk_adj ,mask, ts_drp_adj)
        all_embs, out_emb = self.model.forward(ts_pk_adj, tuned_emb, all_layer=True)  

        if layer == -2:
            tuned_emb[:args.user], tuned_emb[args.user:]
        elif layer == -1:
            return out_emb[:args.user], out_emb[args.user:]
        else:
            return out_emb[layer][:args.user], out_emb[layer][args.user:]
            

    def cal_loss(self, batch_data, ori_adj, ts_pk_adj , mask, ts_drp_adj, drp_edges,  pk_edges=None):
        ancs, poss, negs = list(map(lambda x: x.long(), batch_data))

        self.model.training = True
        tuned_emb,  gcnEmbedsLst, hyperEmbedsLst  = self.forward(ori_adj, ts_pk_adj, mask, ts_drp_adj)
        out_emb = self.model.forward(ts_pk_adj, tuned_emb)
        
        usr_embeds, itm_embeds = out_emb[:2]


        base_loss, loss_dict =  self.model.cal_loss(batch_data,  tuned_emb=tuned_emb , ori_adj=ori_adj, ts_pk_adj=ts_pk_adj , mask=mask, ts_drp_adj=ts_drp_adj, drp_edges=drp_edges,  pk_edges=None)

        if args.unlearn_type =='v1':
            unlearn_loss = cal_neg_aug_v1(usr_embeds[drp_edges[0]],  itm_embeds[drp_edges[1]])
        elif args.unlearn_type =='v2':
            unlearn_loss = cal_neg_aug_v2(usr_embeds[drp_edges[0]],  itm_embeds[drp_edges[1]])

        tar_fnl_uEmbeds, tar_fnl_iEmbeds  = self.fnl_embeds[ :args.user].detach(), self.fnl_embeds[args.user: ].detach()
            
        if args.fineTune:
            loss_dict['unlearn_loss'] = unlearn_loss
            # loss_dict['align_loss'] = align_loss
            align_loss= cal_positive_pred_align_v2(usr_embeds[ancs], tar_fnl_uEmbeds[ancs], itm_embeds[poss], tar_fnl_iEmbeds[poss],   cal_l2_distance, temp=args.align_temp)
            loss_dict['align_loss'] =  align_loss
            loss_dict['unlearn_ssl'] = t.tensor(0.)
            return base_loss + args.unlearn_wei * unlearn_loss + args.align_wei*align_loss ,    loss_dict
            # return base_loss + args.unlearn_wei * unlearn_loss  ,    loss_dict


        

        if not args.fineTune:
            if args.align_type == 'v2':
                align_loss = cal_positive_pred_align_v2(usr_embeds[ancs], tar_fnl_uEmbeds[ancs], itm_embeds[poss], tar_fnl_iEmbeds[poss],   cal_l2_distance, temp=args.align_temp)
            elif args.align_type == 'v3':
                align_loss = cal_positive_pred_align_v3(usr_embeds[ancs], tar_fnl_uEmbeds[ancs], itm_embeds[poss], tar_fnl_iEmbeds[poss],   cal_l2_distance, temp=args.align_temp)        

        sslLoss = 0
        for i in range(args.gnn_layer):
            embeds1 = gcnEmbedsLst[i].detach()
            embeds2 = hyperEmbedsLst[i]
            sslLoss += contrastLoss(embeds1[:args.user], embeds2[:args.user], t.unique(ancs), args.hyper_temp) + contrastLoss(embeds1[args.user:], embeds2[args.user:], t.unique(poss), args.hyper_temp)
  
        loss = args.unlearn_wei * unlearn_loss  + args.align_wei*align_loss + base_loss + args.unlearn_ssl*sslLoss
        loss_dict['unlearn_loss'] = unlearn_loss
        loss_dict['align_loss'] = align_loss
        loss_dict['unlearn_ssl'] = sslLoss
                

        return loss, loss_dict

    # (self.handler.ts_ori_adj, self.handler.ts_pk_adj, self.handler.mask,  self.handler.ts_drp_adj, usrs, trn_mask)
    def full_predict(self, ori_adj, ts_pk_adj, mask, ts_drp_adj , usrs, trn_mask):

        self.model.training = False
        usr_embeds, itm_embeds = self.outforward(ori_adj, ts_pk_adj, mask, ts_drp_adj)

        pck_usr_embeds = usr_embeds[usrs]
        full_preds = pck_usr_embeds @ itm_embeds.T
        full_preds = full_preds * (1 - trn_mask) - trn_mask * 1e8
        return full_preds


# ---------------------------------------------------------------------------
# Autoencoder Influence Encoder  (TRUE graph autoencoder)
# ---------------------------------------------------------------------------
class AutoencoderGraphUnlearning(nn.Module):
    """
    Graph autoencoder: a GCN encoder produces latent Z from the influence
    graph (A_delta = ts_drp_adj), and a decoder reconstructs A_delta via
    A_hat = sigmoid(Z @ Z.T).  The reconstruction loss keeps the latent
    space faithful to the influence topology.
    """

    def __init__(self, handler, model, ini_embeds, fnl_embeds):
        super(AutoencoderGraphUnlearning, self).__init__()
        self.edge_embeds1 = nn.Parameter(t.zeros(args.user + args.item, args.latdim).cuda())

        bottleneck = args.latdim // 2
        # GCN encoder on influence graph
        self.enc_gcn1 = GCNLayer()
        self.enc_gcn2 = GCNLayer()
        self.enc_linear = nn.Linear(args.latdim, bottleneck)
        self.enc_act = nn.LeakyReLU(negative_slope=args.leaky)
        # Projection back to embedding dim
        self.dec_linear = nn.Linear(bottleneck, args.latdim)

        self.mlp_layers = nn.Sequential(
            *[FeedForwardLayer(args.latdim, args.latdim, act=args.act) for _ in range(args.layer_mlp)]
        )
        self.act = nn.LeakyReLU(negative_slope=args.leaky)

        if args.withdraw_rate_init == 1:
            self.withdraw_rate = nn.Parameter(t.ones(args.user + args.item, 1) * args.lr * 2)
        else:
            self.withdraw_rate = nn.Parameter(t.zeros(args.user + args.item, 1) * args.lr * 10)

        self.edgeDropper = SpAdjDropEdge()
        self.gcnLayer = GCNLayer()

        self.ini_embeds = ini_embeds.detach()
        self.fnl_embeds = fnl_embeds.detach()
        self.ini_embeds.requires_grad = False
        self.fnl_embeds.requires_grad = False
        self.model = model
        _freeze_base_model(self.model)

    # ---- graph autoencoder reconstruction loss (sampled) ----
    def _adj_reconstruction_loss(self, Z, adj):
        """loss_rec = ||A_delta - A_hat||^2  (sampled edges + negative edges)"""
        row, col, _ = adj.coo()
        E = row.size(0)
        if E == 0:
            return t.tensor(0.0, device=Z.device)
        N = Z.size(0)
        # positive edges in A_delta
        pos_scores = t.sigmoid((Z[row] * Z[col]).sum(-1))
        pos_targets = t.ones_like(pos_scores)
        # negative samples (random non-edges)
        neg_row = t.randint(0, N, (E,), device=Z.device)
        neg_col = t.randint(0, N, (E,), device=Z.device)
        neg_scores = t.sigmoid((Z[neg_row] * Z[neg_col]).sum(-1))
        neg_targets = t.zeros_like(neg_scores)
        all_scores = t.cat([pos_scores, neg_scores])
        all_targets = t.cat([pos_targets, neg_targets])
        return F.mse_loss(all_scores, all_targets)

    def forward(self, ori_adj, ts_pk_adj, mask, ts_drp_adj):
        comp = _base_forward_components(
            self.edge_embeds1, self.gcnLayer, self.edgeDropper,
            ts_drp_adj, ts_pk_adj,
            self.ini_embeds, self.fnl_embeds, self.withdraw_rate)
        gnnLats = comp['gnnLats']
        hyperLats = comp['hyperLats']
        influence_sig = comp['influence_signal']

        # GCN-encoder on influence graph
        h = self.enc_act(self.enc_gcn1(ts_drp_adj, influence_sig))
        h = self.enc_act(self.enc_gcn2(ts_drp_adj, h))
        Z = self.enc_linear(h)                      # (N, bottleneck)
        self._last_Z = Z
        self._last_drp_adj = ts_drp_adj

        Z_proj = self.dec_linear(Z)                  # (N, D)

        delta_emb = -args.overall_withdraw_rate * comp['withdraw'] + comp['edge_out'] + Z_proj

        for layer in self.mlp_layers:
            delta_emb = layer(delta_emb)

        tuned_emb = self.ini_embeds + delta_emb
        return tuned_emb, gnnLats, hyperLats

    def outforward(self, ori_adj, ts_pk_adj, mask, ts_drp_adj):
        self.model.training = False
        tuned_emb, _, _ = self.forward(ori_adj, ts_pk_adj, mask, ts_drp_adj)
        out_emb = self.model.forward(ts_pk_adj, tuned_emb, keepRate=1.0)
        return out_emb[0], out_emb[1]

    def cal_loss(self, batch_data, ori_adj, ts_pk_adj, mask, ts_drp_adj, drp_edges, pk_edges=None):
        ancs, poss, negs = list(map(lambda x: x.long(), batch_data))
        self.model.training = True
        tuned_emb, gcnEmbedsLst, hyperEmbedsLst = self.forward(ori_adj, ts_pk_adj, mask, ts_drp_adj)
        out_emb = self.model.forward(ts_pk_adj, tuned_emb)
        usr_embeds, itm_embeds = out_emb[:2]

        base_loss, loss_dict = self.model.cal_loss(
            batch_data, tuned_emb=tuned_emb, ori_adj=ori_adj, ts_pk_adj=ts_pk_adj,
            mask=mask, ts_drp_adj=ts_drp_adj, drp_edges=drp_edges, pk_edges=None)

        if args.unlearn_type == 'v1':
            unlearn_loss = cal_neg_aug_v1(usr_embeds[drp_edges[0]], itm_embeds[drp_edges[1]])
        elif args.unlearn_type == 'v2':
            unlearn_loss = cal_neg_aug_v2(usr_embeds[drp_edges[0]], itm_embeds[drp_edges[1]])

        tar_fnl_uEmbeds = self.fnl_embeds[:args.user].detach()
        tar_fnl_iEmbeds = self.fnl_embeds[args.user:].detach()

        recon_loss = self._adj_reconstruction_loss(self._last_Z, self._last_drp_adj)

        if args.fineTune:
            align_loss = cal_positive_pred_align_v2(
                usr_embeds[ancs], tar_fnl_uEmbeds[ancs], itm_embeds[poss], tar_fnl_iEmbeds[poss],
                cal_l2_distance, temp=args.align_temp)
            loss_dict['unlearn_loss'] = unlearn_loss
            loss_dict['align_loss'] = align_loss
            loss_dict['unlearn_ssl'] = recon_loss
            return (base_loss + args.unlearn_wei * unlearn_loss
                    + args.align_wei * align_loss
                    + args.lambda_rec * recon_loss), loss_dict

        if args.align_type == 'v2':
            align_loss = cal_positive_pred_align_v2(
                usr_embeds[ancs], tar_fnl_uEmbeds[ancs], itm_embeds[poss], tar_fnl_iEmbeds[poss],
                cal_l2_distance, temp=args.align_temp)
        elif args.align_type == 'v3':
            align_loss = cal_positive_pred_align_v3(
                usr_embeds[ancs], tar_fnl_uEmbeds[ancs], itm_embeds[poss], tar_fnl_iEmbeds[poss],
                cal_l2_distance, temp=args.align_temp)

        sslLoss = _compute_ssl_loss(gcnEmbedsLst, hyperEmbedsLst, ancs, poss)

        loss = (args.unlearn_wei * unlearn_loss + args.align_wei * align_loss
                + base_loss + args.unlearn_ssl * sslLoss
                + args.lambda_rec * recon_loss)
        loss_dict['unlearn_loss'] = unlearn_loss
        loss_dict['align_loss'] = align_loss
        loss_dict['unlearn_ssl'] = sslLoss
        return loss, loss_dict

    def full_predict(self, ori_adj, ts_pk_adj, mask, ts_drp_adj, usrs, trn_mask):
        self.model.training = False
        usr_embeds, itm_embeds = self.outforward(ori_adj, ts_pk_adj, mask, ts_drp_adj)
        pck_usr_embeds = usr_embeds[usrs]
        full_preds = pck_usr_embeds @ itm_embeds.T
        full_preds = full_preds * (1 - trn_mask) - trn_mask * 1e8
        return full_preds


# ---------------------------------------------------------------------------
# Attention-based Influence Encoder  (LOCAL neighbour-based attention)
# ---------------------------------------------------------------------------
class AttentionGraphUnlearning(nn.Module):
    """
    Multi-head GAT-style attention restricted to edges in the influence
    graph (A_delta = ts_drp_adj).  Attention is normalised per node via
    softmax over neighbours, with a residual connection and LayerNorm.
    """

    def __init__(self, handler, model, ini_embeds, fnl_embeds):
        super(AttentionGraphUnlearning, self).__init__()
        self.edge_embeds1 = nn.Parameter(t.zeros(args.user + args.item, args.latdim).cuda())

        num_heads = max(1, args.latdim // 32)  # e.g. 4 for dim=128
        self.local_attn = LocalGraphAttention(args.latdim, num_heads)
        self.attn_norm = nn.LayerNorm(args.latdim)

        self.mlp_layers = nn.Sequential(
            *[FeedForwardLayer(args.latdim, args.latdim, act=args.act) for _ in range(args.layer_mlp)]
        )
        self.act = nn.LeakyReLU(negative_slope=args.leaky)

        if args.withdraw_rate_init == 1:
            self.withdraw_rate = nn.Parameter(t.ones(args.user + args.item, 1) * args.lr * 2)
        else:
            self.withdraw_rate = nn.Parameter(t.zeros(args.user + args.item, 1) * args.lr * 10)

        self.edgeDropper = SpAdjDropEdge()
        self.gcnLayer = GCNLayer()

        self.ini_embeds = ini_embeds.detach()
        self.fnl_embeds = fnl_embeds.detach()
        self.ini_embeds.requires_grad = False
        self.fnl_embeds.requires_grad = False
        self.model = model
        _freeze_base_model(self.model)

    def forward(self, ori_adj, ts_pk_adj, mask, ts_drp_adj):
        comp = _base_forward_components(
            self.edge_embeds1, self.gcnLayer, self.edgeDropper,
            ts_drp_adj, ts_pk_adj,
            self.ini_embeds, self.fnl_embeds, self.withdraw_rate)
        gnnLats = comp['gnnLats']
        hyperLats = comp['hyperLats']
        influence_sig = comp['influence_signal']

        delta_emb = -args.overall_withdraw_rate * comp['withdraw'] + comp['edge_out']
        H_old = delta_emb + influence_sig               # combine learned + influence

        # Local multi-head attention over edges of A_delta
        H_attn = self.local_attn(H_old, ts_drp_adj)
        # Residual + LayerNorm
        delta_emb = self.attn_norm(H_old + H_attn)

        for layer in self.mlp_layers:
            delta_emb = layer(delta_emb)

        tuned_emb = self.ini_embeds + delta_emb
        return tuned_emb, gnnLats, hyperLats

    def outforward(self, ori_adj, ts_pk_adj, mask, ts_drp_adj):
        self.model.training = False
        tuned_emb, _, _ = self.forward(ori_adj, ts_pk_adj, mask, ts_drp_adj)
        out_emb = self.model.forward(ts_pk_adj, tuned_emb, keepRate=1.0)
        return out_emb[0], out_emb[1]

    def cal_loss(self, batch_data, ori_adj, ts_pk_adj, mask, ts_drp_adj, drp_edges, pk_edges=None):
        ancs, poss, negs = list(map(lambda x: x.long(), batch_data))
        self.model.training = True
        tuned_emb, gcnEmbedsLst, hyperEmbedsLst = self.forward(ori_adj, ts_pk_adj, mask, ts_drp_adj)
        out_emb = self.model.forward(ts_pk_adj, tuned_emb)
        usr_embeds, itm_embeds = out_emb[:2]

        base_loss, loss_dict = self.model.cal_loss(
            batch_data, tuned_emb=tuned_emb, ori_adj=ori_adj, ts_pk_adj=ts_pk_adj,
            mask=mask, ts_drp_adj=ts_drp_adj, drp_edges=drp_edges, pk_edges=None)

        if args.unlearn_type == 'v1':
            unlearn_loss = cal_neg_aug_v1(usr_embeds[drp_edges[0]], itm_embeds[drp_edges[1]])
        elif args.unlearn_type == 'v2':
            unlearn_loss = cal_neg_aug_v2(usr_embeds[drp_edges[0]], itm_embeds[drp_edges[1]])

        tar_fnl_uEmbeds = self.fnl_embeds[:args.user].detach()
        tar_fnl_iEmbeds = self.fnl_embeds[args.user:].detach()

        if args.fineTune:
            align_loss = cal_positive_pred_align_v2(
                usr_embeds[ancs], tar_fnl_uEmbeds[ancs], itm_embeds[poss], tar_fnl_iEmbeds[poss],
                cal_l2_distance, temp=args.align_temp)
            loss_dict['unlearn_loss'] = unlearn_loss
            loss_dict['align_loss'] = align_loss
            loss_dict['unlearn_ssl'] = t.tensor(0.)
            return base_loss + args.unlearn_wei * unlearn_loss + args.align_wei * align_loss, loss_dict

        if args.align_type == 'v2':
            align_loss = cal_positive_pred_align_v2(
                usr_embeds[ancs], tar_fnl_uEmbeds[ancs], itm_embeds[poss], tar_fnl_iEmbeds[poss],
                cal_l2_distance, temp=args.align_temp)
        elif args.align_type == 'v3':
            align_loss = cal_positive_pred_align_v3(
                usr_embeds[ancs], tar_fnl_uEmbeds[ancs], itm_embeds[poss], tar_fnl_iEmbeds[poss],
                cal_l2_distance, temp=args.align_temp)

        sslLoss = _compute_ssl_loss(gcnEmbedsLst, hyperEmbedsLst, ancs, poss)

        loss = args.unlearn_wei * unlearn_loss + args.align_wei * align_loss + base_loss + args.unlearn_ssl * sslLoss
        loss_dict['unlearn_loss'] = unlearn_loss
        loss_dict['align_loss'] = align_loss
        loss_dict['unlearn_ssl'] = sslLoss
        return loss, loss_dict

    def full_predict(self, ori_adj, ts_pk_adj, mask, ts_drp_adj, usrs, trn_mask):
        self.model.training = False
        usr_embeds, itm_embeds = self.outforward(ori_adj, ts_pk_adj, mask, ts_drp_adj)
        pck_usr_embeds = usr_embeds[usrs]
        full_preds = pck_usr_embeds @ itm_embeds.T
        full_preds = full_preds * (1 - trn_mask) - trn_mask * 1e8
        return full_preds


# ---------------------------------------------------------------------------
# Hypernetwork-based Influence Encoder
# ---------------------------------------------------------------------------
class HypernetGraphUnlearning(nn.Module):
    """
    A hyper-network observes a *rich* summary of the dropped edges
    (concat of mean-pool and max-pool over the influence signal) and
    generates the full weight set for a 2-layer MLP that transforms
    the influence delta, conditioning the unlearning on *what* is
    being removed.
    """

    def __init__(self, handler, model, ini_embeds, fnl_embeds):
        super(HypernetGraphUnlearning, self).__init__()
        self.edge_embeds1 = nn.Parameter(t.zeros(args.user + args.item, args.latdim).cuda())

        D = args.latdim
        H = D  # hidden dim of generated MLP
        summary_dim = D * 2  # concat(mean_pool, max_pool)

        # Generators for 2-layer MLP:  D -> H -> D
        self.gen_W1 = nn.Sequential(
            nn.Linear(summary_dim, D), nn.LeakyReLU(negative_slope=args.leaky),
            nn.Linear(D, D * H))
        self.gen_b1 = nn.Sequential(
            nn.Linear(summary_dim, D), nn.LeakyReLU(negative_slope=args.leaky),
            nn.Linear(D, H))
        self.gen_W2 = nn.Sequential(
            nn.Linear(summary_dim, D), nn.LeakyReLU(negative_slope=args.leaky),
            nn.Linear(D, H * D))
        self.gen_b2 = nn.Sequential(
            nn.Linear(summary_dim, D), nn.LeakyReLU(negative_slope=args.leaky),
            nn.Linear(D, D))

        self.mlp_layers = nn.Sequential(
            *[FeedForwardLayer(args.latdim, args.latdim, act=args.act) for _ in range(args.layer_mlp)]
        )
        self.act = nn.LeakyReLU(negative_slope=args.leaky)

        if args.withdraw_rate_init == 1:
            self.withdraw_rate = nn.Parameter(t.ones(args.user + args.item, 1) * args.lr * 2)
        else:
            self.withdraw_rate = nn.Parameter(t.zeros(args.user + args.item, 1) * args.lr * 10)

        self.edgeDropper = SpAdjDropEdge()
        self.gcnLayer = GCNLayer()

        self.ini_embeds = ini_embeds.detach()
        self.fnl_embeds = fnl_embeds.detach()
        self.ini_embeds.requires_grad = False
        self.fnl_embeds.requires_grad = False
        self.model = model
        _freeze_base_model(self.model)

    def forward(self, ori_adj, ts_pk_adj, mask, ts_drp_adj):
        comp = _base_forward_components(
            self.edge_embeds1, self.gcnLayer, self.edgeDropper,
            ts_drp_adj, ts_pk_adj,
            self.ini_embeds, self.fnl_embeds, self.withdraw_rate)
        gnnLats = comp['gnnLats']
        hyperLats = comp['hyperLats']
        influence_sig = comp['influence_signal']

        # Rich summary:  concat(mean_pool, max_pool)  over influence signal
        m_pool = mean_pool_sparse(ts_drp_adj, influence_sig)   # (1, D)
        x_pool = max_pool_sparse(ts_drp_adj, influence_sig)    # (1, D)
        summary = t.cat([m_pool, x_pool], dim=-1)              # (1, 2D)

        D = args.latdim
        H = D
        W1 = self.gen_W1(summary).view(D, H)
        b1 = self.gen_b1(summary).view(1, H)
        W2 = self.gen_W2(summary).view(H, D)
        b2 = self.gen_b2(summary).view(1, D)

        # Apply generated 2-layer MLP:  delta = act(sig @ W1 + b1) @ W2 + b2
        delta_generated = self.act(influence_sig @ W1 + b1) @ W2 + b2

        delta_emb = (-args.overall_withdraw_rate * comp['withdraw']
                     + comp['edge_out'] + delta_generated)

        for layer in self.mlp_layers:
            delta_emb = layer(delta_emb)

        tuned_emb = self.ini_embeds + delta_emb
        return tuned_emb, gnnLats, hyperLats

    def outforward(self, ori_adj, ts_pk_adj, mask, ts_drp_adj):
        self.model.training = False
        tuned_emb, _, _ = self.forward(ori_adj, ts_pk_adj, mask, ts_drp_adj)
        out_emb = self.model.forward(ts_pk_adj, tuned_emb, keepRate=1.0)
        return out_emb[0], out_emb[1]

    def cal_loss(self, batch_data, ori_adj, ts_pk_adj, mask, ts_drp_adj, drp_edges, pk_edges=None):
        ancs, poss, negs = list(map(lambda x: x.long(), batch_data))
        self.model.training = True
        tuned_emb, gcnEmbedsLst, hyperEmbedsLst = self.forward(ori_adj, ts_pk_adj, mask, ts_drp_adj)
        out_emb = self.model.forward(ts_pk_adj, tuned_emb)
        usr_embeds, itm_embeds = out_emb[:2]

        base_loss, loss_dict = self.model.cal_loss(
            batch_data, tuned_emb=tuned_emb, ori_adj=ori_adj, ts_pk_adj=ts_pk_adj,
            mask=mask, ts_drp_adj=ts_drp_adj, drp_edges=drp_edges, pk_edges=None)

        if args.unlearn_type == 'v1':
            unlearn_loss = cal_neg_aug_v1(usr_embeds[drp_edges[0]], itm_embeds[drp_edges[1]])
        elif args.unlearn_type == 'v2':
            unlearn_loss = cal_neg_aug_v2(usr_embeds[drp_edges[0]], itm_embeds[drp_edges[1]])

        tar_fnl_uEmbeds = self.fnl_embeds[:args.user].detach()
        tar_fnl_iEmbeds = self.fnl_embeds[args.user:].detach()

        if args.fineTune:
            align_loss = cal_positive_pred_align_v2(
                usr_embeds[ancs], tar_fnl_uEmbeds[ancs], itm_embeds[poss], tar_fnl_iEmbeds[poss],
                cal_l2_distance, temp=args.align_temp)
            loss_dict['unlearn_loss'] = unlearn_loss
            loss_dict['align_loss'] = align_loss
            loss_dict['unlearn_ssl'] = t.tensor(0.)
            return base_loss + args.unlearn_wei * unlearn_loss + args.align_wei * align_loss, loss_dict

        if args.align_type == 'v2':
            align_loss = cal_positive_pred_align_v2(
                usr_embeds[ancs], tar_fnl_uEmbeds[ancs], itm_embeds[poss], tar_fnl_iEmbeds[poss],
                cal_l2_distance, temp=args.align_temp)
        elif args.align_type == 'v3':
            align_loss = cal_positive_pred_align_v3(
                usr_embeds[ancs], tar_fnl_uEmbeds[ancs], itm_embeds[poss], tar_fnl_iEmbeds[poss],
                cal_l2_distance, temp=args.align_temp)

        sslLoss = _compute_ssl_loss(gcnEmbedsLst, hyperEmbedsLst, ancs, poss)

        loss = args.unlearn_wei * unlearn_loss + args.align_wei * align_loss + base_loss + args.unlearn_ssl * sslLoss
        loss_dict['unlearn_loss'] = unlearn_loss
        loss_dict['align_loss'] = align_loss
        loss_dict['unlearn_ssl'] = sslLoss
        return loss, loss_dict

    def full_predict(self, ori_adj, ts_pk_adj, mask, ts_drp_adj, usrs, trn_mask):
        self.model.training = False
        usr_embeds, itm_embeds = self.outforward(ori_adj, ts_pk_adj, mask, ts_drp_adj)
        pck_usr_embeds = usr_embeds[usrs]
        full_preds = pck_usr_embeds @ itm_embeds.T
        full_preds = full_preds * (1 - trn_mask) - trn_mask * 1e8
        return full_preds


# ---------------------------------------------------------------------------
# Causal Influence Encoder
# ---------------------------------------------------------------------------
class CausalGraphUnlearning(nn.Module):
    """
    Models unlearning as a causal intervention.  Factual (full graph) and
    counterfactual (residual graph) embeddings define the causal effect.
    A learned gate blends the causal effect with the base edge-embedding
    path.  A causal-consistency loss additionally supervises the predicted
    delta to match the approximate counterfactual target.
    """

    def __init__(self, handler, model, ini_embeds, fnl_embeds):
        super(CausalGraphUnlearning, self).__init__()
        self.edge_embeds1 = nn.Parameter(t.zeros(args.user + args.item, args.latdim).cuda())

        # Gate:  sigmoid(MLP([causal_effect || influence_signal]))  -> (N,1)
        self.causal_gate = nn.Sequential(
            nn.Linear(args.latdim * 2, args.latdim),
            nn.LeakyReLU(negative_slope=args.leaky),
            nn.Linear(args.latdim, 1),
            nn.Sigmoid(),
        )

        self.mlp_layers = nn.Sequential(
            *[FeedForwardLayer(args.latdim, args.latdim, act=args.act) for _ in range(args.layer_mlp)]
        )
        self.act = nn.LeakyReLU(negative_slope=args.leaky)

        if args.withdraw_rate_init == 1:
            self.withdraw_rate = nn.Parameter(t.ones(args.user + args.item, 1) * args.lr * 2)
        else:
            self.withdraw_rate = nn.Parameter(t.zeros(args.user + args.item, 1) * args.lr * 10)

        self.edgeDropper = SpAdjDropEdge()
        self.gcnLayer = GCNLayer()

        self.ini_embeds = ini_embeds.detach()
        self.fnl_embeds = fnl_embeds.detach()
        self.ini_embeds.requires_grad = False
        self.fnl_embeds.requires_grad = False
        self.model = model
        _freeze_base_model(self.model)

    def forward(self, ori_adj, ts_pk_adj, mask, ts_drp_adj):
        comp = _base_forward_components(
            self.edge_embeds1, self.gcnLayer, self.edgeDropper,
            ts_drp_adj, ts_pk_adj,
            self.ini_embeds, self.fnl_embeds, self.withdraw_rate)
        gnnLats = comp['gnnLats']
        hyperLats = comp['hyperLats']
        influence_sig = comp['influence_signal']

        # ---- causal branches (model-agnostic, no backbone retraining) ----
        factual = graph_propagate(ori_adj, self.ini_embeds, args.gnn_layer)
        counterfactual = graph_propagate(ts_pk_adj, self.ini_embeds, args.gnn_layer)
        causal_effect = factual - counterfactual  # contribution of deleted edges

        # Stable gating
        gate_input = t.cat([causal_effect, influence_sig], dim=-1)
        gate = self.causal_gate(gate_input)   # (N, 1) in [0, 1]
        blended = gate * causal_effect + (1 - gate) * comp['edge_out']

        delta_emb = -args.overall_withdraw_rate * comp['withdraw'] + blended

        for layer in self.mlp_layers:
            delta_emb = layer(delta_emb)

        tuned_emb = self.ini_embeds + delta_emb

        # Store for causal consistency loss
        # delta_target = E_cf_approx - E  where E_cf_approx = counterfactual
        self._delta_target = (counterfactual - factual).detach()  # = -causal_effect
        self._delta_pred = tuned_emb - self.ini_embeds            # predicted delta

        return tuned_emb, gnnLats, hyperLats

    def outforward(self, ori_adj, ts_pk_adj, mask, ts_drp_adj):
        self.model.training = False
        tuned_emb, _, _ = self.forward(ori_adj, ts_pk_adj, mask, ts_drp_adj)
        out_emb = self.model.forward(ts_pk_adj, tuned_emb, keepRate=1.0)
        return out_emb[0], out_emb[1]

    def cal_loss(self, batch_data, ori_adj, ts_pk_adj, mask, ts_drp_adj, drp_edges, pk_edges=None):
        ancs, poss, negs = list(map(lambda x: x.long(), batch_data))
        self.model.training = True
        tuned_emb, gcnEmbedsLst, hyperEmbedsLst = self.forward(ori_adj, ts_pk_adj, mask, ts_drp_adj)
        out_emb = self.model.forward(ts_pk_adj, tuned_emb)
        usr_embeds, itm_embeds = out_emb[:2]

        base_loss, loss_dict = self.model.cal_loss(
            batch_data, tuned_emb=tuned_emb, ori_adj=ori_adj, ts_pk_adj=ts_pk_adj,
            mask=mask, ts_drp_adj=ts_drp_adj, drp_edges=drp_edges, pk_edges=None)

        if args.unlearn_type == 'v1':
            unlearn_loss = cal_neg_aug_v1(usr_embeds[drp_edges[0]], itm_embeds[drp_edges[1]])
        elif args.unlearn_type == 'v2':
            unlearn_loss = cal_neg_aug_v2(usr_embeds[drp_edges[0]], itm_embeds[drp_edges[1]])

        tar_fnl_uEmbeds = self.fnl_embeds[:args.user].detach()
        tar_fnl_iEmbeds = self.fnl_embeds[args.user:].detach()

        # Causal consistency:  ||delta_pred - delta_target||^2
        causal_loss = F.mse_loss(self._delta_pred, self._delta_target)

        if args.fineTune:
            align_loss = cal_positive_pred_align_v2(
                usr_embeds[ancs], tar_fnl_uEmbeds[ancs], itm_embeds[poss], tar_fnl_iEmbeds[poss],
                cal_l2_distance, temp=args.align_temp)
            loss_dict['unlearn_loss'] = unlearn_loss
            loss_dict['align_loss'] = align_loss
            loss_dict['unlearn_ssl'] = causal_loss
            return (base_loss + args.unlearn_wei * unlearn_loss
                    + args.align_wei * align_loss
                    + args.lambda_causal * causal_loss), loss_dict

        if args.align_type == 'v2':
            align_loss = cal_positive_pred_align_v2(
                usr_embeds[ancs], tar_fnl_uEmbeds[ancs], itm_embeds[poss], tar_fnl_iEmbeds[poss],
                cal_l2_distance, temp=args.align_temp)
        elif args.align_type == 'v3':
            align_loss = cal_positive_pred_align_v3(
                usr_embeds[ancs], tar_fnl_uEmbeds[ancs], itm_embeds[poss], tar_fnl_iEmbeds[poss],
                cal_l2_distance, temp=args.align_temp)

        sslLoss = _compute_ssl_loss(gcnEmbedsLst, hyperEmbedsLst, ancs, poss)

        loss = (args.unlearn_wei * unlearn_loss + args.align_wei * align_loss
                + base_loss + args.unlearn_ssl * sslLoss
                + args.lambda_causal * causal_loss)
        loss_dict['unlearn_loss'] = unlearn_loss
        loss_dict['align_loss'] = align_loss
        loss_dict['unlearn_ssl'] = sslLoss
        return loss, loss_dict

    def full_predict(self, ori_adj, ts_pk_adj, mask, ts_drp_adj, usrs, trn_mask):
        self.model.training = False
        usr_embeds, itm_embeds = self.outforward(ori_adj, ts_pk_adj, mask, ts_drp_adj)
        pck_usr_embeds = usr_embeds[usrs]
        full_preds = pck_usr_embeds @ itm_embeds.T
        full_preds = full_preds * (1 - trn_mask) - trn_mask * 1e8
        return full_preds


# ---------------------------------------------------------------------------
# Factory – select encoder by name
# ---------------------------------------------------------------------------
ENCODER_REGISTRY = {
    'default': GraphUnlearning,
    'autoencoder': AutoencoderGraphUnlearning,
    'attention': AttentionGraphUnlearning,
    'hypernet': HypernetGraphUnlearning,
    'causal': CausalGraphUnlearning,
}


def build_unlearning_encoder(handler, model, ini_embeds, fnl_embeds):
    """Instantiate the unlearning encoder selected by ``args.encoder_type``."""
    key = getattr(args, 'encoder_type', 'default')
    cls = ENCODER_REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"Unknown encoder_type '{key}'. Choose from {list(ENCODER_REGISTRY.keys())}")
    return cls(handler, model, ini_embeds, fnl_embeds)


class LightGCN(nn.Module):
    def __init__(self, handler):
        super(LightGCN, self).__init__()

        # self.adj = handler.torch_adj
        self.handler = handler
        self.adj = handler.ts_ori_adj
        self.ini_embeds = nn.Parameter(init(t.empty(args.user + args.item, args.latdim)))

    def forward(self, adj, ini_embeds=None, all_layer=False, keepRate=None):
        if ini_embeds is None:
            ini_embeds = self.ini_embeds

        embedsList = [ini_embeds]
        for _ in range(args.gnn_layer):
            embedsList.append(adj.matmul(embedsList[-1]))
        embeds = sum(embedsList)

        if all_layer:
            return (embedsList, embeds)

        return embeds[:args.user], embeds[args.user:]
    
    def cal_loss(self, batch_data,  tuned_emb=None , ori_adj=None, ts_pk_adj=None , mask=None, ts_drp_adj=None, drp_edges=None,  pk_edges=None):
        ancs, poss, negs = list(map(lambda x: x.long(), batch_data))
        usr_embeds, itm_embeds = self.forward( ts_pk_adj , tuned_emb)
        bpr_loss = cal_bpr(usr_embeds[ancs], itm_embeds[poss], itm_embeds[negs]) * args.bpr_wei

        reg_loss = cal_reg(self) * args.reg        
        loss = bpr_loss + reg_loss 
    
        loss_dict = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss}
        return loss, loss_dict

    def full_predict(self, usrs, trn_mask, adj):
        usr_embeds, itm_embeds = self.forward(adj)
        pck_usr_embeds = usr_embeds[usrs]
        full_preds = pck_usr_embeds @ itm_embeds.T
        full_preds = full_preds * (1 - trn_mask) - trn_mask * 1e8
        return full_preds





init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class SimGCL(nn.Module):
    def __init__(self, handler):
        super(SimGCL, self).__init__()
        
        self.adj = handler.ts_ori_adj
        self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
        self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
        self.gcnLayers = nn.Sequential(*[SimGclGCNLayer() for i in range(args.gnn_layer)])
        self.perturbGcnLayers1 = nn.Sequential(*[SimGclGCNLayer(perturb=True) for i in range(args.gnn_layer)])
        self.perturbGcnLayers2 = nn.Sequential(*[SimGclGCNLayer(perturb=True) for i in range(args.gnn_layer)])


    def getEgoEmbeds(self, adj):
        uEmbeds, iEmbeds = self.forward(adj)
        return t.concat([uEmbeds, iEmbeds], axis=0)

    def forward(self, adj, iniEmbeds=None, all_layer=False, keepRate=None):
        if iniEmbeds is None:          
            iniEmbeds = t.concat([self.uEmbeds, self.iEmbeds], axis=0)        
                                            
        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        mainEmbeds = sum(embedsLst[1:]) / len(embedsLst[1:])
        if all_layer:
            return (embedsLst,mainEmbeds)

        if self.training:
            perturbEmbedsLst1 = [iniEmbeds]
            for gcn in self.perturbGcnLayers1:
                embeds = gcn(adj, perturbEmbedsLst1[-1])
                perturbEmbedsLst1.append(embeds)
            perturbEmbeds1 = sum(perturbEmbedsLst1[1:]) / len(embedsLst[1:])

            perturbEmbedsLst2 = [iniEmbeds]
            for gcn in self.perturbGcnLayers2:
                embeds = gcn(adj, perturbEmbedsLst2[-1])
                perturbEmbedsLst2.append(embeds)
            perturbEmbeds2 = sum(perturbEmbedsLst2[1:]) / len(embedsLst[1:])

            return mainEmbeds[:args.user], mainEmbeds[args.user:], perturbEmbeds1[:args.user], perturbEmbeds1[args.user:], perturbEmbeds2[:args.user], perturbEmbeds2[args.user:]
        return mainEmbeds[:args.user], mainEmbeds[args.user:]

    def cal_loss(self, batch_data,  tuned_emb=None , ori_adj=None, ts_pk_adj=None , mask=None, ts_drp_adj=None, drp_edges=None,  pk_edges=None):
        ancs, poss, negs = list(map(lambda x: x.long(), batch_data))
        # ancs, poss, negs = tem
        ancs = ancs.long().cuda()
        poss = poss.long().cuda()
        negs = negs.long().cuda()
        self.train()
        # print("###################cal_loss self.train########################")
        # print(self.training)
        usrEmbeds, itmEmbeds, pUsrEmbeds1, pItmEmbeds1, pUsrEmbeds2, pItmEmbeds2 = self.forward(ts_pk_adj, tuned_emb )

        ancEmbeds = usrEmbeds[ancs]
        posEmbeds = itmEmbeds[poss]
        negEmbeds = itmEmbeds[negs]

        scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        bprLoss = - (scoreDiff).sigmoid().log().mean()
        if args.reg_version == 'v1':
            regLoss = SimGCL_calcRegLoss(ancEmbeds, posEmbeds) 
        elif args.reg_version == 'v2':
            regLoss = SimGCL_calcRegLoss_v2(ancEmbeds, posEmbeds) 
        else:
            regLoss = SimGCL_calcRegLoss_v3(ancEmbeds, posEmbeds)                 

        contrastLoss = (contrast(pUsrEmbeds1, pUsrEmbeds2, ancs, args.temp) + contrast(pItmEmbeds1, pItmEmbeds2, poss, args.temp)) 
        # contrastLoss = 0


        loss = args.bpr_wei * bprLoss +  args.reg * regLoss + args.ssl_reg * contrastLoss   

        loss_dict = {'bpr_loss': bprLoss, 'reg_loss': regLoss, "contrast_loss": contrastLoss}
        return loss, loss_dict                

    def full_predict(self, usrs, trn_mask, adj):
        self.training = False
        usr_embeds, itm_embeds = self.forward(adj)
        pck_usr_embeds = usr_embeds[usrs]
        full_preds = pck_usr_embeds @ itm_embeds.T
        full_preds = full_preds * (1 - trn_mask) - trn_mask * 1e8
        return full_preds


class SimGclGCNLayer(nn.Module):
    def __init__(self, perturb=False):
        super(SimGclGCNLayer, self).__init__()
        self.perturb = perturb

    def forward(self, adj, embeds):
        # ret = t.spmm(adj, embeds)
        ret = adj.matmul(embeds)
        if not self.perturb:
            return ret
        # noise = (F.normalize(t.rand(ret.shape).cuda(), p=2) * t.sign(ret)) * args.eps
        random_noise = t.rand_like(ret).cuda()
        noise = t.sign(ret) * F.normalize(random_noise, dim=-1) * args.eps
        return ret + noise


def get_shape(adj):
    if isinstance(adj, ts.SparseTensor):
        return adj.sizes()
    else:
        return adj.shape


class FeedForwardLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=False, act=None):
        super(FeedForwardLayer, self).__init__()
        # self.linear = nn.Linear(in_feat, out_feat, bias=bias)
        # self.W = nn.Parameter(t.zeros(args.latdim, args.latdim).cuda())
        self.W = nn.Parameter(t.eye(args.latdim, args.latdim).cuda(), requires_grad=False)
        self.bias = nn.Parameter(t.zeros( 1, args.latdim).cuda(), requires_grad=False)

        
        if act == 'identity' or act is None:
            self.act = None
        elif act == 'leaky':
            self.act = nn.LeakyReLU(negative_slope=args.leaky)
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'relu6':
            self.act = nn.ReLU6()
        else:
            raise Exception('Error')
    
    def forward(self, embeds):
        if self.act is None:
            # return self.linear(embeds)
            return  embeds @ self.W 
        # return (self.act(  embeds @ self.W + self.bias )) + embeds  #  default   v1
        return self.act(  embeds @ self.W + self.bias )  #  v2
    

class SGL(nn.Module):
    def __init__(self):
        super(SGL, self).__init__()

        self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
        self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

        self.edgeDropper = SpAdjDropEdge()

    def getEgoEmbeds(self, adj):
        uEmbeds, iEmbeds = self.forward(adj)
        return t.concat([uEmbeds, iEmbeds], axis=0)

    def forward(self, adj, iniEmbeds=None , keepRate=args.sglkeepRate):
        if iniEmbeds is None:
            iniEmbeds = t.concat([self.uEmbeds, self.iEmbeds], axis=0)

        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        mainEmbeds = sum(embedsLst) / len(embedsLst)

        if keepRate == 1.0 or self.training == False:
            return mainEmbeds[:args.user], mainEmbeds[args.user:]

        adjView1 = self.edgeDropper(adj, keepRate)
        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adjView1, embedsLst[-1])
            embedsLst.append(embeds)
        embedsView1 = sum(embedsLst)

        adjView2 = self.edgeDropper(adj, keepRate)
        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adjView2, embedsLst[-1])
            embedsLst.append(embeds)
        embedsView2 = sum(embedsLst)
        return mainEmbeds[:args.user], mainEmbeds[args.user:], embedsView1[:args.user], embedsView1[args.user:], embedsView2[:args.user], embedsView2[args.user:]		
     
    def cal_loss(self, batch_data,  tuned_emb=None , ori_adj=None, ts_pk_adj=None , mask=None, ts_drp_adj=None, drp_edges=None,  pk_edges=None):     
        # ancs, poss, negs = batch_data
        ancs, poss, negs = list(map(lambda x: x.long(), batch_data))
        ancs = ancs.long().cuda()
        poss = poss.long().cuda()
        negs = negs.long().cuda()
        usrEmbeds, itmEmbeds, usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2 = self.forward(ts_pk_adj, iniEmbeds=tuned_emb ,keepRate = args.sglkeepRate)
        ancEmbeds = usrEmbeds[ancs]
        posEmbeds = itmEmbeds[poss]
        negEmbeds = itmEmbeds[negs]

        clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, args.sgltemp) + contrastLoss(itmEmbeds1, itmEmbeds2, poss, args.sgltemp)) * args.sgl_ssl_reg

        scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        # bprLoss = - (scoreDiff).sigmoid().log().sum()
        bprLoss = - ((scoreDiff).sigmoid() + 1e-8 ).log().mean()
        regLoss = calcRegLoss([self.uEmbeds[ancs], self.iEmbeds[poss], self.iEmbeds[negs]]) * args.reg
        # regLoss = calcRegLoss(self.model) * args.reg
        loss = bprLoss + regLoss + clLoss

        loss_dict = {'bpr_loss': bprLoss, 'reg_loss': regLoss}

        return loss, loss_dict

    def full_predict(self, usrs, trn_mask, adj):
        self.training = False
        usr_embeds, itm_embeds = self.forward(adj, keepRate=1.0)
        pck_usr_embeds = usr_embeds[usrs]
        full_preds = pck_usr_embeds @ itm_embeds.T
        full_preds = full_preds * (1 - trn_mask) - trn_mask * 1e8
        return full_preds
