import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.linear import Identity
from tglite._block import TBlock
from tglite._context import TContext
from tglite.nn import TimeEncode
from tglite.op import precomputed_zeros, precomputed_times, edge_reduce, edge_view, edge_softmax
import nvtx
from ..operators.fused_tmpAttn import GATConvFuse, GATConvFuse_inference

class TemporalAttnLayer0_2_perfCeil(torch.nn.Module): # our tmpAttnLayer
    def __init__(self, ctx: TContext, num_heads: int,
                 dim_node: int, dim_edge: int, dim_time: int, dim_out: int,
                 dropout=0.1):
        """
        Initializes the Temporal Attention Layer for processing dynamic graphs with temporal features.
        This layer uses multi-head attention mechanism to incorporate node, edge, and time features.

        :param ctx: context object
        :param num_heads: number of heads
        :param dim_node: dimension of node features
        :param dim_edge: dimension of edge features
        :param dim_time: dimension of time features
        :param dim_out: dimension of output features
        :param dropout: dropout rate
        """
        super().__init__()
        assert (dim_out % num_heads == 0)
        self.ctx = ctx
        self.num_heads = num_heads
        self.dim_edge = dim_edge
        self.dim_out = dim_out
        self.time_encode = TimeEncode(dim_time)
        # 拆分后前反向的性能开销 TODO
        self.w_q = torch.nn.Linear(dim_node + dim_time, dim_out)
        self.w_kv = torch.nn.Linear(dim_node + dim_edge + dim_time, dim_out * 2)
        '''
        self.w_kv_node = self.w_kv[dim_node, :] # TypeError: 'Linear' object is not subscriptable
        self.w_kv_edge = self.w_kv[dim_node : dim_edge, :]
        self.w_kv_time = self.w_kv[dim_node + dim_edge :, :]
        print(f"self.w_kv_node {self.w_kv_node}")
        print(f"self.w_kv_edge {self.w_kv_edge}")
        print(f"self.w_kv_time {self.w_kv_time}")'''
        self.w_q_node = torch.nn.Linear(dim_node, dim_out)
        self.w_q_time = torch.nn.Linear(dim_time, dim_out)
        self.w_kv_node = torch.nn.Linear(dim_node, dim_out * 2)
        self.w_qkv_node = torch.nn.Linear(dim_node, dim_out * 3)
        self.w_kv_edge = torch.nn.Linear(dim_edge, dim_out * 2)
        self.w_kv_time = torch.nn.Linear(dim_time, dim_out * 2)
        self.w_out = torch.nn.Linear(dim_node + dim_out, dim_out)
        self.attn_act = torch.nn.LeakyReLU(0.2)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(dim_out)
    
    def compute_z_ours(self, idx, nodeData, node_inverse, node_dst_inverse, efeat_unique, efeat_inverse, time_unique, time_inverse, time_dst_unique, time_dst_inverse):
        # print(f"Q= {Q.shape}")
        # print(f"idx= {idx.shape}")
        # print(f"node_unique= {node_unique.shape}")
        # print(f"node_inverse= {node_inverse.shape}")
        # print(f"efeat_unique= {efeat_unique.shape}")
        # print(f"efeat_inverse= {efeat_inverse.shape}")
        # print(f"time_unique= {time_unique.shape}")
        # print(f"time_inverse= {time_inverse.shape}")
        # QKV_node = self.w_qkv_node(nodeData)

        Q_node = self.w_q_node(nodeData)
        # Q_node = QKV_node[:, :self.dim_out]
        Q_time = self.w_q_time(time_dst_unique)

        Q_node = Q_node[node_dst_inverse]
        Q_time = Q_time[time_dst_inverse]
        Q_our = Q_node + Q_time
        
        Q = Q_our[idx]


        Z_node = self.w_kv_node(nodeData)
        # Z_node = QKV_node[:, self.dim_out:]
        Z_edge = self.w_kv_edge(efeat_unique)
        Z_time = self.w_kv_time(time_unique)
        
        Z_node = Z_node[node_inverse]
        Z_edge = Z_edge[efeat_inverse]
        Z_time = Z_time[time_inverse]
        Z_our = Z_node + Z_edge + Z_time

        K = Z_our[:, :self.dim_out]
        V = Z_our[:, self.dim_out:]
        

        assert Q.shape[0] == K.shape[0]

        Q = torch.reshape(Q, (Q.shape[0], self.num_heads, -1))
        K = torch.reshape(K, (K.shape[0], self.num_heads, -1))
        V = torch.reshape(V, (V.shape[0], self.num_heads, -1))

        attn = torch.sum(Q * K, dim=2)
        attn = self.attn_act(attn)
        return attn, V

    def compute_z_ours_nvtx(self, blk: TBlock, zero_time_feat, nbrs_time_feat):
        with nvtx.annotate("redundancy-mul", color="red"): # TODO 这段的执行时间确实长
            # concat < mul
            with nvtx.annotate("redundancy-concat: Q", color="red"):
                Q = torch.cat([blk.dstdata['h'], zero_time_feat], dim=1)
            with nvtx.annotate("redundancy-concat: KV", color="red"): # TODO 这段的执行时间很意外 torch.cat 原理
                Z = torch.cat([blk.srcdata['h'], blk.efeat(), nbrs_time_feat], dim=1)

            del zero_time_feat
            # del nbrs_time_feat

            with nvtx.annotate("redundancy-mul: Q", color="red"):
                Q = self.w_q(Q)
            with nvtx.annotate("redundancy-mul: KV", color="red"): # TODO 这段的执行时间确实长
                Z = self.w_kv(Z) # 消融一下
            

            with torch.no_grad():
                node_unique, node_inverse = torch.unique(blk.srcdata['h'], dim=0, return_inverse=True)
                efeat_unique, efeat_inverse = torch.unique(blk.efeat(), dim=0, return_inverse=True) # 172的长度可能乘起来不够高效
                time_unique, time_inverse = torch.unique(nbrs_time_feat, dim=0, return_inverse=True)
            # print(f"node_unique {node_unique.shape} efeat_unique {efeat_unique.shape} time_unique {time_unique.shape}")
            
            with nvtx.annotate("redundancy-our: KV cal", color="red"): # TODO 这段的执行时间很意外 torch.cat 原理
                # concat > mul

                # Z_node = self.w_kv_node(blk.srcdata['h'])
                # Z_edge = self.w_kv_edge(blk.efeat())
                # Z_time = self.w_kv_time(nbrs_time_feat)
                # [DOING] 怎么把unique提前拿到
                with nvtx.annotate("redundancy-our: KV cal node", color="red"): # TODO 减少计算真的能减少执行时间
                    Z_node = self.w_kv_node(node_unique)
                with nvtx.annotate("redundancy-our: KV cal edge", color="red"):
                    Z_edge = self.w_kv_edge(efeat_unique)
                with nvtx.annotate("redundancy-our: KV cal time", color="red"):
                    Z_time = self.w_kv_time(time_unique)
            with nvtx.annotate("redundancy-our: KV reverse", color="red"): 
                with nvtx.annotate("redundancy-our: KV reverse node", color="red"):
                    Z_node = Z_node[node_inverse]
                with nvtx.annotate("redundancy-our: KV reverse edge", color="red"):
                    Z_edge = Z_edge[efeat_inverse]
                with nvtx.annotate("redundancy-our: KV reverse time", color="red"):
                    Z_time = Z_time[time_inverse]
            with nvtx.annotate("redundancy-our: KV add", color="red"): # TODO 这段的执行时间很意外 torch.cat 原理(index_elementwise + catArrayBatchedCopy)
                # TODO 怎么把unique提前拿到; element-wise
                Z_our = Z_node + Z_edge + Z_time

            # print(f"Z {Z.shape}")
            # print(f"Z {Z.shape} Z_our {Z_our.shape}")
            # print(f"Z = {Z}")
            # print(f"Z_our = {Z_our}")
            
            # assert Z.shape == Z_our.shape
                
            del nbrs_time_feat
        

        with nvtx.annotate("else", color="red"): # else这段没有显存变化 torch的机制？
            with nvtx.annotate("else-K", color="red"):
                K = Z_our[:, :self.dim_out]
            with nvtx.annotate("else-V", color="red"):
                V = Z_our[:, self.dim_out:]
                del Z
                
            with nvtx.annotate("else-Q", color="red"):
                # Q = edge_view(blk, Q) # 对Q进行scatter
                idx = torch.from_numpy(blk._dstindex)
                idx = idx.to(device=Q.device, dtype=torch.long)
                Q = Q[idx]
            with nvtx.annotate("else-reshape", color="red"):
                Q = torch.reshape(Q, (Q.shape[0], self.num_heads, -1))
                K = torch.reshape(K, (K.shape[0], self.num_heads, -1))
                V = torch.reshape(V, (V.shape[0], self.num_heads, -1))

            with nvtx.annotate("else-attn", color="red"):
                attn = torch.sum(Q * K, dim=2)
                del Q
                del K
        return attn, V
    
    # @torch.compile TODO
    # input: tail, nodeData, _reverse_nids
    def forward_redundancy_mul(self, blk: TBlock, nodeData, _reverse_nids, efeat_unique, _reverse_eids, _unique_time_delta, _reverse_time_delta): # TODO build pipeline
        # [DOING] 先拆分计算；纵切只要传入对应的indices范围即可
        # 处理bottleneck 的KV 部分
        with nvtx.annotate("forward_redundancy_mul", color="blue"):
            with nvtx.annotate("precompute", color="blue"):
                # no scatter
                # zero_time_feat = precomputed_zeros(self.ctx, blk.layer, self.time_encode, blk.num_dst()) # TODO 行全0 优化
                time_dst_unique = precomputed_zeros(self.ctx, blk.layer, self.time_encode, 1) 
                time_dst_inverse = torch.zeros(blk.num_dst(), dtype=torch.int64, device="cuda")
                # nbrs_time_feat_st = precomputed_times(self.ctx, blk.layer, self.time_encode, blk.time_deltas()) # TODO
                nbrs_time_feat = precomputed_times(self.ctx, blk.layer, self.time_encode, _unique_time_delta.to("cuda")) # precompute是elementwise 理论上讲行不变

            # Q = edge_view(blk, Q) # 对Q进行scatter
            idx = torch.from_numpy(blk._dstindex)
            idx = idx.to(device="cuda", dtype=torch.long)

            '''
            # 给具体可能的值 TODO
            # with torch.no_grad():
                # node_unique, node_inverse = torch.unique(blk.srcdata['h'], dim=0, return_inverse=True)
                # efeat_unique, efeat_inverse = torch.unique(blk.efeat(), dim=0, return_inverse=True) # 172的长度可能乘起来不够高效
                # time_unique, time_inverse = torch.unique(nbrs_time_feat_st, dim=0, return_inverse=True)
                # print(f"nbrs_time_feat_st.shape {nbrs_time_feat_st.shape}")
                # print(f"nbrs_time_feat_st = {nbrs_time_feat_st}")
                # print(f"nbrs_time_feat[_reverse_time_delta].shape {nbrs_time_feat[_reverse_time_delta].shape}")
                # print(f"nbrs_time_feat[_reverse_time_delta] = {nbrs_time_feat[_reverse_time_delta]}")
                # time_d_unique, time_d_inverse = torch.unique(blk.time_deltas(), dim=0, return_inverse=True)
                # print(f"_unique_time_delta.shape {_unique_time_delta[_reverse_time_delta].shape}")
                # print(f"_unique_time_delta[_reverse_time_delta] = {_unique_time_delta[_reverse_time_delta]}")
                # print(f"blk.time_deltas().shape {blk.time_deltas().shape}")
                # print(f"blk.time_deltas() = {blk.time_deltas()}")
                # print(f"_reverse_time_delta.shape {_reverse_time_delta.shape}")
                # print(f"_reverse_time_delta = {_reverse_time_delta}")
                # print(f"time_inverse.shape {time_inverse.shape}")
                # print(f"time_inverse = {time_inverse}")
                # print(f"nbrs_time_feat.shape {nbrs_time_feat.shape}")
                # print(f"nbrs_time_feat = {nbrs_time_feat}")
                # print(f"time_unique.shape {time_unique.shape}")
                # print(f"time_unique = {time_unique}")
                # assert torch.allclose(_unique_time_delta[_reverse_time_delta].to("cuda"), blk.time_deltas())
                # assert torch.allclose(nbrs_time_feat[_reverse_time_delta], nbrs_time_feat_st) # \O/ acc test passed!
                # print(f"time_unique {time_unique.shape}")
                # print(f"time_inverse = {time_inverse}")
                # print(f"time_delta_unique {time_d_unique.shape}")
                # print(f"time_delta_inverse = {time_d_inverse}")

                # node_dst_unique, node_dst_inverse = torch.unique(blk.dstdata['h'], dim=0, return_inverse=True)
                ## time_dst_unique_st, time_dst_inverse_st = torch.unique(zero_time_feat, dim=0, return_inverse=True) # 优先处理zero time feat
                ## assert torch.allclose(time_dst_unique, time_dst_unique_st)
                ## assert torch.allclose(time_dst_inverse, time_dst_inverse_st) # \O/ acc test passed!

                ## print("===") # TODO dst的冗余也有很多
                ## print(f"blk.dstdata['h'] {blk.dstdata['h'].shape}")
                ## print(f"node_dst_unique {node_dst_unique.shape}")
                ## print()
            '''

        # TODO 但是现在srcnode算的变多了，其实应该分开去重 -> 但矩阵乘的时间可以被pipeline掩盖/过于短的kernel对GPU而言也不友好，所以无所谓 -> 同一份nodeData 把两种W(三组W)拼在一起
        return self.ctx.compiled_forward_redundancy_mul(idx, nodeData, _reverse_nids[blk.num_dst():], _reverse_nids[:blk.num_dst()], efeat_unique, _reverse_eids, nbrs_time_feat, _reverse_time_delta, time_dst_unique, time_dst_inverse)
        ## no scatter
        # return self.ctx.compiled_forward_redundancy_mul(idx, node_unique, node_inverse, efeat_unique, efeat_inverse, time_unique, time_inverse, node_dst_unique, node_dst_inverse, time_dst_unique, time_dst_inverse)
        ## no-bug @torch.compile
        # return self.compute_z_ours(blk, zero_time_feat, nbrs_time_feat, idx) # TODO
        ## with nvtx
        # return self.compute_z_ours_nvtx(blk, zero_time_feat, nbrs_time_feat)
        
    # input: nodeData, _reverse_nids
    def forward(self, blk: TBlock, nodeData, _reverse_nids, efeat_unique, _reverse_eids, _unique_time_delta, _reverse_time_delta) -> Tensor:
        attn, V = self.forward_redundancy_mul(blk, nodeData, _reverse_nids, efeat_unique, _reverse_eids, _unique_time_delta, _reverse_time_delta)

        with nvtx.annotate("edge-softmax", color="blue"):
            with nvtx.annotate("edge-softmax", color="blue"):
                attn = edge_softmax(blk, attn)  # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            with nvtx.annotate("dropout", color="blue"):
                attn = self.dropout(attn)
            with nvtx.annotate("reshape", color="blue"):
                out = torch.reshape(V * attn[:, :, None], (V.shape[0], -1))  # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
                del attn

            with nvtx.annotate("edge-reduce", color="blue"):
                out = edge_reduce(blk, out, op='sum') # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):]
            # tail.dstdata['h'] = nfeat[:tail.num_dst()] + mem[:tail.num_dst()]
            blk.dstdata['h'] = nodeData[_reverse_nids[:blk.num_dst()]]
            out = torch.cat([out, blk.dstdata['h']], dim=1)

        with nvtx.annotate("output", color="blue"):
            out = self.w_out(out) # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            out = torch.nn.functional.relu(self.dropout(out))
            out = self.layer_norm(out) # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        return out


class TemporalAttnLaye_OptimizedLayer_Kernel(torch.nn.Module): # our tmpAttnLayer
    def __init__(self, ctx: TContext, num_heads: int,
                 dim_node: int, dim_edge: int, dim_time: int, dim_out: int,
                 dropout=0.1):
        """
        Initializes the Temporal Attention Layer for processing dynamic graphs with temporal features.
        This layer uses multi-head attention mechanism to incorporate node, edge, and time features.

        :param ctx: context object
        :param num_heads: number of heads
        :param dim_node: dimension of node features
        :param dim_edge: dimension of edge features
        :param dim_time: dimension of time features
        :param dim_out: dimension of output features
        :param dropout: dropout rate
        """
        super().__init__()
        assert (dim_out % num_heads == 0)
        self.ctx = ctx
        self.num_heads = num_heads
        self.dim_edge = dim_edge
        self.dim_out = dim_out
        self.time_encode = TimeEncode(dim_time)
        # 拆分后前反向的性能开销 TODO
        self.w_q = torch.nn.Linear(dim_node + dim_time, dim_out)
        self.w_kv = torch.nn.Linear(dim_node + dim_edge + dim_time, dim_out * 2)
        self.w_q_node = torch.nn.Linear(dim_node, dim_out)
        self.w_q_time = torch.nn.Linear(dim_time, dim_out)
        self.w_kv_node = torch.nn.Linear(dim_node, dim_out * 2)
        self.w_qkv_node = torch.nn.Linear(dim_node, dim_out * 3)
        self.w_kv_edge = torch.nn.Linear(dim_edge, dim_out * 2)
        self.w_kv_time = torch.nn.Linear(dim_time, dim_out * 2)
        self.w_out = torch.nn.Linear(dim_node + dim_out, dim_out)
        self.attn_act = torch.nn.LeakyReLU(0.2)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(dim_out)
    
    # def compute_z_ours(self, idx, nodeData, node_inverse, node_dst_inverse, efeat_unique, efeat_inverse, time_unique, time_inverse, time_dst_unique, time_dst_inverse):

    # def forward_redundancy_mul(self, blk: TBlock, nodeData, _reverse_nids, efeat_unique, _reverse_eids, _unique_time_delta, _reverse_time_delta): # TODO build pipeline
    #     # [DOING] 先拆分计算；纵切只要传入对应的indices范围即可
    #     # 处理bottleneck 的KV 部分
        
        
    # ! 去冗余的唯一行id，
    # nodeData, node_inverse = torch.unique(torch.cat(blk.srcdata['h'], blk.dstdata['h']), return_inverse=True)
    # efeat_unique, efeat_inverse = torch.unique(blk.efeat(), return_inverse=True)
    # # nbr_time_feat
    # time_unique, time_inverse = torch.unique(blk.time_deltas(), return_inverse=True)
    # # zero_time_feat
    # time_dst_unique, time_dst_inverse = TODO
    
    def forward(self, blk: TBlock, nodeData, _reverse_nids, efeat_unique, _reverse_eids, _unique_time_delta, _reverse_time_delta) -> Tensor:
        # attn, V = self.forward_redundancy_mul(blk, nodeData, _reverse_nids, efeat_unique, _reverse_eids, _unique_time_delta, _reverse_time_delta)
        with nvtx.annotate("forward_redundancy_mul", color="blue"):
            with nvtx.annotate("precompute", color="blue"):
                # no scatter
                # zero_time_feat = precomputed_zeros(self.ctx, blk.layer, self.time_encode, blk.num_dst()) # TODO 行全0 优化
                time_dst_unique = precomputed_zeros(self.ctx, blk.layer, self.time_encode, 1) 
                time_dst_inverse = torch.zeros(blk.num_dst(), dtype=torch.int64, device="cuda")
                # nbrs_time_feat_st = precomputed_times(self.ctx, blk.layer, self.time_encode, blk.time_deltas()) # TODO
                nbrs_time_feat = precomputed_times(self.ctx, blk.layer, self.time_encode, _unique_time_delta) # precompute是elementwise 理论上讲行不变

            # # Q = edge_view(blk, Q) # 对Q进行scatter
            # idx = torch.from_numpy(blk._g_dstindex)
            # idx = idx.to(device="cuda", dtype=torch.long)

            '''
            # 给具体可能的值 TODO
            # with torch.no_grad():
                # node_unique, node_inverse = torch.unique(blk.srcdata['h'], dim=0, return_inverse=True)
                # efeat_unique, efeat_inverse = torch.unique(blk.efeat(), dim=0, return_inverse=True) # 172的长度可能乘起来不够高效
                # time_unique, time_inverse = torch.unique(nbrs_time_feat_st, dim=0, return_inverse=True)
                # print(f"nbrs_time_feat_st.shape {nbrs_time_feat_st.shape}")
                # print(f"nbrs_time_feat_st = {nbrs_time_feat_st}")
                # print(f"nbrs_time_feat[_reverse_time_delta].shape {nbrs_time_feat[_reverse_time_delta].shape}")
                # print(f"nbrs_time_feat[_reverse_time_delta] = {nbrs_time_feat[_reverse_time_delta]}")
                # time_d_unique, time_d_inverse = torch.unique(blk.time_deltas(), dim=0, return_inverse=True)
                # print(f"_unique_time_delta.shape {_unique_time_delta[_reverse_time_delta].shape}")
                # print(f"_unique_time_delta[_reverse_time_delta] = {_unique_time_delta[_reverse_time_delta]}")
                # print(f"blk.time_deltas().shape {blk.time_deltas().shape}")
                # print(f"blk.time_deltas() = {blk.time_deltas()}")
                # print(f"_reverse_time_delta.shape {_reverse_time_delta.shape}")
                # print(f"_reverse_time_delta = {_reverse_time_delta}")
                # print(f"time_inverse.shape {time_inverse.shape}")
                # print(f"time_inverse = {time_inverse}")
                # print(f"nbrs_time_feat.shape {nbrs_time_feat.shape}")
                # print(f"nbrs_time_feat = {nbrs_time_feat}")
                # print(f"time_unique.shape {time_unique.shape}")
                # print(f"time_unique = {time_unique}")
                # assert torch.allclose(_unique_time_delta[_reverse_time_delta].to("cuda"), blk.time_deltas())
                # assert torch.allclose(nbrs_time_feat[_reverse_time_delta], nbrs_time_feat_st) # \O/ acc test passed!
                # print(f"time_unique {time_unique.shape}")
                # print(f"time_inverse = {time_inverse}")
                # print(f"time_delta_unique {time_d_unique.shape}")
                # print(f"time_delta_inverse = {time_d_inverse}")

                # node_dst_unique, node_dst_inverse = torch.unique(blk.dstdata['h'], dim=0, return_inverse=True)
                ## time_dst_unique_st, time_dst_inverse_st = torch.unique(zero_time_feat, dim=0, return_inverse=True) # 优先处理zero time feat
                ## assert torch.allclose(time_dst_unique, time_dst_unique_st)
                ## assert torch.allclose(time_dst_inverse, time_dst_inverse_st) # \O/ acc test passed!

                ## print("===") # TODO dst的冗余也有很多
                ## print(f"blk.dstdata['h'] {blk.dstdata['h'].shape}")
                ## print(f"node_dst_unique {node_dst_unique.shape}")
                ## print()
            '''

        # TODO 但是现在srcnode算的变多了，其实应该分开去重 -> 但矩阵乘的时间可以被pipeline掩盖/过于短的kernel对GPU而言也不友好，所以无所谓 -> 同一份nodeData 把两种W(三组W)拼在一起
        # ! 先以压缩为核心写kernel，即相同的只存储一次
        # return self.ctx.compiled_forward_redundancy_mul(idx, nodeData, _reverse_nids[blk.num_dst():], _reverse_nids[:blk.num_dst()], efeat_unique, _reverse_eids, nbrs_time_feat, _reverse_time_delta, time_dst_unique, time_dst_inverse)
        # def compute_z_ours(self, idx, nodeData, node_inverse, node_dst_inverse, efeat_unique, efeat_inverse, time_unique, time_inverse, time_dst_unique, time_dst_inverse):
        node_inverse = _reverse_nids[blk.num_dst():]
        node_dst_inverse = _reverse_nids[:blk.num_dst()]
        efeat_inverse = _reverse_eids
        time_unique = nbrs_time_feat
        time_inverse = _reverse_time_delta
        # print(f"nodeData.shape {nodeData.shape}")

        Q_node = self.w_q_node(nodeData)
        Q_time = self.w_q_time(time_dst_unique)

        Q_node = Q_node[node_dst_inverse]
        Q_time = Q_time[time_dst_inverse]
        Q_our = Q_node + Q_time
        
        Q = Q_our[blk._g_dstindex]


        Z_node = self.w_kv_node(nodeData)
        Z_edge = self.w_kv_edge(efeat_unique)
        Z_time = self.w_kv_time(time_unique)
        
        Z_node = Z_node[node_inverse]
        Z_edge = Z_edge[efeat_inverse]
        Z_time = Z_time[time_inverse]
        Z_our = Z_node + Z_edge + Z_time

        K = Z_our[:, :self.dim_out]
        V = Z_our[:, self.dim_out:]
        

        assert Q.shape[0] == K.shape[0]

        Q = torch.reshape(Q, (Q.shape[0], self.num_heads, -1))
        K = torch.reshape(K, (K.shape[0], self.num_heads, -1))
        V = torch.reshape(V, (V.shape[0], self.num_heads, -1))

        attn = torch.sum(Q * K, dim=2)
        attn = self.attn_act(attn)

        with nvtx.annotate("edge-softmax", color="blue"):
            with nvtx.annotate("edge-softmax", color="blue"):
                attn = edge_softmax(blk, attn)  # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            with nvtx.annotate("dropout", color="blue"):
                attn = self.dropout(attn)
            with nvtx.annotate("reshape", color="blue"):
                out = torch.reshape(V * attn[:, :, None], (V.shape[0], -1))  # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):

            with nvtx.annotate("edge-reduce", color="blue"):
                out = edge_reduce(blk, out, op='sum') # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):]
            # tail.dstdata['h'] = nfeat[:tail.num_dst()] + mem[:tail.num_dst()]
            blk.dstdata['h'] = nodeData[_reverse_nids[:blk.num_dst()]]
            out = torch.cat([out, blk.dstdata['h']], dim=1)

        with nvtx.annotate("output", color="blue"):
            out = self.w_out(out) # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            out = torch.nn.functional.relu(self.dropout(out))
            out = self.layer_norm(out) # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        return out


class TemporalAttnLaye_OptimizedLayer(torch.nn.Module): # our tmpAttnLayer
    def __init__(self, ctx: TContext, num_heads: int,
                 dim_node: int, dim_edge: int, dim_time: int, dim_out: int,
                 dropout=0.1):
        """
        Initializes the Temporal Attention Layer for processing dynamic graphs with temporal features.
        This layer uses multi-head attention mechanism to incorporate node, edge, and time features.

        :param ctx: context object
        :param num_heads: number of heads
        :param dim_node: dimension of node features
        :param dim_edge: dimension of edge features
        :param dim_time: dimension of time features
        :param dim_out: dimension of output features
        :param dropout: dropout rate
        """
        super().__init__()
        assert (dim_out % num_heads == 0)
        self.ctx = ctx
        self.num_heads = num_heads
        self.dim_edge = dim_edge
        self.dim_out = dim_out
        self.time_encode = TimeEncode(dim_time)
        # 拆分后前反向的性能开销 TODO
        # self.w_q = torch.nn.Linear(dim_node + dim_time, dim_out)
        # self.w_kv = torch.nn.Linear(dim_node + dim_edge + dim_time, dim_out * 2)
        self.w_q_node = torch.nn.Linear(dim_node, dim_out)
        self.w_q_time = torch.nn.Linear(dim_time, dim_out)
        self.w_kv_node = torch.nn.Linear(dim_node, dim_out * 2)
        self.w_qkv_node = torch.nn.Linear(dim_node, dim_out * 3)
        self.w_kv_edge = torch.nn.Linear(dim_edge, dim_out * 2)
        self.w_kv_time = torch.nn.Linear(dim_time, dim_out * 2)
        self.w_out = torch.nn.Linear(dim_node + dim_out, dim_out)
        self.attn_act = torch.nn.LeakyReLU(0.2)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(dim_out)
    
    # def compute_z_ours(self, idx, nodeData, node_inverse, node_dst_inverse, efeat_unique, efeat_inverse, time_unique, time_inverse, time_dst_unique, time_dst_inverse):

    # def forward_redundancy_mul(self, blk: TBlock, nodeData, _reverse_nids, efeat_unique, _reverse_eids, _unique_time_delta, _reverse_time_delta): # TODO build pipeline
    #     # [DOING] 先拆分计算；纵切只要传入对应的indices范围即可
    #     # 处理bottleneck 的KV 部分
        
        
    # ! 去冗余的唯一行id，
    # nodeData, node_inverse = torch.unique(torch.cat(blk.srcdata['h'], blk.dstdata['h']), return_inverse=True)
    # efeat_unique, efeat_inverse = torch.unique(blk.efeat(), return_inverse=True)
    # # nbr_time_feat
    # time_unique, time_inverse = torch.unique(blk.time_deltas(), return_inverse=True)
    # # zero_time_feat
    # time_dst_unique, time_dst_inverse = TODO
    
    def forward(self, blk: TBlock, nodeData, _reverse_nids, efeat_unique, _reverse_eids, _unique_time_delta, _reverse_time_delta) -> Tensor:
        # attn, V = self.forward_redundancy_mul(blk, nodeData, _reverse_nids, efeat_unique, _reverse_eids, _unique_time_delta, _reverse_time_delta)
        with nvtx.annotate("forward_redundancy_mul", color="blue"):
            with nvtx.annotate("precompute", color="blue"):
                # no scatter
                # zero_time_feat = precomputed_zeros(self.ctx, blk.layer, self.time_encode, blk.num_dst()) # TODO 行全0 优化
                time_dst_unique = precomputed_zeros(self.ctx, blk.layer, self.time_encode, 1) 
                time_dst_inverse = torch.zeros(blk.num_dst(), dtype=torch.int64, device="cuda")
                # nbrs_time_feat_st = precomputed_times(self.ctx, blk.layer, self.time_encode, blk.time_deltas()) # TODO
                nbrs_time_feat = precomputed_times(self.ctx, blk.layer, self.time_encode, _unique_time_delta) # precompute是elementwise 理论上讲行不变

            # # Q = edge_view(blk, Q) # 对Q进行scatter
            # idx = torch.from_numpy(blk._g_dstindex)
            # idx = idx.to(device="cuda", dtype=torch.long)

            '''
            # 给具体可能的值 TODO
            # with torch.no_grad():
                # node_unique, node_inverse = torch.unique(blk.srcdata['h'], dim=0, return_inverse=True)
                # efeat_unique, efeat_inverse = torch.unique(blk.efeat(), dim=0, return_inverse=True) # 172的长度可能乘起来不够高效
                # time_unique, time_inverse = torch.unique(nbrs_time_feat_st, dim=0, return_inverse=True)
                # print(f"nbrs_time_feat_st.shape {nbrs_time_feat_st.shape}")
                # print(f"nbrs_time_feat_st = {nbrs_time_feat_st}")
                # print(f"nbrs_time_feat[_reverse_time_delta].shape {nbrs_time_feat[_reverse_time_delta].shape}")
                # print(f"nbrs_time_feat[_reverse_time_delta] = {nbrs_time_feat[_reverse_time_delta]}")
                # time_d_unique, time_d_inverse = torch.unique(blk.time_deltas(), dim=0, return_inverse=True)
                # print(f"_unique_time_delta.shape {_unique_time_delta[_reverse_time_delta].shape}")
                # print(f"_unique_time_delta[_reverse_time_delta] = {_unique_time_delta[_reverse_time_delta]}")
                # print(f"blk.time_deltas().shape {blk.time_deltas().shape}")
                # print(f"blk.time_deltas() = {blk.time_deltas()}")
                # print(f"_reverse_time_delta.shape {_reverse_time_delta.shape}")
                # print(f"_reverse_time_delta = {_reverse_time_delta}")
                # print(f"time_inverse.shape {time_inverse.shape}")
                # print(f"time_inverse = {time_inverse}")
                # print(f"nbrs_time_feat.shape {nbrs_time_feat.shape}")
                # print(f"nbrs_time_feat = {nbrs_time_feat}")
                # print(f"time_unique.shape {time_unique.shape}")
                # print(f"time_unique = {time_unique}")
                # assert torch.allclose(_unique_time_delta[_reverse_time_delta].to("cuda"), blk.time_deltas())
                # assert torch.allclose(nbrs_time_feat[_reverse_time_delta], nbrs_time_feat_st) # \O/ acc test passed!
                # print(f"time_unique {time_unique.shape}")
                # print(f"time_inverse = {time_inverse}")
                # print(f"time_delta_unique {time_d_unique.shape}")
                # print(f"time_delta_inverse = {time_d_inverse}")

                # node_dst_unique, node_dst_inverse = torch.unique(blk.dstdata['h'], dim=0, return_inverse=True)
                ## time_dst_unique_st, time_dst_inverse_st = torch.unique(zero_time_feat, dim=0, return_inverse=True) # 优先处理zero time feat
                ## assert torch.allclose(time_dst_unique, time_dst_unique_st)
                ## assert torch.allclose(time_dst_inverse, time_dst_inverse_st) # \O/ acc test passed!

                ## print("===") # TODO dst的冗余也有很多
                ## print(f"blk.dstdata['h'] {blk.dstdata['h'].shape}")
                ## print(f"node_dst_unique {node_dst_unique.shape}")
                ## print()
            '''

        # TODO 但是现在srcnode算的变多了，其实应该分开去重 -> 但矩阵乘的时间可以被pipeline掩盖/过于短的kernel对GPU而言也不友好，所以无所谓 -> 同一份nodeData 把两种W(三组W)拼在一起
        # ! 先以压缩为核心写kernel，即相同的只存储一次
        # return self.ctx.compiled_forward_redundancy_mul(idx, nodeData, _reverse_nids[blk.num_dst():], _reverse_nids[:blk.num_dst()], efeat_unique, _reverse_eids, nbrs_time_feat, _reverse_time_delta, time_dst_unique, time_dst_inverse)
        # def compute_z_ours(self, idx, nodeData, node_inverse, node_dst_inverse, efeat_unique, efeat_inverse, time_unique, time_inverse, time_dst_unique, time_dst_inverse):
        node_inverse = _reverse_nids[blk.num_dst():]
        node_dst_inverse = _reverse_nids[:blk.num_dst()]
        efeat_inverse = _reverse_eids
        time_unique = nbrs_time_feat
        time_inverse = _reverse_time_delta
        # print(f"nodeData.shape {nodeData.shape}")

        Q_node = self.w_q_node(nodeData)
        Q_time = self.w_q_time(time_dst_unique)

        Q_node = Q_node[node_dst_inverse]
        Q_time = Q_time[time_dst_inverse]
        Q_our = Q_node + Q_time
        
        Q = Q_our[blk._g_dstindex]


        Z_node = self.w_kv_node(nodeData)
        Z_edge = self.w_kv_edge(efeat_unique)
        Z_time = self.w_kv_time(time_unique)
        
        Z_node = Z_node[node_inverse]
        Z_edge = Z_edge[efeat_inverse]
        Z_time = Z_time[time_inverse]
        Z_our = Z_node + Z_edge + Z_time

        K = Z_our[:, :self.dim_out]
        V = Z_our[:, self.dim_out:]
        

        assert Q.shape[0] == K.shape[0]

        Q = torch.reshape(Q, (Q.shape[0], self.num_heads, -1))
        K = torch.reshape(K, (K.shape[0], self.num_heads, -1))
        V = torch.reshape(V, (V.shape[0], self.num_heads, -1))

        attn = torch.sum(Q * K, dim=2)
        attn = self.attn_act(attn)

        with nvtx.annotate("edge-softmax", color="blue"):
            with nvtx.annotate("edge-softmax", color="blue"):
                attn = edge_softmax(blk, attn)  # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            with nvtx.annotate("dropout", color="blue"):
                attn = self.dropout(attn)
            with nvtx.annotate("reshape", color="blue"):
                out = torch.reshape(V * attn[:, :, None], (V.shape[0], -1))  # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):

            with nvtx.annotate("edge-reduce", color="blue"):
                out = edge_reduce(blk, out, op='sum') # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):]
            # tail.dstdata['h'] = nfeat[:tail.num_dst()] + mem[:tail.num_dst()]
            blk.dstdata['h'] = nodeData[_reverse_nids[:blk.num_dst()]]
            out = torch.cat([out, blk.dstdata['h']], dim=1)

        with nvtx.annotate("output", color="blue"):
            out = self.w_out(out) # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            out = torch.nn.functional.relu(self.dropout(out))
            out = self.layer_norm(out) # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        return out

class GATConv(nn.Module): # our gat layer
    def __init__(self,
                in_feats,
                out_feats,
                num_heads,
                feat_drop=0.,
                attn_drop=0.,
                negative_slope=0.2,
                residual=False,
                activation=None,
                bias=True
                ):
        super(GATConv,self).__init__()
        self.in_feats=in_feats
        self.out_feats=out_feats
        self.num_heads=num_heads
        self.W = nn.Parameter(torch.FloatTensor(in_feats, out_feats * num_heads))
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.negative_slope=negative_slope
        self.attn_drop=attn_drop
        self.feat_drop=nn.Dropout(feat_drop)
        if bias:
            self.bias=nn.Parameter(torch.FloatTensor(size=(num_heads*out_feats,)))
        else:
            self.register_buffer('bias',None)
        
        if residual:
            if in_feats!=out_feats*num_heads:
                self.res_fc=nn.Linear(in_feats,out_feats*num_heads,bias=False)
            else:
                self.res_fc=Identity()
        else:
            self.register_buffer('res_fc',None)
        
        self.reset_parameters()
        self.activation=activation
    
    
    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.W, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        

    def forward(self,row_ptr,col_ind,col_ptr,row_ind,permute,feat):

        h=torch.matmul(feat,self.W).view(-1,self.num_heads,self.out_feats)
        h=self.feat_drop(h)

        attn_row = (self.attn_l * h).sum(dim=-1)
        attn_col = (self.attn_r * h).sum(dim=-1)
        
        if not self.training:
            rst=GATConvFuse_inference(attn_row,attn_col,row_ptr,col_ind,self.negative_slope,h)
        else:
            rst=GATConvFuse(attn_row,attn_col,row_ptr,col_ind,col_ptr,row_ind,permute,self.negative_slope,h,self.attn_drop)
        
        if self.res_fc is not None:
            resval=self.res_fc(h).view(-1,self.num_heads,self.out_feats)
            rst=rst+resval
        
        if self.bias is not None:
            rst=rst+self.bias.view(-1,self.num_heads,self.out_feats)

        if self.activation:
            rst=self.activation(rst)

        return rst

