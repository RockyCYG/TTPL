import torch
import torch.nn as nn
import torch.nn.functional as F


class TSPModel(nn.Module):
    """
    The main model for the Traveling Salesperson Problem.
    It consists of an encoder and a decoder.
    """

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.mode = model_params["mode"]
        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        self.encoded_nodes = None

    def forward(
        self, state, selected_node_list, solution, current_step, **kwargs
    ):
        """
        Forward pass of the model.

        Args:
            state: The current state of the environment.
            selected_node_list: The list of already selected nodes.
            solution: The ground truth solution (for training).
            current_step: The current step in the decoding process.
            repair: A boolean indicating if the model is in repair mode.

        Returns:
            A tuple containing the selected teacher node, probability, and selected student node.
        """
        batch_size_V = state.data.size(0)

        if self.mode == "train":
            probs = self.decoder(self.encoder(state.data), selected_node_list)
            selected_student = probs.argmax(dim=1)
            selected_teacher = solution[:, current_step - 1]
            prob = probs[
                torch.arange(batch_size_V)[:, None], selected_teacher[:, None]
            ].reshape(batch_size_V, 1)
        else:  # test mode
            probs = self.decoder(
                self.encoder,
                state.data,
                selected_node_list,
                **kwargs,
            )
            selected_student = probs.argmax(dim=1)
            selected_teacher = selected_student
            prob = 1

        return selected_teacher, prob, 1, selected_student


########################################
# ENCODER
########################################
class TSP_Encoder(nn.Module):
    """
    The encoder for the TSP model.
    It uses a linear embedding layer followed by multiple encoder layers.
    """

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        encoder_layer_num = 1
        self.embedding = nn.Linear(2, embedding_dim, bias=True)
        self.layers = nn.ModuleList(
            [EncoderLayer(**model_params) for _ in range(encoder_layer_num)]
        )

    def forward(self, data):
        """
        Forward pass of the encoder.

        Args:
            data: The input data (node coordinates).

        Returns:
            The encoded node embeddings.
        """
        embedded_input = self.embedding(data)
        out = embedded_input
        for layer in self.layers:
            out = layer(out)
        return out


class TSP_Decoder(nn.Module):
    """
    The decoder for the TSP model.
    It generates the solution sequence node by node.
    """

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        encoder_layer_num = self.model_params["decoder_layer_num"]

        self.embedding_first_node = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node = nn.Linear(embedding_dim, embedding_dim, bias=True)

        self.layers = nn.ModuleList(
            [DecoderLayer(**model_params) for _ in range(encoder_layer_num)]
        )

        self.k_1 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.Linear_final = nn.Linear(embedding_dim, 1, bias=True)

    def _get_new_data(self, data, selected_node_list, prob_size, B_V):
        """
        Get the data for the unselected nodes.
        """
        list = selected_node_list
        new_list = torch.arange(prob_size)[None, :].repeat(B_V, 1)
        new_list_len = prob_size - list.shape[1]
        index_2 = list.type(torch.long)
        index_1 = torch.arange(B_V, dtype=torch.long)[:, None].expand(
            B_V, index_2.shape[1]
        )
        new_list[index_1, index_2] = -2
        unselect_list = new_list[torch.gt(new_list, -1)].view(B_V, new_list_len)

        new_data = data
        emb_dim = data.shape[-1]
        new_data_len = new_list_len
        index_2_ = unselect_list.repeat_interleave(repeats=emb_dim, dim=1)
        index_1_ = torch.arange(B_V, dtype=torch.long)[:, None].expand(
            B_V, index_2_.shape[1]
        )
        index_3_ = torch.arange(emb_dim)[None, :].repeat(repeats=(B_V, new_data_len))
        new_data_ = new_data[index_1_, index_2_, index_3_].view(
            B_V, new_data_len, emb_dim
        )

        return new_data_

    def _get_encoding(self, encoded_nodes, node_index_to_pick):
        """
        Get the embeddings of specific nodes.
        """
        batch_size = node_index_to_pick.size(0)
        pomo_size = node_index_to_pick.size(1)
        embedding_dim = encoded_nodes.size(2)

        gathering_index = node_index_to_pick[:, :, None].expand(
            batch_size, pomo_size, embedding_dim
        )

        assert torch.all(
            (gathering_index >= 0) & (gathering_index < encoded_nodes.size(1))
        ), (
            f"Index out of bounds: gathering_index contains values up to {gathering_index.max().item()}, "
            f"but dimension 1 of encoded_nodes has size {encoded_nodes.size(1)}. "
            f"Check gathering_index generation logic. Device: {gathering_index.device}"
        )

        picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
        return picked_nodes

    def forward(self, encoder, data, selected_node_list, **kwargs):
        projection = kwargs.get("projection", None)
        MVDF = kwargs.get("MVDF", False)
        batch_size_V = data.shape[0]
        problem_size = data.shape[1]
        new_data = data

        left_node_coor = self._get_new_data(
            new_data, selected_node_list, problem_size, batch_size_V
        )
        left_node_num = left_node_coor.shape[1]

        first_and_last_node = self._get_encoding(
            new_data, selected_node_list[:, [0, -1]]
        )
        first_node_coor = first_and_last_node[:, [0]]
        last_node_coor = first_and_last_node[:, [1]]

        k_nearest_nodes = self.model_params["k_nearest_nodes"]
        knearest = self.model_params["knearest"]

        if knearest:
            if left_node_num > k_nearest_nodes:
                k = k_nearest_nodes
                distance1 = torch.norm(left_node_coor - last_node_coor, dim=2)
                _, sort_index = torch.topk(distance1, k=k, dim=1, largest=False)
                left_node_coor = (
                    self._get_encoding(left_node_coor, sort_index).clone().detach()
                )

        all_nodes_coor = torch.cat(
            (first_node_coor, left_node_coor, last_node_coor), dim=1
        )

        if self.model_params["coor_projection"] and (left_node_num > 1):
            all_nodes_coor = projection(all_nodes_coor)
            if MVDF:
                all_nodes_coor = MVDF_POMO(all_nodes_coor)

        encoded_nodes = encoder(all_nodes_coor)
        first_node_embed = encoded_nodes[:, [0], :]
        last_node_embed = encoded_nodes[:, [-1], :]
        left_encoded_node = encoded_nodes[:, 1:-1, :]

        embedded_first_node_ = self.embedding_first_node(first_node_embed)
        embedded_last_node_ = self.embedding_last_node(last_node_embed)

        out = torch.cat(
            (embedded_first_node_, left_encoded_node, embedded_last_node_), dim=1
        )

        for layer in self.layers:
            out = layer(out)

        out = self.Linear_final(out).squeeze(-1)
        out[:, [0, -1]] = out[:, [0, -1]] + float("-inf")
        if MVDF and self.model_params["coor_projection"] and (left_node_num > 1):
            out = out.reshape(-1, batch_size_V, out.size(1)).sum(0)

        props = F.softmax(out, dim=-1)
        props = props[:, 1:-1]

        index_small = torch.le(props, 1e-5)
        props_clone = props.clone()
        props_clone[index_small] = props_clone[index_small] + torch.tensor(
            1e-7, dtype=props_clone[index_small].dtype
        )
        props = props_clone

        if knearest:
            if left_node_num > k_nearest_nodes:
                new_props = torch.zeros(batch_size_V, left_node_num)
                index_1_ = torch.arange(batch_size_V, dtype=torch.long)[:, None].expand(
                    batch_size_V, k
                )
                index_2_ = sort_index.type(torch.long)
                new_props[index_1_, index_2_] = props.reshape(
                    batch_size_V, sort_index.shape[1]
                )
                props = new_props

        new_props = torch.zeros(batch_size_V, problem_size)
        index_1_ = torch.arange(batch_size_V, dtype=torch.long)[:, None].expand(
            batch_size_V, selected_node_list.shape[1]
        )
        index_2_ = selected_node_list.type(torch.long)
        new_props[index_1_, index_2_] = -2
        index = torch.gt(new_props, -1).view(batch_size_V, -1)
        new_props[index] = props.ravel()

        return new_props


class EncoderLayer(nn.Module):
    """
    A single layer of the encoder.
    It consists of a multi-head attention layer and a feed-forward layer.
    """

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        head_num = self.model_params["head_num"]
        qkv_dim = self.model_params["qkv_dim"]

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.feedForward = Feed_Forward_Module(**model_params)

    def forward(self, input1):
        head_num = self.model_params["head_num"]
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)
        out1 = input1 + multi_head_out
        out2 = self.feedForward(out1)
        out3 = out1 + out2
        return out3


class DecoderLayer(nn.Module):
    """
    A single layer of the decoder.
    It consists of a multi-head attention layer and a feed-forward layer.
    """

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        head_num = self.model_params["head_num"]
        qkv_dim = self.model_params["qkv_dim"]

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.feedForward = Feed_Forward_Module(**model_params)

    def forward(self, input1):
        head_num = self.model_params["head_num"]
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)
        out1 = input1 + multi_head_out
        out2 = self.feedForward(out1)
        out3 = out1 + out2
        return out3


def reshape_by_heads(qkv, head_num):
    """
    Reshapes the query, key, or value tensor for multi-head attention.
    """
    batch_s, n, _ = qkv.size()
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)
    return q_transposed


def multi_head_attention(q, k, v):
    """
    Multi-head attention mechanism.
    """
    batch_s, head_num, n, key_dim = q.size()
    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    weights = nn.Softmax(dim=3)(score_scaled)
    out = torch.matmul(weights, v)
    out_transposed = out.transpose(1, 2)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    return out_concat


class Feed_Forward_Module(nn.Module):
    """
    A feed-forward module used in the encoder and decoder layers.
    """

    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        ff_hidden_dim = model_params["ff_hidden_dim"]
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))


def MVDF_POMO(problems):
    # problems.shape: (batch, problem, 2)
    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)
    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)
    MVDF_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)
    return MVDF_problems
