# TSP Template
template_program_tsp = '''
import torch

def normalize(coor1: torch.tensor) -> torch.tensor:
    """
    Args:
        coor1: coordinates of nodes, shape: (batch, 1+k+1, 2)

    Return:
        all_coors: a tensor that containing normalized coordinates of nodes, shape: (batch, 1+k+1, 2)

    Note:
        first_node: coor1[:,[0],:], left_node: coor1[:,1:-1,:], last_node: [:,[-1],:]. left_node is the topk close to the last_node

    # This is a template function, you can develop new function based on it.
    batch_size = coor1.shape[0]
    all_coors = coor1
    graph = all_coors[:, 1:, :]
    min_values = torch.reshape(torch.min(graph, 1).values, (batch_size, 1, 2))
    all_coors = all_coors - min_values  # translate
    ratio_x = torch.reshape(torch.max(graph[:, :, 0], 1).values - torch.min(graph[:, :, 0], 1).values, (-1, 1))
    ratio_y = torch.reshape(torch.max(graph[:, :, 1], 1).values - torch.min(graph[:, :, 1], 1).values, (-1, 1))
    ratio = torch.max(torch.cat((ratio_x, ratio_y), 1), 1).values
    ratio[ratio == 0] = 1
    all_coors = all_coors / (torch.reshape(ratio, (batch_size, 1, 1)))
    all_coors[ratio == 0, :, :] = all_coors[ratio == 0, :, :] + min_values[ratio == 0, :, :]
    all_coors = torch.clip(all_coors, 0, 1)
    """
    return all_coors
'''

task_description_tsp = """
                    I need help designing an innovative coordinate normalize strategy function implemented in PyTorch to normalize a set of nodes' coordinate, aiming to
                    maximize the final negative gap . The input is a tensor with shape (batch, num_nodes, 2)
                    and you must keep the 'all_coors' be a tensor with the same shape as 'coor1'. Avoid use 'for' to deal with batch"""


# POMO TSP Logit-Bias Template
template_program_tsp_logit_bias = '''
import torch

def make_logit_bias(coords: torch.Tensor, current_node: torch.Tensor, ninf_mask: torch.Tensor, step: int) -> torch.Tensor:
    """
    Args:
        coords: normalized coordinates of all nodes, shape: (batch, problem, 2)
        current_node: current node index for each POMO rollout, shape: (batch, pomo)
        ninf_mask: visited-node mask, shape: (batch, pomo, problem). Visited nodes are -inf.
        step: current decoding step, starting from 1 after the first fixed POMO move.

    Return:
        bias: additive decoder score bias, shape: (batch, pomo, problem)

    # This is a template function. It implements the fixed log-distance prior:
    # nearest topK candidates use -log(distance), while the rest use -distance.
    # You can develop a better inference-time bias based on distance, mask, step,
    # problem size, and smooth scaling, but keep the output shape unchanged.
    batch_size, problem_size, _ = coords.shape
    pomo_size = current_node.shape[1]
    gather_index = current_node[:, :, None].expand(batch_size, pomo_size, 2)
    current_xy = coords.gather(dim=1, index=gather_index)
    dist = torch.cdist(current_xy, coords).clamp_min(1e-6)
    valid_mask = ninf_mask == 0

    topk = min(20, problem_size)
    ranked_dist = dist.masked_fill(~valid_mask, float("inf"))
    topk_index = torch.topk(ranked_dist, k=topk, dim=2, largest=False).indices
    topk_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    topk_mask.scatter_(dim=2, index=topk_index, value=True)

    near_bias = -torch.log(dist)
    far_bias = -dist
    bias = torch.where(topk_mask, near_bias, far_bias)
    bias = bias.masked_fill(~valid_mask, 0.0)
    bias = torch.where(torch.isfinite(bias), bias, torch.zeros_like(bias))
    bias = 0.1 * bias
    """
    return bias
'''

task_description_tsp_logit_bias = """
                    I need help designing an inference-time additive logit-bias function for a POMO-style TSP decoder.
                    The function is implemented in PyTorch and receives normalized full-graph coordinates, the current
                    node of each POMO rollout, the visited-node mask, and the decoding step. It must return a tensor with
                    shape (batch, pomo, problem) that will be added to decoder scores before tanh clipping and masking.
                    Good solutions should use geometric information such as current-to-candidate distance, nearest-neighbor
                    structure, decoding step, and problem size, while avoiding Python loops over the batch or rollout axes.
                    The goal is to maximize final negative gap, so lower TSP tour gap is better."""


# CVRP Template
template_program_cvrp = '''
import torch

def normalize(coor1: torch.Tensor, coor2: torch.Tensor, coor3: torch.Tensor) -> torch.Tensor:
    """
    Args:
        coor1: indicate the first node, shape: (batch, 1, 2)
        coor2: coordinates of the rest of nodes, shape: (batch, 100, 2)
        coor3: coordinate of the last node, shape: (batch, 1, 2)

    Return:
        coor1: normalized coordinate of the first node, shape: (batch, 1, 2)
        coor2: normalized coordinates of the rest of nodes, shape: (batch, left_num, 2)
        coor3: normalized coordinate of the last node, shape: (batch, 1, 2)

    # This is a template function, you can develop new function based on it.
    lengths = [coor1.shape[1], coor2.shape[1], coor3.shape[1]]
    all_coors = torch.cat((coor1, coor2, coor3), dim=1)
    last_neighbors_xy = all_coors[:, 1:, :]
    # shape: (batch, 1+neighbor_k, 2)
    xy_max = torch.max(last_neighbors_xy, dim=1, keepdim=True).values
    xy_min = torch.min(last_neighbors_xy, dim=1, keepdim=True).values
    # shape: (batch, 1, 2)
    ratio = torch.max((xy_max - xy_min), dim=-1, keepdim=True).values
    ratio[ratio == 0] = 1
    # shape: (batch, 1, 1)
    all_coors = torch.clip((all_coors - xy_min) / ratio.expand(-1, 1, 2), 0, 1)
    coor1, coor2, coor3 = torch.split(all_coors, lengths, dim=1)
    """

    return coor1, coor2, coor3
'''

task_description_cvrp = """I need to develop a coordinate normalization method for CVRP sequences that preserves critical geometric relationships between nodes while enabling effective neural network processing. The function should take depot/vehicle coordinates (coor1), customer nodes (coor2), and final stop coordinates (coor3) as PyTorch tensors, maintaining their original shapes. Key objectives are: 1) Establish spatial consistency across batches without distorting relative positions, 2) Use the initial node as an anchor point for stable reference, 3) Prevent information loss from hard clipping while controlling magnitude variance, and 4) Ensure scale-invariant features that help the downstream model generalize across problem sizes. The solution should particularly focus on maintaining directional relationships and proportional distances rather than absolute positional constraints."""


# Function to get template and description based on problem type
def get_template_and_description(problem_type: str):
    """
    Get template program and task description based on problem type.

    Args:
        problem_type: Either 'tsp' or 'cvrp'

    Returns:
        tuple: (template_program, task_description)
    """
    problem_type = problem_type.lower()
    if problem_type == "tsp":
        return template_program_tsp, task_description_tsp
    elif problem_type in ("tsp_logit_bias", "tsp_bias"):
        return template_program_tsp_logit_bias, task_description_tsp_logit_bias
    elif problem_type == "cvrp":
        return template_program_cvrp, task_description_cvrp
    else:
        raise ValueError(
            f"Unknown problem type: {problem_type}. Must be 'tsp' or 'cvrp'."
        )


# Default values (for backward compatibility)
template_program = template_program_tsp
task_description = task_description_tsp
