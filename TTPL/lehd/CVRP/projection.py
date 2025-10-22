import torch
def projection_10k(
    coor1: torch.Tensor, coor2: torch.Tensor, coor3: torch.Tensor
) -> torch.Tensor:
    lengths = [coor1.shape[1], coor2.shape[1], coor3.shape[1]]
    all_coors = torch.cat((coor1, coor2, coor3), dim=1)

    # Centering the coordinates around the first node
    center = coor1.squeeze(1)  # shape: (batch, 2)
    relative_coors = all_coors - center.unsqueeze(1)  # shape: (batch, 1 + k, 2)

    # Calculate distances from the first node
    distances = torch.norm(
        relative_coors, dim=-1, keepdim=True
    )  # shape: (batch, 1 + k, 1)

    # Apply a non-linear transformation to the distances (e.g., exponential scaling)
    transformed_distances = torch.exp(distances) - 1  # shape: (batch, 1 + k, 1)

    # Scale transformed distances to [0, 1]
    max_distance = torch.max(transformed_distances, dim=1, keepdim=True).values
    max_distance[max_distance == 0] = 1  # Prevent division by zero
    normalized_distances = (
        transformed_distances / max_distance
    )  # shape: (batch, 1 + k, 1)

    # Maintain direction by re-constructing relative coordinates
    direction = relative_coors / (distances + 1e-6)  # shape: (batch, 1 + k, 2)
    normalized_coors = normalized_distances * direction  # shape: (batch, 1 + k, 2)

    all_coors = normalized_coors + center.unsqueeze(
        1
    )  # Translate back to original position

    coor1, coor2, coor3 = torch.split(all_coors, lengths, dim=1)

    return coor1, coor2, coor3


def projection_1k(
    coor1: torch.Tensor, coor2: torch.Tensor, coor3: torch.Tensor
) -> torch.Tensor:
    lengths = [coor1.shape[1], coor2.shape[1], coor3.shape[1]]
    all_coors = torch.cat((coor1, coor2, coor3), dim=1)

    # Calculate the relative vectors
    relative_vectors = all_coors - coor1

    # Compute the magnitudes of the vectors
    magnitudes = torch.norm(relative_vectors, dim=-1, keepdim=True)

    # Avoid division by zero
    magnitudes[magnitudes == 0] = 1

    # Normalize the vectors to unit vectors
    unit_vectors = relative_vectors / magnitudes

    # Scale unit vectors by the normalized magnitudes
    normalized_vectors = unit_vectors * (
        magnitudes / magnitudes.max(dim=1, keepdim=True).values
    )

    # Combine normalized vectors with the anchor point
    normalized_coors = coor1 + normalized_vectors

    coor1, coor2, coor3 = torch.split(normalized_coors, lengths, dim=1)

    return coor1, coor2, coor3


def projection_5k(
    coor1: torch.Tensor, coor2: torch.Tensor, coor3: torch.Tensor
) -> torch.Tensor:
    lengths = [coor1.shape[1], coor2.shape[1], coor3.shape[1]]
    all_coors = torch.cat((coor1, coor2, coor3), dim=1)

    # Get anchor point (coor1)
    anchor = coor1

    # Translate coordinates based on the anchor
    translated_coors = all_coors - anchor

    # Calculate distances to the anchor point
    distances = torch.norm(translated_coors, p=2, dim=-1, keepdim=True)

    # Find the furthest distance for scaling
    farthest_distance, _ = distances.max(dim=1, keepdim=True)
    scaling_factor = torch.sqrt(farthest_distance)

    scaling_factor[scaling_factor == 0] = 1  # Prevent division by zero

    # Normalize the coordinates using the scaling factor
    normalized_coors = translated_coors / scaling_factor.expand_as(translated_coors)

    # Combine back with the anchor
    normalized_coors += anchor

    # Split back to original coordinates
    coor1, coor2, coor3 = torch.split(normalized_coors, lengths, dim=1)

    return coor1, coor2, coor3
