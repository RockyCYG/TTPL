import torch

def projection_10k(coor1):
    batch_size = coor1.shape[0]
    all_coors = coor1.clone()
    graph = all_coors[:, 1:, :]

    # Step 1: Calculate midpoints for translation
    midpoints = (torch.max(graph, 1).values + torch.min(graph, 1).values) / 2
    midpoints = torch.reshape(midpoints, (batch_size, 1, 2))
    all_coors = all_coors - midpoints  # translate

    # Step 2: Calculate the new range after translation
    range_x = torch.reshape(
        torch.max(graph[:, :, 0], 1).values - torch.min(graph[:, :, 0], 1).values,
        (-1, 1),
    )
    range_y = torch.reshape(
        torch.max(graph[:, :, 1], 1).values - torch.min(graph[:, :, 1], 1).values,
        (-1, 1),
    )
    range_val = torch.max(torch.cat((range_x, range_y), 1), 1).values
    range_val[range_val == 0] = 1  # Prevent division by zero

    # Step 3: Scale coordinates
    all_coors = all_coors / (torch.reshape(range_val, (batch_size, 1, 1)))

    # Step 4: Shift values to center in [0, 1]
    all_coors = all_coors + 0.5

    # Step 5: Clip the values to [0, 1]
    all_coors = torch.clamp(all_coors, 0, 1)

    return all_coors


def projection_5k(coor1: torch.tensor) -> torch.tensor:
    batch_size = coor1.shape[0]
    all_coors = coor1
    graph = all_coors[:, 1:, :]

    # Translate by the minimum values in the graph
    min_values = torch.reshape(torch.min(graph, 1).values, (batch_size, 1, 2))
    all_coors = all_coors - min_values  # translate

    # Apply a non-linear transformation
    all_coors = torch.tanh(all_coors)

    # Calculate scaling ratios with a slight modification
    ratio_x = torch.reshape(
        torch.max(graph[:, :, 0], 1).values - torch.min(graph[:, :, 0], 1).values,
        (-1, 1),
    )
    ratio_y = torch.reshape(
        torch.max(graph[:, :, 1], 1).values - torch.min(graph[:, :, 1], 1).values,
        (-1, 1),
    )
    ratio = torch.max(torch.cat((ratio_x, ratio_y), 1), 1).values

    # Avoid division by zero
    ratio[ratio == 0] = 1
    all_coors = all_coors / (torch.reshape(ratio, (batch_size, 1, 1)))

    # Post-process coordinates to ensure they are clipped within [0, 1]
    all_coors = torch.clip(all_coors, 0, 1)

    return all_coors


def projection_1k(coor1: torch.tensor) -> torch.tensor:
    batch_size = coor1.shape[0]
    all_coors = coor1
    graph = all_coors[:, 1:, :]

    # Translate to the maximum values
    max_values = torch.reshape(torch.max(graph, dim=1).values, (batch_size, 1, 2))
    all_coors = max_values - all_coors  # translate

    # Calculate ranges for normalization
    ratio_x = torch.reshape(
        torch.max(graph[:, :, 0], dim=1).values
        - torch.min(graph[:, :, 0], dim=1).values,
        (-1, 1),
    )
    ratio_y = torch.reshape(
        torch.max(graph[:, :, 1], dim=1).values
        - torch.min(graph[:, :, 1], dim=1).values,
        (-1, 1),
    )

    # Find the maximum scale factor
    ratio = torch.max(torch.cat((ratio_x, ratio_y), dim=1), dim=1).values
    ratio[ratio == 0] = 1  # Avoid division by zero

    # Normalize the coordinates
    all_coors = all_coors / (torch.reshape(ratio, (batch_size, 1, 1)))
    all_coors[ratio == 0, :, :] = (
        all_coors[ratio == 0, :, :] + max_values[ratio == 0, :, :]
    )

    # Clip to ensure values are within [0, 1]
    all_coors = torch.clip(all_coors, 0, 1)

    return all_coors