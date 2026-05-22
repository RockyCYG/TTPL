from dataclasses import dataclass
import os
import re
import numpy as np
import torch
from tqdm import tqdm


TSPLIB_BEST_KNOWN = {
    "ch150": 6528,
    "eil101": 629,
    "kroa100": 21282,
    "kroa200": 29368,
    "krob150": 26130,
    "kroc100": 20749,
    "kroe100": 22068,
    "pr124": 59030,
    "pr226": 80369,
    "pr299": 48191,
}


@dataclass
class Reset_State:
    """Data class for the reset state of the environment."""

    problems: torch.Tensor
    # shape: (batch, problem, 2)


@dataclass
class Step_State:
    """Data class for the state of each step in the environment."""

    data: torch.Tensor
    first_node: torch.Tensor
    current_node: torch.Tensor


class TSPEnv:
    """
    TSP Environment for Reinforcement Learning.
    Manages TSP problems, states, and rewards.
    """

    def __init__(self, **env_params):
        # Environment Parameters
        self.env_params = env_params
        self.problem_size = None
        self.data_path = env_params.get("data_path")
        self.sub_path = env_params.get("sub_path", False)

        # State Variables
        self.batch_size = None
        self.problems = None  # shape: (B, V, 2)
        self.first_node = None
        self.current_node = None
        self.selected_node_list = None
        self.selected_student_list = None
        self.selected_count = None

        # Data Loading
        self.raw_data_nodes = []
        self.raw_data_tours = []

        # TSPLIB Specific
        self.test_in_tsplib = env_params.get("test_in_tsplib", False)
        self.tsplib_path = env_params.get("tsplib_path")
        self.tsplib_cost = None
        self.tsplib_name = None
        self.tsplib_problems = None
        self.tsplib_edge_weight_type = "EUC_2D"
        self.tsplib_opt_costs = {
            str(name).lower(): float(cost)
            for name, cost in env_params.get("tsplib_opt_costs", {}).items()
        }
        self.problem_max_min = None
        self.episode = None

    def load_problems(self, episode, batch_size):
        """Load a batch of problems."""
        self.episode = episode
        self.batch_size = batch_size

        if not self.test_in_tsplib:
            self.problems = self.raw_data_nodes[episode : episode + batch_size]
            self.solution = self.raw_data_tours[episode : episode + batch_size]

            if self.sub_path:
                self.problems, self.solution = self.sampling_subpaths(
                    self.problems, self.solution, mode="train"
                )

            # Randomly flip the tour
            if torch.rand(1).item() < 0.5:
                self.solution = torch.flip(self.solution, dims=[1])
        else:
            self.tsplib_problems, self.tsplib_cost, self.tsplib_name = (
                self.make_tsplib_data(self.tsplib_path, episode)
            )
            device = torch.empty(0).device
            self.tsplib_cost = torch.tensor(
                self.tsplib_cost, dtype=torch.float32, device=device
            )
            self.problems = torch.from_numpy(
                self.tsplib_problems.reshape(1, -1, 2)
            ).to(device=device, dtype=torch.float32)

            if self.problems.shape[0] != batch_size:
                self.batch_size = self.problems.shape[0]
                batch_size = self.batch_size
            self.selected_node_list = torch.zeros(
                (batch_size, 0), dtype=torch.long, device=device
            )

            # Normalize problems
            self.problem_max_min = [torch.max(self.problems), torch.min(self.problems)]
            self.problems = (self.problems - self.problem_max_min[1]) / (
                self.problem_max_min[0] - self.problem_max_min[1]
            )
            self.solution = None

        self.problem_size = self.problems.shape[1]

    def sampling_subpaths(
        self, problems, solution, length_fix=False, mode="test", repair=False
    ):
        """Sample subpaths from the problems."""
        problems_size = problems.shape[1]
        batch_size = problems.shape[0]
        embedding_size = problems.shape[2]

        first_node_index = torch.randint(low=0, high=problems_size, size=(1,)).item()

        RRC_range = self.env_params.get("RRC_range", problems_size)

        # Length of subpath: uniform sampling
        if mode == "test":
            length_of_subpath = torch.randint(
                low=4, high=min(RRC_range, problems_size + 1), size=(1,)
            ).item()
        else:
            length_of_subpath = (
                problems_size
                if length_fix
                else torch.randint(
                    low=4, high=min(RRC_range, problems_size + 1), size=(1,)
                ).item()
            )

        # Create new solution
        double_solution = torch.cat([solution, solution], dim=-1)
        new_solution = double_solution[
            :, first_node_index : first_node_index + length_of_subpath
        ]
        new_solution_ascending, rank = torch.sort(new_solution, dim=-1)
        _, new_solution_rank = torch.sort(rank, dim=-1)

        # Create new problems from subpath
        index_2, _ = torch.sort(new_solution_ascending.repeat(1, 2).long(), dim=-1)
        index_1 = torch.arange(batch_size, dtype=torch.long)[:, None].expand_as(index_2)
        index_3 = (
            torch.arange(embedding_size, dtype=torch.long)[None, :]
            .expand(batch_size, embedding_size)
            .repeat(1, length_of_subpath)
        )

        new_data = problems[index_1, index_2, index_3].view(
            batch_size, length_of_subpath, 2
        )

        if repair:
            return (
                new_data,
                new_solution_rank,
                first_node_index,
                length_of_subpath,
                double_solution,
            )

        return new_data, new_solution_rank

    def load_raw_data(self, episode, begin_index=0):
        """Load raw data from file."""
        print("Loading raw dataset...")
        self.raw_data_nodes = []
        self.raw_data_tours = []

        with open(self.data_path, "r") as f:
            for line in tqdm(
                f.readlines()[begin_index : episode + begin_index], ascii=True
            ):
                parts = line.split(" ")
                output_index = parts.index("output")
                num_nodes = output_index // 2

                nodes = [
                    [float(parts[i]), float(parts[i + 1])]
                    for i in range(0, 2 * num_nodes, 2)
                ]
                self.raw_data_nodes.append(nodes)

                tour_nodes = [int(node) - 1 for node in parts[output_index + 1 : -1]]
                self.raw_data_tours.append(tour_nodes)

        self.raw_data_nodes = torch.tensor(self.raw_data_nodes, requires_grad=False)
        self.raw_data_tours = torch.tensor(self.raw_data_tours, requires_grad=False)
        print("Raw dataset loaded successfully!")

    def make_tsplib_data(self, filename, episode):
        """Load one TSPLIB instance.

        This supports both the repository's original one-line CSV format and
        standard TSPLIB .tsp files or directories of .tsp files.
        """
        standard_tsp_path = self._get_standard_tsplib_path(filename, episode)
        if standard_tsp_path is not None:
            coords, cost, name, edge_weight_type = self._read_standard_tsplib_file(
                standard_tsp_path
            )
            self.tsplib_edge_weight_type = edge_weight_type
            return (
                np.array([coords], dtype=float),
                np.array([cost], dtype=float),
                np.array([name], dtype=str),
            )

        instance_data = []
        cost = []
        instance_name = []
        self.tsplib_edge_weight_type = "EUC_2D"
        with open(filename, "r") as f:
            lines = f.readlines()
        for line in lines[episode : episode + 1]:
            line = line.rstrip("\n")
            line = line.replace("[", "")
            line = line.replace("]", "")
            line = line.replace("'", "")
            line = line.split(sep=",")
            line_data = np.array(line[2:], dtype=float).reshape(-1, 2)
            instance_data.append(line_data)
            cost.append(np.array(line[1], dtype=float))
            instance_name.append(np.array(line[0], dtype=str))
        instance_data = np.array(instance_data)
        cost = np.array(cost)
        instance_name = np.array(instance_name)

        return instance_data, cost, instance_name

    def _get_standard_tsplib_path(self, filename, episode):
        if os.path.isdir(filename):
            tsp_files = sorted(
                os.path.join(filename, name)
                for name in os.listdir(filename)
                if name.lower().endswith(".tsp")
            )
            if episode >= len(tsp_files):
                raise IndexError(
                    f"TSPLIB episode {episode} is out of range for {filename}"
                )
            return tsp_files[episode]

        if filename.lower().endswith(".tsp"):
            if episode != 0:
                raise IndexError(
                    f"Single TSPLIB file {filename} only supports episode 0"
                )
            return filename

        return None

    def _read_standard_tsplib_file(self, tsp_path):
        header = {}
        coords = []
        reading_coords = False

        with open(tsp_path, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                upper_line = line.upper()
                if upper_line == "NODE_COORD_SECTION":
                    reading_coords = True
                    continue
                if upper_line == "EOF":
                    break

                if reading_coords:
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    coords.append([float(parts[1]), float(parts[2])])
                else:
                    key, value = self._split_tsplib_header_line(line)
                    if key:
                        header[key] = value

        if not coords:
            raise ValueError(f"No NODE_COORD_SECTION found in {tsp_path}")

        name = header.get("NAME") or os.path.splitext(os.path.basename(tsp_path))[0]
        dimension = self._extract_int(header.get("DIMENSION"))
        if dimension is not None and dimension != len(coords):
            raise ValueError(
                f"{tsp_path} declares DIMENSION={dimension}, "
                f"but contains {len(coords)} coordinates"
            )

        edge_weight_type = header.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper()
        if edge_weight_type not in ("EUC_2D", "CEIL_2D"):
            raise ValueError(
                f"Unsupported TSPLIB EDGE_WEIGHT_TYPE={edge_weight_type!r} "
                f"in {tsp_path}"
            )

        cost = self._lookup_tsplib_cost(tsp_path, name, header)
        return np.array(coords, dtype=float), cost, name, edge_weight_type

    @staticmethod
    def _split_tsplib_header_line(line):
        if ":" in line:
            key, value = line.split(":", 1)
        else:
            parts = line.split(None, 1)
            if len(parts) != 2:
                return None, None
            key, value = parts
        return key.strip().upper(), value.strip()

    @staticmethod
    def _extract_int(value):
        if value is None:
            return None
        match = re.search(r"-?\d+", str(value))
        return int(match.group()) if match else None

    @staticmethod
    def _extract_float(value):
        if value is None:
            return None
        match = re.search(r"-?\d+(?:\.\d+)?", str(value))
        return float(match.group()) if match else None

    def _lookup_tsplib_cost(self, tsp_path, name, header):
        normalized_name = name.lower()
        if normalized_name in self.tsplib_opt_costs:
            return self.tsplib_opt_costs[normalized_name]

        for key in ("OPTIMAL_VALUE", "OPTIMUM", "BEST_KNOWN", "BEST_KNOWN_VALUE"):
            value = self._extract_float(header.get(key))
            if value is not None:
                return value

        for key, value in header.items():
            if "COMMENT" in key and re.search(r"opt|best|bks", value, re.IGNORECASE):
                parsed_value = self._extract_float(value)
                if parsed_value is not None:
                    return parsed_value

        sidecar_cost = self._read_tsplib_sidecar_cost(tsp_path)
        if sidecar_cost is not None:
            return sidecar_cost

        if normalized_name in TSPLIB_BEST_KNOWN:
            return float(TSPLIB_BEST_KNOWN[normalized_name])

        raise ValueError(
            f"No optimal/best-known cost found for TSPLIB instance {name!r}. "
            "Pass --tsplib_opt_costs name=value or add a sidecar .opt/.sol file."
        )

    def _read_tsplib_sidecar_cost(self, tsp_path):
        base_path, _ = os.path.splitext(tsp_path)
        for suffix in (".opt", ".opt.txt", ".sol", ".bks"):
            sidecar_path = base_path + suffix
            if not os.path.isfile(sidecar_path):
                continue
            with open(sidecar_path, "r") as f:
                for line in f:
                    value = self._extract_float(line)
                    if value is not None:
                        return value
        return None

    def destroy_solution(self, problem, complete_solution):
        """Destroy a part of the solution for repair."""
        (
            self.problems,
            self.solution,
            first_node_index,
            length_of_subpath,
            double_solution,
        ) = self.sampling_subpaths(
            problem, complete_solution, mode=self.env_params["mode"], repair=True
        )

        partial_solution_length = self._get_travel_distance_2(
            self.problems, self.solution, need_optimal=False
        )
        return (
            partial_solution_length,
            first_node_index,
            length_of_subpath,
            double_solution,
        )

    def reset(self, mode):
        """Reset the environment for a new episode."""
        self.selected_count = 0
        self.selected_node_list = torch.zeros((self.batch_size, 0), dtype=torch.long)
        self.selected_student_list = torch.zeros((self.batch_size, 0), dtype=torch.long)

        self.step_state = Step_State(
            data=self.problems, first_node=None, current_node=None
        )

        return Reset_State(self.problems), None, False

    def pre_step(self):
        """Prepare for a step."""
        return self.step_state, None, None, False

    def step(self, selected, selected_student):
        """Take a step in the environment."""
        self.selected_count += 1

        gather_index = selected[:, None, None].expand(-1, 1, 2)
        self.current_node = self.problems.gather(index=gather_index, dim=1).squeeze(1)

        if self.selected_count == 1:
            self.first_node = self.current_node

        self.selected_node_list = torch.cat(
            [self.selected_node_list, selected[:, None]], dim=1
        )
        self.selected_student_list = torch.cat(
            [self.selected_student_list, selected_student[:, None]], dim=1
        )

        self.step_state.current_node = self.current_node[:, None, :]
        if self.selected_count == 1:
            self.step_state.first_node = self.step_state.current_node

        done = self.selected_count == self.problems.shape[1]
        reward, reward_student = self._get_travel_distance() if done else (None, None)

        return self.step_state, reward, reward_student, done

    def _get_travel_distance(self):
        """Calculate the travel distance for the current tour."""
        if self.test_in_tsplib:
            travel_distances = self.tsplib_cost
            # Denormalize problems
            self.problems = (
                self.problems * (self.problem_max_min[0] - self.problem_max_min[1])
                + self.problem_max_min[1]
            )
        else:
            gathering_index = self.solution.unsqueeze(2).expand(
                self.batch_size, self.problems.shape[1], 2
            )
            ordered_seq = self.problems.gather(dim=1, index=gathering_index)
            travel_distances = self._calculate_tour_lengths(ordered_seq)

        # Calculate distance for the student model's tour
        gathering_index_student = self.selected_student_list.unsqueeze(2).expand(
            -1, self.problems.shape[1], 2
        )
        ordered_seq_student = self.problems.gather(dim=1, index=gathering_index_student)
        travel_distances_student = self._calculate_tour_lengths(ordered_seq_student)

        return travel_distances, travel_distances_student

    def _get_travel_distance_2(self, problems, solution, need_optimal=False):
        """Calculate travel distance for a given solution."""
        if self.test_in_tsplib:
            if need_optimal:
                return self.tsplib_cost, self.tsplib_name
            else:
                # Denormalize for distance calculation
                problems_copy = (
                    problems.clone().detach()
                    * (self.problem_max_min[0] - self.problem_max_min[1])
                    + self.problem_max_min[1]
                )
                gathering_index = solution.unsqueeze(2).expand(
                    problems_copy.shape[0], problems_copy.shape[1], 2
                )
                ordered_seq = problems_copy.gather(dim=1, index=gathering_index)
        else:
            gathering_index = solution.unsqueeze(2).expand(
                problems.shape[0], problems.shape[1], 2
            )
            ordered_seq = problems.gather(dim=1, index=gathering_index)

        return self._calculate_tour_lengths(ordered_seq)

    def _calculate_tour_lengths(self, ordered_seq):
        rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
        edge_lengths = ((ordered_seq - rolled_seq) ** 2).sum(2).sqrt()

        if self.test_in_tsplib:
            if self.tsplib_edge_weight_type == "EUC_2D":
                edge_lengths = torch.floor(edge_lengths + 0.5)
            elif self.tsplib_edge_weight_type == "CEIL_2D":
                edge_lengths = torch.ceil(edge_lengths)

        return edge_lengths.sum(1)
