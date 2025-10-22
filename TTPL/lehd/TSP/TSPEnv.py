from dataclasses import dataclass
import numpy as np
import torch
from tqdm import tqdm


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
            self.tsplib_cost = torch.tensor(self.tsplib_cost)
            self.problems = (
                torch.from_numpy(self.tsplib_problems.reshape(1, -1, 2)).cuda().float()
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
        instance_data = []
        cost = []
        instance_name = []
        for line in open(filename, "r").readlines()[episode : episode + 1]:
            line = line.rstrip("\n")
            line = line.replace("[", "")
            line = line.replace("]", "")
            line = line.replace("'", "")
            line = line.split(sep=",")
            line_data = np.array(line[2:], dtype=float).reshape(-1, 2)
            instance_data.append(line_data)
            cost.append(np.array(line[1], dtype=float))
            instance_name.append(np.array(line[0], dtype=str))
        instance_data = np.array(
            instance_data
        )
        cost = np.array(cost)
        instance_name = np.array(instance_name)

        return instance_data, cost, instance_name

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
            rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
            travel_distances = ((ordered_seq - rolled_seq) ** 2).sum(2).sqrt().sum(1)

        # Calculate distance for the student model's tour
        gathering_index_student = self.selected_student_list.unsqueeze(2).expand(
            -1, self.problems.shape[1], 2
        )
        ordered_seq_student = self.problems.gather(dim=1, index=gathering_index_student)
        rolled_seq_student = ordered_seq_student.roll(dims=1, shifts=-1)
        travel_distances_student = (
            ((ordered_seq_student - rolled_seq_student) ** 2).sum(2).sqrt().sum(1)
        )

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

        rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
        travel_distances = ((ordered_seq - rolled_seq) ** 2).sum(2).sqrt().sum(1)
        return travel_distances
