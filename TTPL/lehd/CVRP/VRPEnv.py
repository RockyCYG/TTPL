import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

from tqdm import tqdm


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)


@dataclass
class Step_State:
    problems: torch.Tensor
    # shape: (batch, pomo)
    first_node: torch.Tensor
    current_node: torch.Tensor
    # shape: (batch, pomo)


class VRPEnv:
    def __init__(self, **env_params):
        ####################################
        self.env_params = env_params
        self.problem_size = None

        self.data_path = env_params["data_path"]
        self.sub_path = env_params["sub_path"]
        ####################################
        self.batch_size = None
        self.problems = None
        self.first_node = None

        self.loc_all = (
            []
        )  # the first node is depot, others are customers. shape (B,V+1,2)
        self.demand_all = (
            []
        )  # the first one is depot's demand, which is set to zero. shape (B,V+1)
        self.capacity_all = []  # shape (B)
        self.cost_all = []  # shape (B)
        self.solution_all = []  # shape (B,V,2)
        self.duration_all = []
        self.start_capacity = None

        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        self.selected_student_list = None

        self.test_in_vrplib = env_params["test_in_vrplib"]
        self.vrplib_path = env_params["vrplib_path"]
        self.vrplib_cost = None
        self.vrplib_name = None
        self.vrplib_problems = None
        self.problem_max_min = None
        self.episode = None

        # shape: (batch, pomo, 0~problem)

    @staticmethod
    def _split_node_flag(node_flag):
        """Convert a flattened node/flag array into a two-column representation."""

        value_count = len(node_flag) // 2
        return [[node_flag[i], node_flag[i + value_count]] for i in range(value_count)]

    def _parse_dataset_line(self, raw_line):
        """Parse a single dataset line into structured tensors."""

        tokens = raw_line.split(",")

        depot_index = int(tokens.index("depot"))
        customer_index = int(tokens.index("customer"))
        capacity_index = int(tokens.index("capacity"))
        demand_index = int(tokens.index("demand"))
        cost_index = int(tokens.index("cost"))
        node_flag_index = int(tokens.index("node_flag"))

        depot = [[float(tokens[depot_index + 1]), float(tokens[depot_index + 2])]]
        customers = [
            [float(tokens[idx]), float(tokens[idx + 1])]
            for idx in range(customer_index + 1, capacity_index, 2)
        ]
        locations = depot + customers

        capacity = int(float(tokens[capacity_index + 1]))
        demands_raw = [int(tokens[idx]) for idx in range(demand_index + 1, cost_index)]
        if int(tokens[demand_index + 1]) == 0:
            demand = demands_raw
        else:
            demand = [0] + demands_raw

        cost = float(tokens[cost_index + 1])
        node_flag_flat = [
            int(tokens[idx]) for idx in range(node_flag_index + 1, len(tokens))
        ]
        node_flag = self._split_node_flag(node_flag_flat)

        return locations, capacity, demand, cost, node_flag

    def _load_dataset_segment(self, lines, show_progress=True):
        """Load a portion of the dataset into tensors."""

        iterator = tqdm(lines, ascii=True) if show_progress else lines

        nodes = []
        capacities = []
        demands = []
        costs = []
        node_flags = []

        for raw_line in iterator:
            loc, capacity, demand, cost, node_flag = self._parse_dataset_line(raw_line)
            nodes.append(loc)
            capacities.append(capacity)
            demands.append(demand)
            costs.append(cost)
            node_flags.append(node_flag)

        if not nodes:
            return None

        nodes_tensor = torch.tensor(nodes, dtype=torch.float32, requires_grad=False)
        capacities_tensor = torch.tensor(
            capacities, dtype=torch.float32, requires_grad=False
        )
        demands_tensor = torch.tensor(demands, dtype=torch.float32, requires_grad=False)
        costs_tensor = torch.tensor(costs, dtype=torch.float32, requires_grad=False)
        node_flags_tensor = torch.tensor(
            node_flags, dtype=torch.long, requires_grad=False
        )

        return (
            nodes_tensor,
            capacities_tensor,
            demands_tensor,
            costs_tensor,
            node_flags_tensor,
        )

    def load_problems(
        self,
        episode,
        batch_size,
    ):
        self.episode = episode
        self.batch_size = batch_size

        if not self.test_in_vrplib:

            self.problems_nodes = self.raw_data_nodes[episode : episode + batch_size]
            # shape (B,V+1,2)
            self.Batch_demand = self.raw_data_demand[episode : episode + batch_size]
            # shape (B,V+1)

            self.Batch_capacity = self.raw_data_capacity[episode : episode + batch_size]

            self.solution = self.raw_data_node_flag[episode : episode + batch_size]
            # shape (B,V,2)
            self.Batch_capacity = self.Batch_capacity[:, None].repeat(
                1, self.solution.shape[1] + 1
            )
            # shape (B,V+1)

            self.problems = torch.cat(
                (
                    self.problems_nodes,
                    self.Batch_demand[:, :, None],
                    self.Batch_capacity[:, :, None],
                ),
                dim=2,
            )
            # shape (B,V+1,4)

            if self.sub_path:
                self.problems, self.solution = self.sampling_subpaths(
                    self.problems, self.solution
                )

        else:

            self.cvrp_node_coords = self.cvrp_node_coords_all[episode]
            self.cvrp_demands = self.cvrp_demands_all[episode]
            self.cvrp_capacitys = self.cvrp_capacitys_all[episode]
            self.vrplib_cost = self.vrplib_cost_all[episode]
            self.vrplib_name = self.vrplib_name_all[episode]

            problem_nodes = self.cvrp_node_coords
            problem_demands = self.cvrp_demands
            problem_capacitys = self.cvrp_capacitys

            problem_size = len(problem_nodes)

            problem_nodes = np.array(problem_nodes).reshape(1, problem_size, 2)
            problem_demands = np.array(problem_demands).reshape(1, problem_size, 1)

            problem_nodes = torch.from_numpy(problem_nodes).cuda().float()

            self.problem_max_min = [torch.max(problem_nodes), torch.min(problem_nodes)]
            problem_nodes = (problem_nodes - self.problem_max_min[1]) / (
                self.problem_max_min[0] - self.problem_max_min[1]
            )

            problem_demands = torch.from_numpy(problem_demands).cuda().float()

            capacity_repeat = (
                torch.tensor([problem_capacitys])
                .cuda()
                .float()
                .unsqueeze(0)
                .unsqueeze(1)
                .repeat(1, problem_size, 1)
            )
            self.raw_data_capacity = capacity_repeat
            self.problems = torch.cat(
                (problem_nodes, problem_demands, capacity_repeat), dim=2
            )

            self.solution = None

        self.problem_size = self.problems.shape[1] - 1

    def vrp_whole_and_solution_subrandom_inverse(self, solution, lib_flag):

        clockwise_or_not = torch.rand(1)[0]

        if clockwise_or_not >= 0.5:
            solution = torch.flip(solution, dims=[1])
            index = torch.arange(solution.shape[1]).roll(shifts=1)
            solution[:, :, 1] = solution[:, index, 1]

        return solution

    def vrp_whole_and_solution_subrandom_shift_V2inverse(self, solution):
        """
        For each instance, shift randomly so that different end_with depot nodes can reach the last digit.
        """

        problem_size = solution.shape[1]
        batch_size = solution.shape[0]

        start_from_depot = solution[:, :, 1].nonzero()
        end_with_depot = start_from_depot.clone()
        end_with_depot[:, 1] = end_with_depot[:, 1] - 1
        end_with_depot[end_with_depot.le(-0.5)] = solution.shape[1] - 1
        end_with_depot[:, 1] = torch.roll(end_with_depot[:, 1], dims=0, shifts=-1)
        visit_depot_num = solution[:, :, 1].sum(1)
        min_length = torch.min(visit_depot_num)

        first_node_index = torch.randint(low=0, high=min_length, size=[1])[
            0
        ]  # in [0,N)

        temp_tri = np.triu(np.ones((len(visit_depot_num), len(visit_depot_num))), k=1)
        visit_depot_num_numpy = visit_depot_num.clone().cpu().numpy()

        temp_index = np.dot(visit_depot_num_numpy, temp_tri)
        temp_index_torch = torch.from_numpy(temp_index).long().cuda()

        pick_end_with_depot_index = temp_index_torch + first_node_index
        pick_end_with_depot_ = end_with_depot[pick_end_with_depot_index][:, 1]
        first_index = pick_end_with_depot_
        end_indeex = pick_end_with_depot_ + problem_size

        index = torch.arange(2 * problem_size)[None, :].repeat(batch_size, 1)
        x1 = index > first_index[:, None]
        x2 = index <= end_indeex[:, None]
        x3 = x1.int() * x2.int()
        double_solution = solution.repeat(1, 2, 1)
        solution = double_solution[x3.gt(0.5)[:, :, None].repeat(1, 1, 2)].reshape(
            batch_size, problem_size, 2
        )

        return solution

    def sampling_subpaths(self, problems, solution, length_fix=False):
        # problems shape (B,V+1,4)
        # solution shape (B,V,2)

        # step:
        # 1.Extract subtour

        problems_size = problems.shape[1] - 1

        batch_size = problems.shape[0]
        embedding_size = problems.shape[2]

        # the first node of subpath: uniform sampling, from 0 to N
        length_of_subpath = torch.randint(
            low=4, high=min(problems_size, self.env_params["RRC_range"]) + 1, size=[1]
        )[
            0
        ]  # in [4,V]

        solution = self.vrp_whole_and_solution_subrandom_inverse(solution)
        solution = self.vrp_whole_and_solution_subrandom_shift_V2inverse(solution)
        #  Find the points that start from the depot, then subtract 1 to get the point that ends with the depot

        start_from_depot = solution[:, :, 1].nonzero()

        end_with_depot = start_from_depot
        end_with_depot[:, 1] = end_with_depot[:, 1] - 1
        end_with_depot[end_with_depot.le(-0.5)] = solution.shape[1] - 1

        visit_depot_num = torch.sum(solution[:, :, 1], dim=1)

        p = torch.rand(len(visit_depot_num))
        select_end_with_depot_node_index = p * visit_depot_num
        select_end_with_depot_node_index = torch.floor(
            select_end_with_depot_node_index
        ).long()

        temp_tri = np.triu(np.ones((len(visit_depot_num), len(visit_depot_num))), k=1)
        visit_depot_num_numpy = visit_depot_num.clone().cpu().numpy()

        temp_index = np.dot(visit_depot_num_numpy, temp_tri)
        temp_index_torch = torch.from_numpy(temp_index).long().cuda()
        select_end_with_depot_node_index_ = (
            select_end_with_depot_node_index + temp_index_torch
        )

        # This is the point at which each instance is randomly selected with an end with depot
        select_end_with_depot_node = end_with_depot[
            select_end_with_depot_node_index_, 1
        ]

        double_solution = torch.cat((solution, solution), dim=1)

        select_end_with_depot_node = select_end_with_depot_node + problems_size

        indexx = torch.arange(length_of_subpath).repeat(batch_size, 1)
        offset = select_end_with_depot_node - length_of_subpath + 1

        indexxxx = indexx + offset[:, None]

        sub_tour = double_solution[:, indexxxx, :]

        sub_tour = sub_tour.view(-1, length_of_subpath, 2)

        index_1 = torch.arange(0, batch_size * batch_size, batch_size)
        index_2 = torch.arange(batch_size)
        index_3 = index_1 + index_2
        sub_solution = sub_tour[index_3, :, :]

        # Calculate the capacity of the first point

        offset_index = problems.shape[0]
        start_index = indexxxx[:, 0]

        x1 = (
            torch.arange(double_solution[:offset_index, :, 1].shape[1])
            <= start_index[:offset_index][:, None]
        )

        before_is_via_depot_all = double_solution[:offset_index, :, 1] * x1
        before_is_via_depot = before_is_via_depot_all.nonzero()

        visit_depot_num_2 = torch.sum(before_is_via_depot_all, dim=1)

        select_end_with_depot_node_index_2 = visit_depot_num_2 - 1

        temp_tri_2 = np.triu(
            np.ones((len(visit_depot_num_2), len(visit_depot_num_2))), k=1
        )
        visit_depot_num_numpy_2 = visit_depot_num_2.clone().cpu().numpy()

        temp_index_2 = np.dot(visit_depot_num_numpy_2, temp_tri_2)
        temp_index_torch_2 = torch.from_numpy(temp_index_2).long().cuda()

        select_end_with_depot_node_index_2 = (
            select_end_with_depot_node_index_2 + temp_index_torch_2
        )
        before_is_via_depot_index = before_is_via_depot[
            select_end_with_depot_node_index_2
        ]

        before_start_index = before_is_via_depot_index[:, 1]
        x2 = (
            torch.arange(double_solution[:offset_index, :, 1].shape[1])
            < start_index[:offset_index][:, None]
        )
        x3 = (
            torch.arange(double_solution[:offset_index, :, 1].shape[1])
            >= before_start_index[:, None]
        )
        x4 = x2 * x3
        double_solution_demand = problems[:offset_index, :, 2][
            torch.arange(offset_index)[:, None].repeat(1, double_solution.shape[1]),
            double_solution[:offset_index, :, 0],
        ]
        before_demand = double_solution_demand * x4
        self.satisfy_demand = before_demand.sum(1)

        problems[:offset_index, :, 3] = (
            problems[:offset_index, :, 3] - self.satisfy_demand[:, None]
        )

        sub_solution_node = sub_solution[:, :, 0]

        new_sulution_ascending, rank = torch.sort(
            sub_solution_node, dim=-1, descending=False
        )  # ascending
        _, new_sulution_rank = torch.sort(rank, dim=-1, descending=False)  # ascending
        sub_solution[:, :, 0] = new_sulution_rank + 1

        index_2, _ = (
            torch.cat(
                (
                    new_sulution_ascending,
                    new_sulution_ascending,
                    new_sulution_ascending,
                    new_sulution_ascending,
                ),
                dim=1,
            )
            .type(torch.long)
            .sort(dim=-1, descending=False)
        )

        index_1 = torch.arange(batch_size, dtype=torch.long)[:, None].expand(
            batch_size, index_2.shape[1]
        )  # shape: [B, 2current_step]
        temp = torch.arange((embedding_size), dtype=torch.long)[None, :].expand(
            batch_size, embedding_size
        )  # shape: [B, current_step]
        index_3 = temp.repeat([1, length_of_subpath])

        new_data = problems[index_1, index_2, index_3].view(
            batch_size, length_of_subpath, embedding_size
        )
        new_data = torch.cat((problems[:, 0, :].unsqueeze(dim=1), new_data), dim=1)

        return new_data, sub_solution

    def load_raw_data(self, episode=1000000):
        # Because the dataset is too large, split the workload into two passes.

        if self.env_params["mode"] == "train":
            with open(self.data_path, "r", encoding="utf-8") as dataset_file:
                lines = dataset_file.readlines()
            limit = min(int(episode), len(lines))
            midpoint = max(limit // 2, 0)

            segment_1 = self._load_dataset_segment(lines[:midpoint])
            segment_2 = self._load_dataset_segment(lines[midpoint:limit])

            segments = [
                segment for segment in (segment_1, segment_2) if segment is not None
            ]
            if not segments:
                raise RuntimeError("Dataset loading failed: no data was parsed.")

            nodes, capacities, demands, costs, node_flags = zip(*segments)

            self.raw_data_nodes = torch.cat(nodes, dim=0)
            self.raw_data_capacity = torch.cat(capacities, dim=0)
            self.raw_data_demand = torch.cat(demands, dim=0)
            self.raw_data_cost = torch.cat(costs, dim=0)
            self.raw_data_node_flag = torch.cat(node_flags, dim=0)

        if self.env_params["mode"] == "test":
            if not self.test_in_vrplib:
                with open(self.data_path, "r", encoding="utf-8") as dataset_file:
                    lines = dataset_file.readlines()
                limit = min(int(episode), len(lines))
                segment = self._load_dataset_segment(lines[:limit])
                if segment is None:
                    raise RuntimeError(
                        "Dataset loading failed: no data was parsed for testing."
                    )

                (
                    self.raw_data_nodes,
                    self.raw_data_capacity,
                    self.raw_data_demand,
                    self.raw_data_cost,
                    self.raw_data_node_flag,
                ) = segment
            else:
                (
                    self.cvrp_node_coords_all,
                    self.cvrp_demands_all,
                    self.cvrp_capacitys_all,
                    self.vrplib_cost_all,
                    self.vrplib_name_all,
                ) = self.make_vrplib_data(self.vrplib_path, episode)

        print("load raw dataset done!")

    def make_vrplib_data(self, filename, episode):

        node_coords = []
        demands = []
        capacitys = []
        costs = []
        names = []

        with open(filename, "r", encoding="utf-8") as vrplib_file:
            vrplib_lines = vrplib_file.readlines()

        for line in tqdm(vrplib_lines, ascii=True):
            line = line.split(", ")

            name_index = int(line.index("['name'"))
            depot_index = int(line.index("'depot'"))
            customer_index = int(line.index("'customer'"))
            capacity_index = int(line.index("'capacity'"))
            demand_index = int(line.index("'demand'"))
            cost_index = int(line.index("'cost'"))

            depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
            customer = [
                [float(line[idx]), float(line[idx + 1])]
                for idx in range(customer_index + 1, demand_index, 2)
            ]

            loc = depot + customer

            capacity = int(float(line[capacity_index + 1]))
            demand = [int(line[idx]) for idx in range(demand_index + 1, capacity_index)]
            cost = float(line[cost_index + 1])

            node_coords.append(loc)
            demands.append(demand)
            capacitys.append(capacity)
            costs.append(cost)
            names.append(line[name_index + 1][1:-1])

        # Each row of data represents an instance, and the size of each instance is different
        node_coords = np.array(node_coords, dtype=object)
        demands = np.array(demands, dtype=object)
        capacitys = np.array(capacitys, dtype=object)
        costs = np.array(costs, dtype=object)
        names = np.array(names, dtype=object)

        return node_coords, demands, capacitys, costs, names

    def reset(self, mode, sample_size=1):
        self.selected_count = 0

        if mode == "train":
            self.selected_node_list = torch.zeros(
                (self.batch_size, 0), dtype=torch.long
            )
            self.selected_teacher_flag = torch.zeros(
                (self.batch_size, 0), dtype=torch.long
            )
            self.selected_student_list = torch.zeros(
                (self.batch_size, 0), dtype=torch.long
            )
            self.selected_student_flag = torch.zeros(
                (self.batch_size, 0), dtype=torch.long
            )
            self.current_node = self.problems[:, 0, :]
            self.first_node = self.current_node
            self.step_state = Step_State(
                problems=self.problems,
                first_node=self.first_node[:, None, :],
                current_node=self.current_node[:, None, :],
            )
        if mode == "test":

            self.selected_node_list = torch.zeros(
                (self.batch_size, 0), dtype=torch.long
            )
            self.selected_teacher_flag = torch.zeros(
                (self.batch_size, 0), dtype=torch.long
            )
            self.selected_student_list = torch.zeros(
                (self.batch_size, 0), dtype=torch.long
            )
            self.selected_student_flag = torch.zeros(
                (self.batch_size, 0), dtype=torch.long
            )
            self.current_node = self.problems[:, 0, :]
            self.first_node = self.current_node
            self.step_state = Step_State(
                problems=self.problems,
                first_node=self.first_node[:, None, :],
                current_node=self.current_node[:, None, :],
            )

        reward = None
        done = False
        return Reset_State(self.problems), reward, done

    def pre_step(self):
        reward = None
        reward_student = None
        done = False
        return self.step_state, reward, reward_student, done

    def step(
        self,
        selected,
        selected_student,
        selected_flag_teacher,
        selected_flag_student,
        use_bs=False,
        beam_select_list=None,
        beam_select_flag=None,
        beam_width=None,
        select_new_beam_index=None,
    ):

        self.selected_count += 1

        gather_index = selected[:, None, None].expand(
            (len(selected), 1, 4)
        )  # shape [B,1,4]

        # Update capacity
        # 1. If flag = 1, the vehicle returns to depot and capacity is refilled
        is_depot = selected_flag_teacher == 1
        self.problems[is_depot, :, 3] = self.raw_data_capacity.ravel()[0].item()

        # 2. If capacity is less than demand, capacity is also refilled and the flag of the current access node is changed to 1

        self.current_node_temp = self.problems.gather(
            index=gather_index, dim=1
        ).squeeze(1)
        demands = self.current_node_temp[:, 2]
        smaller_ = self.problems[:, 0, 3] < demands

        selected_flag_teacher[smaller_] = 1
        self.problems[smaller_, :, 3] = self.raw_data_capacity.ravel()[0].item()
        # 3. Subtract the demand of the currently visited node regardless of whether the vehicle is returned to depot to refill

        self.problems[:, :, 3] = self.problems[:, :, 3] - demands[:, None]

        self.current_node = self.problems.gather(index=gather_index, dim=1).squeeze(1)
        # shape [B,4]

        self.selected_node_list = torch.cat(
            (self.selected_node_list, selected[:, None]), dim=1
        )

        self.selected_teacher_flag = torch.cat(
            (self.selected_teacher_flag, selected_flag_teacher[:, None]), dim=1
        )

        self.selected_student_list = torch.cat(
            (self.selected_student_list, selected_student[:, None]), dim=1
        )

        self.selected_student_flag = torch.cat(
            (self.selected_student_flag, selected_flag_student[:, None]), dim=1
        )

        self.step_state.current_node = self.current_node[:, None, :]

        self.step_state.first_node[:, 0, 2] = 0
        self.step_state.current_node[:, 0, 2] = 0
        self.first_node[:, 2] = 0
        self.current_node[:, 2] = 0
        self.step_state.first_node[:, 0, 3] = self.problems[:, 1, 3].clone()
        self.step_state.current_node[:, 0, 3] = self.problems[:, 1, 3].clone()
        self.first_node[:, 3] = self.problems[:, 1, 3].clone()
        self.current_node[:, 3] = self.problems[:, 1, 3].clone()
        # returning values
        done = self.selected_count == self.problems.shape[1] - 1
        if done:
            reward, reward_student = self._get_travel_distance()  # note the minus sign!
        else:
            reward, reward_student = None, None

        return self.step_state, reward, reward_student, done

    def cal_length(self, problems, order_node, order_flag):
        # problems:   [B,V+1,2]
        # order_node: [B,V]
        # order_flag: [B,V]
        order_node_ = order_node.clone()

        order_flag_ = order_flag.clone()

        index_small = torch.le(order_flag_, 0.5)
        index_bigger = torch.gt(order_flag_, 0.5)

        order_flag_[index_small] = order_node_[index_small]
        order_flag_[index_bigger] = 0

        roll_node = order_node_.roll(dims=1, shifts=1)

        problem_size = problems.shape[1] - 1

        order_gathering_index = order_node_.unsqueeze(2).expand(-1, problem_size, 2)
        order_loc = problems.gather(dim=1, index=order_gathering_index)

        roll_gathering_index = roll_node.unsqueeze(2).expand(-1, problem_size, 2)
        roll_loc = problems.gather(dim=1, index=roll_gathering_index)

        flag_gathering_index = order_flag_.unsqueeze(2).expand(-1, problem_size, 2)
        flag_loc = problems.gather(dim=1, index=flag_gathering_index)

        order_lengths = (order_loc - flag_loc) ** 2

        order_flag_[:, 0] = 0
        flag_gathering_index = order_flag_.unsqueeze(2).expand(-1, problem_size, 2)
        flag_loc = problems.gather(dim=1, index=flag_gathering_index)

        roll_lengths = (roll_loc - flag_loc) ** 2

        length = (order_lengths.sum(2).sqrt() + roll_lengths.sum(2).sqrt()).sum(1)

        return length

    def _get_travel_distance(self):

        # optimal distance
        if self.test_in_vrplib:
            travel_distances = self.vrplib_cost
            self.problems[:, :, :2] = (
                self.problems[:, :, :2]
                * (self.problem_max_min[0] - self.problem_max_min[1])
                + self.problem_max_min[1]
            )
        else:
            # teacher length
            problems = self.problems[:, :, [0, 1]]
            order_node = self.solution[:, :, 0]
            order_flag = self.solution[:, :, 1]
            travel_distances = self.cal_length(problems, order_node, order_flag)

        # trained model's distance
        problems = self.problems[:, :, [0, 1]]
        order_node = self.selected_student_list.clone()
        order_flag = self.selected_student_flag.clone()

        travel_distances_student = self.cal_length(problems, order_node, order_flag)

        return -travel_distances, -travel_distances_student

    def _get_travel_distance_2(self, problems_, solution_, need_optimal=False):

        if self.test_in_vrplib:
            if need_optimal:
                return self.vrplib_cost, self.vrplib_name
            else:
                problems = (
                    problems_[:, :, [0, 1]].clone()
                    * (self.problem_max_min[0] - self.problem_max_min[1])
                    + self.problem_max_min[1]
                )
                order_node = solution_[:, :, 0].clone()
                order_flag = solution_[:, :, 1].clone()
                travel_distances = self.cal_length(problems, order_node, order_flag)
        else:
            problems = problems_[:, :, [0, 1]].clone()
            order_node = solution_[:, :, 0].clone()
            order_flag = solution_[:, :, 1].clone()
            travel_distances = self.cal_length(problems, order_node, order_flag)

        return travel_distances

    def destroy_solution(self, problem, complete_solution):

        (
            self.problems,
            self.solution,
            first_node_index,
            length_of_subpath,
            double_solution,
        ) = self.sampling_subpaths_repair(
            problem, complete_solution, mode=self.env_params["mode"]
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

    def sampling_subpaths_repair(
        self, problems, solution, length_fix=False, mode="test", repair=True
    ):
        # problems shape (B,V+1,4)
        # solution shape (B,V,2) uses 1-based node indices

        problems_size = problems.shape[1] - 1
        # print('problems_size',problems_size)
        batch_size = problems.shape[0]
        embedding_size = problems.shape[2]

        # the first node of subpath: uniform sampling, from 0 to N

        length_of_subpath = torch.randint(
            low=4, high=min(problems_size, self.env_params["RRC_range"]) + 1, size=[1]
        )[
            0
        ]  # in [4,N]

        start_from_depot = solution[:, :, 1].nonzero()

        end_with_depot = start_from_depot
        end_with_depot[:, 1] = end_with_depot[:, 1] - 1
        end_with_depot[end_with_depot.le(-0.5)] = solution.shape[1] - 1

        visit_depot_num = torch.sum(solution[:, :, 1], dim=1)

        p = torch.rand(len(visit_depot_num))
        select_end_with_depot_node_index = p * visit_depot_num
        select_end_with_depot_node_index = torch.floor(
            select_end_with_depot_node_index
        ).long()

        temp_tri = np.triu(np.ones((len(visit_depot_num), len(visit_depot_num))), k=1)
        visit_depot_num_numpy = visit_depot_num.clone().cpu().numpy()

        temp_index = np.dot(visit_depot_num_numpy, temp_tri)
        temp_index_torch = torch.from_numpy(temp_index).long().cuda()

        select_end_with_depot_node_index_ = (
            select_end_with_depot_node_index + temp_index_torch
        )

        select_end_with_depot_node = end_with_depot[
            select_end_with_depot_node_index_, 1
        ]
        double_solution = torch.cat((solution, solution), dim=1)

        select_end_with_depot_node = select_end_with_depot_node + problems_size

        indexx = torch.arange(length_of_subpath).repeat(batch_size, 1)
        offset = select_end_with_depot_node - length_of_subpath + 1

        indexxxx = indexx + offset[:, None]

        sub_solu_index1 = torch.arange(batch_size)[:, None].repeat(
            1, 2 * length_of_subpath
        )
        sub_solu_index2 = indexxxx.repeat_interleave(2, dim=1)
        sub_solu_index3 = torch.arange(double_solution.shape[2])[None, :].repeat(
            batch_size, length_of_subpath
        )
        sub_solution = double_solution[
            sub_solu_index1, sub_solu_index2, sub_solu_index3
        ].reshape(batch_size, length_of_subpath, 2)

        offset_index = problems.shape[0]
        start_index = indexxxx[:, 0]

        x1 = (
            torch.arange(double_solution[:offset_index, :, 1].shape[1])
            <= start_index[:offset_index][:, None]
        )

        before_is_via_depot_all = double_solution[:offset_index, :, 1] * x1
        before_is_via_depot = before_is_via_depot_all.nonzero()

        visit_depot_num_2 = torch.sum(before_is_via_depot_all, dim=1)

        select_end_with_depot_node_index_2 = visit_depot_num_2 - 1

        temp_tri_2 = np.triu(
            np.ones((len(visit_depot_num_2), len(visit_depot_num_2))), k=1
        )
        visit_depot_num_numpy_2 = visit_depot_num_2.clone().cpu().numpy()

        temp_index_2 = np.dot(visit_depot_num_numpy_2, temp_tri_2)
        temp_index_torch_2 = torch.from_numpy(temp_index_2).long().cuda()

        select_end_with_depot_node_index_2 = (
            select_end_with_depot_node_index_2 + temp_index_torch_2
        )
        before_is_via_depot_index = before_is_via_depot[
            select_end_with_depot_node_index_2
        ]

        before_start_index = before_is_via_depot_index[:, 1]
        x2 = (
            torch.arange(double_solution[:offset_index, :, 1].shape[1])
            < start_index[:offset_index][:, None]
        )
        x3 = (
            torch.arange(double_solution[:offset_index, :, 1].shape[1])
            >= before_start_index[:, None]
        )
        x4 = x2 * x3
        double_solution_demand = problems[:offset_index, :, 2][
            torch.arange(offset_index)[:, None].repeat(1, double_solution.shape[1]),
            double_solution[:offset_index, :, 0],
        ]

        before_demand = double_solution_demand * x4

        self.satisfy_demand = before_demand.sum(1)

        problems[:offset_index, :, 3] = (
            problems[:offset_index, :, 3] - self.satisfy_demand[:, None]
        )

        sub_solution_node = sub_solution[:, :, 0]
        new_sulution_ascending, rank = torch.sort(
            sub_solution_node, dim=-1, descending=False
        )  # ascending
        _, new_sulution_rank = torch.sort(rank, dim=-1, descending=False)  # ascending
        sub_solution[:, :, 0] = new_sulution_rank + 1

        index_2, _ = (
            torch.cat(
                (
                    new_sulution_ascending,
                    new_sulution_ascending,
                    new_sulution_ascending,
                    new_sulution_ascending,
                ),
                dim=1,
            )
            .type(torch.long)
            .sort(dim=-1, descending=False)
        )

        index_1 = torch.arange(batch_size, dtype=torch.long)[:, None].expand(
            batch_size, index_2.shape[1]
        )  # shape: [B, 2current_step]
        temp = torch.arange((embedding_size), dtype=torch.long)[None, :].expand(
            batch_size, embedding_size
        )  # shape: [B, current_step]
        index_3 = temp.repeat([1, length_of_subpath])

        new_data = problems[index_1, index_2, index_3].view(
            batch_size, length_of_subpath, embedding_size
        )
        new_data = torch.cat((problems[:, 0, :].unsqueeze(dim=1), new_data), dim=1)
        if repair:
            return (
                new_data,
                sub_solution,
                start_index,
                length_of_subpath,
                double_solution,
            )
        return new_data, sub_solution
