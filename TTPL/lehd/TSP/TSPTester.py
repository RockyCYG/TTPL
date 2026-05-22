from dataclasses import dataclass
from logging import getLogger
import numpy as np
import torch
import random
import pickle
import os

from lehd.TSP.TSPModel import TSPModel as Model
from lehd.TSP.TSPEnv import TSPEnv as Env
from lehd.utils.utils import AverageMeter, TimeEstimator, get_result_folder


@dataclass
class _PomoResetState:
    problems: torch.Tensor


@dataclass
class _PomoStepState:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    current_node: torch.Tensor = None
    ninf_mask: torch.Tensor = None


class _PomoTsplibEnv:
    def __init__(self, problems, original_problems, edge_weight_type):
        self.problems = problems
        self.original_problems = original_problems
        self.edge_weight_type = edge_weight_type
        self.batch_size = problems.size(0)
        self.problem_size = problems.size(1)
        self.pomo_size = self.problem_size
        self.BATCH_IDX = torch.arange(
            self.batch_size, device=problems.device
        )[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(
            self.pomo_size, device=problems.device
        )[None, :].expand(self.batch_size, self.pomo_size)
        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None
        self.step_state = None

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros(
            (self.batch_size, self.pomo_size, 0),
            dtype=torch.long,
            device=self.problems.device,
        )
        self.step_state = _PomoStepState(
            BATCH_IDX=self.BATCH_IDX,
            POMO_IDX=self.POMO_IDX,
            ninf_mask=torch.zeros(
                (self.batch_size, self.pomo_size, self.problem_size),
                device=self.problems.device,
            ),
        )
        return _PomoResetState(self.problems), None, False

    def pre_step(self):
        return self.step_state, None, False

    def step(self, selected):
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat(
            (self.selected_node_list, self.current_node[:, :, None]), dim=2
        )

        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = (
            float("-inf")
        )

        done = self.selected_count == self.problem_size
        reward = -self._get_travel_distance() if done else None
        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list.unsqueeze(3).expand(
            self.batch_size, self.pomo_size, self.problem_size, 2
        )

        original = self.original_problems
        if original.size(0) == 1 and self.batch_size != 1:
            original = original.expand(self.batch_size, -1, -1)

        seq_expanded = original[:, None, :, :].expand(
            self.batch_size, self.pomo_size, self.problem_size, 2
        )
        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        edge_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()

        if self.edge_weight_type == "EUC_2D":
            edge_lengths = torch.floor(edge_lengths + 0.5)
        elif self.edge_weight_type == "CEIL_2D":
            edge_lengths = torch.ceil(edge_lengths)

        return edge_lengths.sum(2)


def _normalize_to_unit_square(node_xy):
    xy_max = torch.max(node_xy, dim=1, keepdim=True).values
    xy_min = torch.min(node_xy, dim=1, keepdim=True).values
    ratio = torch.max((xy_max - xy_min), dim=-1, keepdim=True).values
    ratio[ratio == 0] = 1
    return (node_xy - xy_min) / ratio.expand(-1, 1, 2)


def _augment_xy_data_by_8_fold(problems):
    x = problems[:, :, [0]]
    y = problems[:, :, [1]]

    return torch.cat(
        (
            torch.cat((x, y), dim=2),
            torch.cat((1 - x, y), dim=2),
            torch.cat((x, 1 - y), dim=2),
            torch.cat((1 - x, 1 - y), dim=2),
            torch.cat((y, x), dim=2),
            torch.cat((1 - y, x), dim=2),
            torch.cat((y, 1 - x), dim=2),
            torch.cat((1 - y, 1 - x), dim=2),
        ),
        dim=0,
    )


class TSPTester:
    """
    Tester for the Traveling Salesperson Problem model.
    """

    def __init__(self, env_params, model_params, tester_params):
        # Save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # Set random seed for reproducibility
        seed = 123
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Initialize logger and result folder
        self.logger = getLogger(name="trainer")
        self.result_folder = get_result_folder()

        # Configure CUDA device
        USE_CUDA = self.tester_params["use_cuda"]
        if USE_CUDA:
            cuda_device_num = self.tester_params["cuda_device_num"]
            self.device = torch.device("cuda", cuda_device_num)
            torch.cuda.set_device(cuda_device_num)
        else:
            self.device = torch.device("cpu")
        torch.set_default_device(self.device)
        torch.set_default_dtype(torch.float32)

        # Load pre-trained model
        model_load = tester_params["model_load"]
        checkpoint_fullname = "{path}/checkpoint-{epoch}.pt".format(**model_load)
        # torch.serialization.add_safe_globals([set])
        checkpoint = torch.load(
            checkpoint_fullname, map_location=self.device
        )

        checkpoint_model_params = checkpoint.get("model_params")
        if checkpoint_model_params:
            merged_model_params = dict(checkpoint_model_params)
            merged_model_params.update(self.model_params)
            self.model_params = merged_model_params

        # Initialize environment and model
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        requested_backend = tester_params.get("inference_backend", "auto")
        if requested_backend == "auto":
            self.inference_backend = (
                "pomo" if self.model_params.get("use_polynet", False) else "ttpl"
            )
        else:
            self.inference_backend = requested_backend
        if self.inference_backend == "pomo" and not self.model_params.get(
            "use_polynet", False
        ):
            raise ValueError("POMO inference requires a POMO/PolyNet checkpoint.")
        torch.set_printoptions(precision=20)

        # Initialize time estimators
        self.time_estimator = TimeEstimator()
        self.time_estimator_2 = TimeEstimator()

    def run(self, **kwargs):
        """
        Run the testing process.
        """
        if self.inference_backend == "pomo":
            return self._run_pomo(**kwargs)

        self.time_estimator.reset()
        self.time_estimator_2.reset()

        if not self.env_params["test_in_tsplib"]:
            self.env.load_raw_data(self.tester_params["test_episodes"])

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()

        test_num_episode = self.tester_params["test_episodes"]
        episode = 0

        # Store gaps for different problem sizes
        problem_gaps = {
            "all": [],
            "<100": [],
            "100-200": [],
            "200-500": [],
            "500-1000": [],
            ">=1000": [],
        }

        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params["test_batch_size"], remaining)

            score_teacher, score_student, problems_size = self._test_one_batch(
                episode,
                batch_size,
                clock=self.time_estimator_2,
                **kwargs,
            )
            current_gap = (score_student - score_teacher) / score_teacher

            # Categorize gap based on problem size
            if problems_size < 100:
                problem_gaps["<100"].append(current_gap)
            elif 100 <= problems_size < 200:
                problem_gaps["100-200"].append(current_gap)
            elif 200 <= problems_size < 500:
                problem_gaps["200-500"].append(current_gap)
            elif 500 <= problems_size < 1000:
                problem_gaps["500-1000"].append(current_gap)
            else:
                problem_gaps[">=1000"].append(current_gap)
            problem_gaps["all"].append(current_gap)

            # Print mean gaps
            for key, gaps in problem_gaps.items():
                if key != "all" and gaps:
                    print(
                        f"problems_{key} mean gap: {np.mean(gaps):.4f}, count: {len(gaps)}"
                    )

            score_AM.update(score_teacher, batch_size)
            score_student_AM.update(score_student, batch_size)

            episode += batch_size

            # Log progress
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(
                episode, test_num_episode
            )
            self.logger.info(
                f"episode {episode:3d}/{test_num_episode:3d}, Elapsed[{elapsed_time_str}], "
                f"Remain[{remain_time_str}], Score_teacher:{score_teacher:.4f}, Score_student: {score_student:.4f}"
            )

            if episode == test_num_episode:
                self.logger.info(" *** Test Done *** ")
                if not self.env_params["test_in_tsplib"]:
                    gap_ = (score_student_AM.avg - score_AM.avg) / score_AM.avg * 100
                    self.logger.info(f" Teacher SCORE: {score_AM.avg:.4f} ")
                    self.logger.info(f" Student SCORE: {score_student_AM.avg:.4f} ")
                    self.logger.info(f" Gap: {gap_:.4f}%")
                else:
                    average_gap = np.mean(problem_gaps["all"])
                    self.logger.info(f" Average Gap: {average_gap * 100:.4f}%")
                    gap_ = average_gap
                    print(problem_gaps["all"])

        return score_AM.avg, score_student_AM.avg, gap_

    def _run_pomo(self, **kwargs):
        if not self.env_params["test_in_tsplib"]:
            raise ValueError("POMO inference is only wired for TSPLIB .tsp tests.")

        self.time_estimator.reset()
        self.time_estimator_2.reset()
        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        gaps = []

        test_num_episode = self.tester_params["test_episodes"]
        for episode in range(test_num_episode):
            (
                score_teacher,
                score_no_aug,
                score_aug,
                problem_size,
                name,
            ) = self._test_one_tsplib_pomo(episode, **kwargs)
            score_student = score_aug
            current_gap = (score_student - score_teacher) / score_teacher
            gaps.append(current_gap)
            score_AM.update(score_teacher, 1)
            score_student_AM.update(score_student, 1)

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(
                episode + 1, test_num_episode
            )
            self.logger.info(
                f"episode {episode + 1:3d}/{test_num_episode:3d}, "
                f"Elapsed[{elapsed_time_str}], Remain[{remain_time_str}], "
                f"name:{name}, size:{problem_size}, opt:{score_teacher:.4f}, "
                f"pomo_no_aug:{score_no_aug:.4f}, pomo_aug:{score_aug:.4f}, "
                f"gap:{current_gap * 100:.4f}%"
            )

        gap_ = float(np.mean(gaps)) if gaps else 0
        self.logger.info(" *** POMO Test Done *** ")
        self.logger.info(f" Teacher SCORE: {score_AM.avg:.4f} ")
        self.logger.info(f" Student SCORE: {score_student_AM.avg:.4f} ")
        self.logger.info(f" Average Gap: {gap_ * 100:.4f}%")
        print(gaps)
        return score_AM.avg, score_student_AM.avg, gap_

    def _test_one_tsplib_pomo(self, episode, **kwargs):
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(episode, 1)

            original_problems = torch.from_numpy(
                self.env.tsplib_problems.reshape(1, -1, 2)
            ).to(device=self.device, dtype=torch.float32)
            node_xy = _normalize_to_unit_square(original_problems)

            aug_factor = self.tester_params.get("pomo_aug_factor", 8)
            if aug_factor == 8:
                problems = _augment_xy_data_by_8_fold(node_xy)
            elif aug_factor == 1:
                problems = node_xy
            else:
                raise ValueError("Only POMO aug factors 1 and 8 are supported.")

            pomo_env = _PomoTsplibEnv(
                problems=problems.to(self.device),
                original_problems=original_problems,
                edge_weight_type=self.env.tsplib_edge_weight_type,
            )

            reset_state, _, _ = pomo_env.reset()
            z = self._make_pomo_z(pomo_env.batch_size, pomo_env.pomo_size)
            self.model.pre_forward(reset_state, z)

            state, reward, done = pomo_env.pre_step()
            while not done:
                logit_bias = self._make_pomo_logit_bias(
                    pomo_env=pomo_env,
                    state=state,
                    logit_bias_func=kwargs.get("logit_bias"),
                )
                selected, _ = self.model(state, logit_bias=logit_bias)
                state, reward, done = pomo_env.step(selected)

            tour_lengths = -reward
            best_len_per_aug = tour_lengths.min(dim=1).values
            no_aug_score = best_len_per_aug[0].item()
            aug_score = best_len_per_aug.min(dim=0).values.item()

            optimal_length = self.env.tsplib_cost.mean().item()
            name = str(self.env.tsplib_name[0])
            return (
                optimal_length,
                float(no_aug_score),
                float(aug_score),
                pomo_env.problem_size,
                name,
            )

    def _make_pomo_z(self, batch_size, rollout_size):
        if not self.model_params.get("use_polynet", False):
            return None

        z_dim = self.model_params["z_dim"]
        rollout_idx = torch.arange(
            rollout_size, device=self.device, dtype=torch.long
        )
        bit_idx = torch.arange(z_dim, device=self.device, dtype=torch.long)
        z = ((rollout_idx[:, None] >> bit_idx[None, :]) & 1).float()
        return z[None, :, :].expand(batch_size, rollout_size, z_dim)

    def _make_pomo_logit_bias(self, pomo_env, state, logit_bias_func=None):
        if state.current_node is None:
            return None

        logit_bias = None
        if self.tester_params.get("pomo_log_dist_bias", False):
            logit_bias = self._make_fixed_log_dist_bias(
                coords=pomo_env.problems,
                current_node=state.current_node,
                ninf_mask=state.ninf_mask,
            )

        if logit_bias_func is not None:
            custom_bias = logit_bias_func(
                pomo_env.problems,
                state.current_node,
                state.ninf_mask,
                pomo_env.selected_count,
            )
            if custom_bias is not None:
                custom_bias = custom_bias.to(
                    device=pomo_env.problems.device,
                    dtype=pomo_env.problems.dtype,
                )
                if custom_bias.shape != state.ninf_mask.shape:
                    raise ValueError(
                        "custom logit bias must have shape "
                        f"{tuple(state.ninf_mask.shape)}, got {tuple(custom_bias.shape)}"
                    )
                custom_bias = torch.where(
                    torch.isfinite(custom_bias),
                    custom_bias,
                    torch.zeros_like(custom_bias),
                )
                custom_bias = custom_bias.masked_fill(state.ninf_mask < 0, 0.0)
                logit_bias = custom_bias if logit_bias is None else logit_bias + custom_bias

        return logit_bias

    def _make_fixed_log_dist_bias(self, coords, current_node, ninf_mask):
        batch_size, problem_size, _ = coords.shape
        pomo_size = current_node.size(1)
        gather_index = current_node[:, :, None].expand(batch_size, pomo_size, 2)
        current_xy = coords.gather(dim=1, index=gather_index)
        dist = torch.cdist(current_xy, coords).clamp_min(
            self.tester_params.get("pomo_log_dist_eps", 1e-6)
        )

        valid_mask = ninf_mask == 0
        topk = min(self.tester_params.get("pomo_log_dist_topk", 20), problem_size)
        topk = max(topk, 0)

        topk_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        if topk > 0:
            ranked_dist = dist.masked_fill(~valid_mask, float("inf"))
            topk_index = torch.topk(ranked_dist, k=topk, dim=2, largest=False).indices
            topk_mask.scatter_(dim=2, index=topk_index, value=True)

        near_bias = -torch.log(dist)
        far_bias = -dist
        bias = torch.where(topk_mask, near_bias, far_bias)
        bias = bias.masked_fill(~valid_mask, 0.0)
        bias = torch.where(torch.isfinite(bias), bias, torch.zeros_like(bias))

        alpha = self.tester_params.get("pomo_log_dist_alpha", 1.0)
        return alpha * bias

    def decide_whether_to_repair_solution(
        self,
        after_repair_sub_solution,
        before_reward,
        after_reward,
        first_node_index,
        length_of_subpath,
        double_solution,
    ):
        """
        Decide whether to accept the repaired solution based on the reward.
        """
        the_whole_problem_size = int(double_solution.shape[1] / 2)
        other_part_1 = double_solution[:, :first_node_index]
        other_part_2 = double_solution[:, first_node_index + length_of_subpath :]
        origin_sub_solution = double_solution[
            :, first_node_index : first_node_index + length_of_subpath
        ]

        jjj, _ = torch.sort(origin_sub_solution, dim=1, descending=False)
        index = torch.arange(jjj.shape[0])[:, None].repeat(1, jjj.shape[1])
        kkk_2 = jjj[index, after_repair_sub_solution]

        if_repair = before_reward > after_reward
        double_solution[if_repair] = torch.cat(
            (other_part_1[if_repair], kkk_2[if_repair], other_part_2[if_repair]), dim=1
        )
        after_repair_complete_solution = double_solution[
            :, first_node_index : first_node_index + the_whole_problem_size
        ]

        return after_repair_complete_solution

    def _test_one_batch(self, episode, batch_size, clock=None, **kwargs):
        """
        Test one batch of TSP instances.
        """
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(episode, batch_size)
            self.origin_problem = self.env.problems
            self.env.reset(self.env_params["mode"])

            if self.env.test_in_tsplib:
                self.optimal_length, name = self.env._get_travel_distance_2(
                    self.origin_problem, self.env.solution, need_optimal=True
                )
            else:
                self.optimal_length = self.env._get_travel_distance_2(
                    self.origin_problem, self.env.solution
                )
                name = f"TSP_visual_1_{self.origin_problem.shape[1]}"

            state, _, _, done = self.env.pre_step()
            current_step = 0

            if self.env_params["random_insertion"]:
                initial_method = "nn"
                best_select_node_list = read_kpl_file(
                    initial_method, self.env.data_path, episode, batch_size
                )
            else:
                while not done:
                    if current_step == 0:
                        selected_student = torch.zeros(batch_size, dtype=torch.int64)
                    else:
                        _, _, _, selected_student = self.model(
                            state,
                            self.env.selected_node_list,
                            self.env.solution,
                            current_step,
                            **kwargs,
                        )
                    state, _, _, done = self.env.step(
                        selected_student, selected_student
                    )
                    current_step += 1
                print("Get first complete solution!")
                best_select_node_list = self.env.selected_node_list

            current_best_length = self.env._get_travel_distance_2(
                self.origin_problem, best_select_node_list
            )
            escape_time, _ = clock.get_est_string(1, 1)
            gap = (
                (current_best_length.mean() - self.optimal_length.mean())
                / self.optimal_length.mean()
            ).item() * 100
            self.logger.info(
                f"greedy, name:{name}, gap:{gap:5f} %, Elapsed[{escape_time}], "
                f"stu_l:{current_best_length.mean().item():5f}, opt_l:{self.optimal_length.mean().item():5f}"
            )

            # Ruin and Recreate (RRC)
            budget = self.env_params["RRC_budget"]
            for bbbb in range(budget):
                self.env.load_problems(episode, batch_size)

                # Randomly inverse the solution
                if torch.randint(low=0, high=100, size=[1]).item() >= 50:
                    best_select_node_list = torch.flip(best_select_node_list, dims=[1])

                # Destroy part of the solution
                (
                    partial_solution_length,
                    first_node_index,
                    length_of_subpath,
                    double_solution,
                ) = self.env.destroy_solution(self.env.problems, best_select_node_list)
                before_reward = partial_solution_length

                # Reset environment for repair
                self.env.reset(self.env_params["mode"])
                state, _, _, done = self.env.pre_step()
                current_step = 0

                # Recreate the solution
                while not done:
                    if current_step == 0:
                        selected_student = self.env.solution[:, -1]
                    elif current_step == 1:
                        selected_student = self.env.solution[:, 0]
                    else:
                        _, _, _, selected_student = self.model(
                            state,
                            self.env.selected_node_list,
                            self.env.solution,
                            current_step,
                            **kwargs,
                        )
                    state, _, reward_student, done = self.env.step(
                        selected_student, selected_student
                    )
                    current_step += 1

                after_repair_sub_solution = torch.roll(
                    self.env.selected_node_list, shifts=-1, dims=1
                )
                after_reward = reward_student

                # Decide whether to accept the new solution
                best_select_node_list = self.decide_whether_to_repair_solution(
                    after_repair_sub_solution,
                    before_reward,
                    after_reward,
                    first_node_index,
                    length_of_subpath,
                    double_solution,
                )
                current_best_length = self.env._get_travel_distance_2(
                    self.origin_problem, best_select_node_list
                )

                escape_time, _ = clock.get_est_string(1, 1)
                gap = (
                    (current_best_length.mean() - self.optimal_length.mean())
                    / self.optimal_length.mean()
                ).item() * 100
                self.logger.info(
                    f"RRC step{bbbb}, name:{name}, gap:{gap:6f} %, Elapsed[{escape_time}], "
                    f"stu_l:{current_best_length.mean().item():6f}, opt_l:{self.optimal_length.mean().item():6f}"
                )

            current_best_length = self.env._get_travel_distance_2(
                self.origin_problem, best_select_node_list
            )
            gap = (
                (current_best_length.mean() - self.optimal_length.mean())
                / self.optimal_length.mean()
                * 100
            )
            print(f"{name}, current_best_length_gap: {gap:.4f} %")

            return (
                self.optimal_length.mean().item(),
                current_best_length.mean().item(),
                self.env.problem_size,
            )


def read_kpl_file(method, file_name, episode, batch):
    """
    Reads a .pkl file containing solutions.
    """
    folder_path = os.path.dirname(file_name)
    basename = os.path.basename(file_name)
    part_needed = basename.split("-")[0]
    file_name_pkl = f"{part_needed}-{method}.pkl"
    path = os.path.join(folder_path, "pkl", file_name_pkl)
    solution = load_pkl_solution_data(path)
    return solution[episode : episode + batch]


def load_pkl_solution_data(solution_filename):
    """
    Loads solution data from a .pkl file.
    """
    with open(solution_filename, "rb") as f:
        solutions, _ = pickle.load(f)

    dataset_size = len(solutions)
    solution_temp = [solutions[i][1] for i in range(dataset_size)]
    solutions = np.array(solution_temp)

    return torch.tensor(solutions, dtype=torch.long)
