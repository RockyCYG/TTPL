from logging import getLogger
import numpy as np
import torch
import random
import os

from lehd.CVRP.VRPModel import VRPModel as Model
from lehd.CVRP.VRPEnv import VRPEnv as Env
from lehd.utils.utils import AverageMeter, TimeEstimator, get_result_folder


class VRPTester:
    """
    Tester for the Vehicle Routing Problem model.
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

        # Initialize environment and model
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Load pre-trained model
        model_load = tester_params["model_load"]
        checkpoint_fullname = "{path}/checkpoint-{epoch}.pt".format(**model_load)
        # torch.serialization.add_safe_globals([set])
        checkpoint = torch.load(
            checkpoint_fullname, map_location=self.device
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        torch.set_printoptions(precision=20)

        # Initialize time estimators
        self.time_estimator = TimeEstimator()
        self.time_estimator_2 = TimeEstimator()

    def run(self, **kwargs):
        """
        Run the testing process.
        """
        self.time_estimator.reset()
        self.time_estimator_2.reset()

        self.env.load_raw_data(self.tester_params["test_episodes"])

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()

        test_num_episode = self.tester_params["test_episodes"]
        episode = self.tester_params.get("begin_index", 0)

        size_buckets = {
            "<100": [],
            "100-200": [],
            "200-500": [],
            "500-1000": [],
            ">=1000": [],
        }

        name_buckets = {
            "An": [],
            "Br": [],
            "Gh": [],
            "Fl": [],
            "Le": [],
            "P-": [],
            "X-": [],
            "Li": [],
            "XX": [],
        }

        all_gaps = []

        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params["test_batch_size"], remaining)

            score_teacher, score_student, problems_size, vrpname = self._test_one_batch(
                episode,
                batch_size,
                clock=self.time_estimator_2,
                **kwargs,
            )

            current_gap = (score_student - score_teacher) / score_teacher

            if problems_size < 100:
                size_buckets["<100"].append(current_gap)
            elif problems_size < 200:
                size_buckets["100-200"].append(current_gap)
            elif problems_size < 500:
                size_buckets["200-500"].append(current_gap)
            elif problems_size < 1000:
                size_buckets["500-1000"].append(current_gap)
            else:
                size_buckets[">=1000"].append(current_gap)

            prefix = vrpname[:2]
            if prefix in name_buckets:
                name_buckets[prefix].append(current_gap)

            score_AM.update(score_teacher, batch_size)
            score_student_AM.update(score_student, batch_size)
            all_gaps.append(current_gap)

            for key, gaps in size_buckets.items():
                if gaps:
                    print(
                        f"problems_{key} mean gap: {np.mean(gaps):.4f}, count: {len(gaps)}"
                    )

            for key, gaps in name_buckets.items():
                if gaps:
                    self.logger.info(
                        f" problems_{key:<3} mean gap:{np.mean(gaps) * 100:6.4f}%, num:{len(gaps)}"
                    )

            episode += batch_size

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(
                episode, test_num_episode
            )
            self.logger.info(
                f"episode {episode:3d}/{test_num_episode:3d}, Elapsed[{elapsed_time_str}], "
                f"Remain[{remain_time_str}], Score_teacher:{score_teacher:.4f}, "
                f"Score_student: {score_student:.4f}"
            )

        self.logger.info(" *** Test Done *** ")

        if self.env_params.get("test_in_vrplib", False):
            average_gap = np.mean(all_gaps) * 100 if all_gaps else 0.0
            self.logger.info(f" Gap: {average_gap:.4f}%")
            gap_ = average_gap
        else:
            self.logger.info(f" Teacher SCORE: {score_AM.avg:.4f} ")
            self.logger.info(f" Student SCORE: {score_student_AM.avg:.4f} ")
            gap_ = (
                (score_student_AM.avg - score_AM.avg) / score_AM.avg * 100
                if score_AM.avg
                else 0.0
            )
            self.logger.info(f" Gap: {gap_:.4f}%")

        return score_AM.avg, score_student_AM.avg, gap_

    def decide_whether_to_repair_solution(
        self,
        after_repair_sub_solution,
        before_reward,
        after_reward,
        first_node_index,
        length_of_subpath,
        double_solution,
    ):

        the_whole_problem_size = int(double_solution.shape[1] / 2)
        batch_size = len(double_solution)

        temp = torch.arange(double_solution.shape[1])

        x3 = temp >= first_node_index[:, None].long()
        x4 = temp < (first_node_index[:, None] + length_of_subpath).long()
        x5 = x3 * x4

        origin_sub_solution = double_solution[x5.unsqueeze(2).repeat(1, 1, 2)].reshape(
            batch_size, length_of_subpath, 2
        )

        jjj, _ = torch.sort(origin_sub_solution[:, :, 0], dim=1, descending=False)

        index = torch.arange(batch_size)[:, None].repeat(1, jjj.shape[1])

        kkk_2 = jjj[index, after_repair_sub_solution[:, :, 0] - 1]

        after_repair_sub_solution[:, :, 0] = kkk_2

        if_repair = after_reward < before_reward

        need_to_repari_double_solution = double_solution[if_repair]
        need_to_repari_double_solution[x5[if_repair].unsqueeze(2).repeat(1, 1, 2)] = (
            after_repair_sub_solution[if_repair].ravel()
        )
        double_solution[if_repair] = need_to_repari_double_solution

        x6 = (
            temp
            >= (
                first_node_index[:, None] + length_of_subpath - the_whole_problem_size
            ).long()
        )

        x7 = temp < (first_node_index[:, None] + length_of_subpath).long()

        x8 = x6 * x7

        after_repair_complete_solution = double_solution[
            x8.unsqueeze(2).repeat(1, 1, 2)
        ].reshape(batch_size, the_whole_problem_size, -1)

        return after_repair_complete_solution

    def _test_one_batch(self, episode, batch_size, clock=None, **kwargs):

        random_seed = 12
        torch.manual_seed(random_seed)

        ###############################################
        self.model.eval()

        with torch.no_grad():

            self.env.load_problems(episode, batch_size)

            _, _, _ = self.env.reset(self.env_params["mode"])

            current_step = 0

            state, reward, reward_student, done = (
                self.env.pre_step()
            )  # state: data, first_node = current_node

            self.origin_problem = self.env.problems.clone().detach()

            if self.env.test_in_vrplib:
                self.optimal_length, name = self.env._get_travel_distance_2(
                    self.origin_problem, self.env.solution, need_optimal=True
                )
            else:
                self.optimal_length = self.env._get_travel_distance_2(
                    self.origin_problem, self.env.solution
                )
                name = "vrp" + str(self.env.solution.shape[1])
            B_V = batch_size * 1

            if self.env_params["random_insertion"]:
                folder_path = os.path.dirname(self.env.data_path)

                file_name = (
                    folder_path
                    + f"/cvrp{self.origin_problem.shape[1] - 1}_C{self.env.raw_data_capacity[0]}_results.pkl-nn.pt"
                )
                best_select_node_list = torch.load(file_name, map_location=self.device)[
                    episode : episode + batch_size
                ]
            else:
                print("greedy")
                while not done:

                    (
                        loss_node,
                        selected_teacher,
                        selected_student,
                        selected_flag_teacher,
                        selected_flag_student,
                    ) = self.model(
                        state,
                        self.env.selected_node_list,
                        self.env.solution,
                        current_step,
                        raw_data_capacity=self.env.raw_data_capacity,
                        **kwargs,
                    )  # update the selected nodes and probabilities

                    if current_step == 0:
                        selected_flag_teacher = torch.ones(B_V, dtype=torch.int)
                        selected_flag_student = selected_flag_teacher
                    current_step += 1

                    state, reward, reward_student, done = self.env.step(
                        selected_teacher,
                        selected_student,
                        selected_flag_teacher,
                        selected_flag_student,
                    )

                print("Get first complete solution!")

                best_select_node_list = torch.cat(
                    (
                        self.env.selected_student_list.reshape(batch_size, -1, 1),
                        self.env.selected_student_flag.reshape(batch_size, -1, 1),
                    ),
                    dim=2,
                )

            current_best_length = self.env._get_travel_distance_2(
                self.origin_problem, best_select_node_list
            )

            escape_time = "--:--"
            if clock is not None:
                escape_time, _ = clock.get_est_string(1, 1)

            if self.env_params["test_in_vrplib"]:
                greedy_gap = (
                    (current_best_length.item() - self.optimal_length)
                    / self.optimal_length
                ) * 100
                self.logger.info(
                    f"Greedy, name:{name}, gap:{greedy_gap:6f} %, Elapsed[{escape_time}], "
                    f"stu_l:{current_best_length.item():6f} , opt_l:{self.optimal_length:6f}"
                )
            else:
                greedy_gap = (
                    (current_best_length.mean() - self.optimal_length.mean())
                    / self.optimal_length.mean()
                ).item() * 100
                self.logger.info(
                    f"Greedy, name:{name}, gap:{greedy_gap:6f} %, Elapsed[{escape_time}], "
                    f"stu_l:{current_best_length.mean().item():6f} , opt_l:{self.optimal_length.mean().item():6f}"
                )

            ####################################################

            budget = self.env_params["RRC_budget"]

            for bbbb in range(budget):
                torch.cuda.empty_cache()

                # 1. The complete solution is obtained, which corresponds to the problems of the current env

                self.env.load_problems(episode, batch_size)

                # 2. Sample the partial solution, reset env, and assign the first node and last node in env

                best_select_node_list = (
                    self.env.vrp_whole_and_solution_subrandom_inverse(
                        best_select_node_list, self.env_params["test_in_vrplib"]
                    )
                )

                (
                    partial_solution_length,
                    first_node_index,
                    length_of_subpath,
                    double_solution,
                ) = self.env.destroy_solution(self.env.problems, best_select_node_list)

                before_reward = partial_solution_length

                current_step = 0

                _, _, _ = self.env.reset(self.env_params["mode"])

                state, reward, reward_student, done = (
                    self.env.pre_step()
                )  # state: data, first_node = current_node

                # 3. Generate solution 2 again, compare the path lengths of solution 1 and solution 2,
                # and decide which path to accept.

                while not done:
                    if current_step == 0:
                        selected_teacher = self.env.solution[:, 0, 0]
                        selected_flag_teacher = self.env.solution[:, 0, 1]
                        selected_student = selected_teacher
                        selected_flag_student = selected_flag_teacher

                    else:
                        (
                            _,
                            selected_teacher,
                            selected_student,
                            selected_flag_teacher,
                            selected_flag_student,
                        ) = self.model(
                            state,
                            self.env.selected_node_list,
                            self.env.solution,
                            current_step,
                            raw_data_capacity=self.env.raw_data_capacity,
                            **kwargs,
                        )

                    current_step += 1

                    state, reward, reward_student, done = self.env.step(
                        selected_teacher,
                        selected_student,
                        selected_flag_teacher,
                        selected_flag_student,
                    )

                ahter_repair_sub_solution = torch.cat(
                    (
                        self.env.selected_student_list.unsqueeze(2),
                        self.env.selected_student_flag.unsqueeze(2),
                    ),
                    dim=2,
                )

                after_reward = -reward_student

                after_repair_complete_solution = self.decide_whether_to_repair_solution(
                    ahter_repair_sub_solution,
                    before_reward,
                    after_reward,
                    first_node_index,
                    length_of_subpath,
                    double_solution,
                )

                best_select_node_list = after_repair_complete_solution

                current_best_length = self.env._get_travel_distance_2(
                    self.origin_problem, best_select_node_list
                )

                escape_time = "--:--"
                if clock is not None:
                    escape_time, _ = clock.get_est_string(1, 1)
                if self.env_params["test_in_vrplib"]:
                    rrc_gap = (
                        (current_best_length.item() - self.optimal_length)
                        / self.optimal_length
                    ) * 100
                    self.logger.info(
                        f"RRC step{bbbb}, name:{name}, gap:{rrc_gap:6f} %, Elapsed[{escape_time}], "
                        f"stu_l:{current_best_length.item():6f} , opt_l:{self.optimal_length:6f}"
                    )
                else:
                    rrc_gap = (
                        (current_best_length.mean() - self.optimal_length.mean())
                        / self.optimal_length.mean()
                    ).item() * 100
                    self.logger.info(
                        f"RRC step{bbbb}, name:{name}, gap:{rrc_gap:6f} %, Elapsed[{escape_time}], "
                        f"stu_l:{current_best_length.mean().item():6f} , opt_l:{self.optimal_length.mean().item():6f}"
                    )

            current_best_length = self.env._get_travel_distance_2(
                self.origin_problem, best_select_node_list
            )
            if self.env_params["test_in_vrplib"]:
                print(
                    "current_best_length",
                    (current_best_length.item() - self.optimal_length)
                    / self.optimal_length
                    * 100,
                    "%",
                    "escape time:",
                    escape_time,
                    f"optimal:{self.optimal_length}, current_best:{current_best_length.item()}",
                )
            else:
                print(
                    "current_best_length",
                    (current_best_length.mean() - self.optimal_length.mean())
                    / self.optimal_length.mean()
                    * 100,
                    "%",
                    "escape time:",
                    escape_time,
                    f"optimal:{self.optimal_length.mean()}, current_best:{current_best_length.mean()}",
                )

            if self.env_params["test_in_vrplib"]:
                return (
                    self.optimal_length,
                    current_best_length.item(),
                    self.env.problem_size,
                    name,
                )
            else:
                return (
                    self.optimal_length.mean().item(),
                    current_best_length.mean().item(),
                    self.env.problem_size,
                    name,
                )
