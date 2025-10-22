from logging import getLogger
import numpy as np
import torch
import random
import pickle
import os

from lehd.TSP.TSPModel import TSPModel as Model
from lehd.TSP.TSPEnv import TSPEnv as Env
from lehd.utils.utils import AverageMeter, TimeEstimator, get_result_folder


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
