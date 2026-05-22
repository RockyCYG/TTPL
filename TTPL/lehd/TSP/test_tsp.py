import argparse
import logging
import os
import sys
import torch

torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = False

# Set the working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Add parent directories to the system path for module imports
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils
from lehd.TSP.TSPTester import TSPTester as Tester
from lehd.utils.utils import create_logger
from lehd.TSP import projection

# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = torch.cuda.is_available()
CUDA_DEVICE_NUM = 0


# Parameters for loading the pre-trained model
model_load_path = "result/TSP100_model"
model_load_epoch = 500

# Test parameters for different problem sizes
# Format: {problem_size: [test_file, test_episodes, batch_size]}
test_paras = {
    1000: ["test/MCTS_tsp1000_test_concorde.txt", 128, 128],
    5000: ["test/test_tsp5000_lkh3_n16.txt", 16, 16],
    10000: ["test/MCTS_tsp10000_test_concorde.txt", 16, 16],
    50000: ["test/test_tsp50000_lkh3_n16.txt", 16, 16],
    100000: ["test/test_tsp100000_lkh3_n16.txt", 16, 16],
    0: ["val", None, 1],
}

# Environment parameters for the TSP tester
env_params = {
    "mode": "test",
    "sub_path": False,
}

# Model parameters for the TSP model
model_params = {
    "mode": "test",
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128 ** (1 / 2),
    "decoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "ff_hidden_dim": 512,
}

# Tester parameters
tester_params = {
    "use_cuda": USE_CUDA,
    "cuda_device_num": CUDA_DEVICE_NUM,
}

# Logger parameters
logger_params = {"log_file": {"desc": "test_log", "filename": "log.txt"}}


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y", "on"):
        return True
    if value in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _resolve_data_path(script_dir, data_path):
    if os.path.isabs(data_path):
        return data_path
    return os.path.join(script_dir, "data", data_path)


def _count_tsplib_episodes(path):
    if os.path.isdir(path):
        return len(
            [
                name
                for name in os.listdir(path)
                if name.lower().endswith(".tsp")
            ]
        )
    if os.path.isfile(path) and path.lower().endswith(".tsp"):
        return 1
    if os.path.isfile(path):
        with open(path, "r") as f:
            return len([line for line in f if line.strip()])
    raise FileNotFoundError(f"Cannot find TSPLIB data path: {path}")


def _parse_tsplib_opt_costs(raw_costs):
    if not raw_costs:
        return {}

    costs = {}
    for item in raw_costs.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(
                "Invalid --tsplib_opt_costs item. Use NAME=COST, e.g. ch150=6528"
            )
        name, value = item.split("=", 1)
        costs[name.strip().lower()] = float(value)
    return costs


def main_test(args, **kwargs):
    """
    Main function to run the TSP test.
    """
    if DEBUG_MODE:
        _set_debug_mode()

    # Set up model loading path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    tester_params["model_load"] = {
        "path": os.path.join(project_root, "lehd", "TSP", args.model_load_path),
        "epoch": args.model_load_epoch,
    }

    if args.problem_size not in test_paras:
        available = ", ".join(str(key) for key in sorted(test_paras))
        raise ValueError(
            f"Unknown problem_size={args.problem_size}. Available keys: {available}"
        )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_filename, default_episodes, default_batch_size = test_paras[
        args.problem_size
    ]
    data_filename = args.tsp_data_path or default_data_filename
    data_path = _resolve_data_path(script_dir, data_filename)

    if args.test_in_tsplib:
        test_episodes = (
            args.test_episodes
            if args.test_episodes is not None
            else _count_tsplib_episodes(data_path)
        )
        test_batch_size = args.test_batch_size or default_batch_size or 1
    else:
        test_episodes = args.test_episodes or default_episodes
        test_batch_size = args.test_batch_size or default_batch_size

    if test_episodes is None or test_batch_size is None:
        raise ValueError("test_episodes and test_batch_size must be configured.")

    if args.use_cuda and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")
        args.use_cuda = False

    # Configure logger description based on arguments
    logger_params["log_file"]["desc"] = (
        f"test_counter_{args.counter_current}_tsplib{args.test_in_tsplib}_tsp{args.problem_size}_"
        f"RRC{args.RRC_budget}_range{args.RRC_range}_knearest{args.knearest}_"
        f"num{args.k_nearest_nodes}_RI_{args.random_insertion}_projection_{args.coor_projection}"
    )

    # Update parameters from arguments
    tester_params["use_cuda"] = args.use_cuda
    tester_params["cuda_device_num"] = args.cuda_device_num
    tester_params["test_episodes"] = test_episodes
    tester_params["test_batch_size"] = test_batch_size
    tester_params["inference_backend"] = args.inference_backend
    tester_params["pomo_aug_factor"] = args.pomo_aug_factor
    tester_params["pomo_log_dist_bias"] = args.pomo_log_dist_bias
    tester_params["pomo_log_dist_topk"] = args.pomo_log_dist_topk
    tester_params["pomo_log_dist_alpha"] = args.pomo_log_dist_alpha
    tester_params["pomo_log_dist_eps"] = args.pomo_log_dist_eps
    model_params["k_nearest_nodes"] = args.k_nearest_nodes
    model_params["knearest"] = args.knearest
    model_params["coor_projection"] = args.coor_projection

    # Set data paths
    env_params["data_path"] = data_path
    env_params["tsplib_path"] = data_path

    # Update environment parameters from arguments
    env_params["test_in_tsplib"] = args.test_in_tsplib
    env_params["tsplib_opt_costs"] = _parse_tsplib_opt_costs(args.tsplib_opt_costs)
    env_params["RRC_budget"] = args.RRC_budget
    env_params["random_insertion"] = args.random_insertion
    env_params["RRC_range"] = args.RRC_range

    # Initialize logger and print configuration
    create_logger(**logger_params)
    _print_config()

    # Initialize and run the tester
    tester = Tester(
        env_params=env_params, model_params=model_params, tester_params=tester_params
    )

    llm_projection = kwargs.get("projection", None)
    score_optimal, score_student, gap = tester.run(
        projection=(
            llm_projection
            if llm_projection is not None
            else getattr(projection, args.projection)
        ),
        logit_bias=kwargs.get("logit_bias"),
        MVDF=args.MVDF if hasattr(args, "MVDF") else False,
    )
    return score_optimal, score_student, gap


def _set_debug_mode():
    """
    Sets the number of test episodes for debug mode.
    """
    global tester_params
    tester_params["test_episodes"] = 100


def _print_config():
    """
    Prints the configuration parameters.
    """
    logger = logging.getLogger("root")
    logger.info(f"DEBUG_MODE: {DEBUG_MODE}")
    logger.info(f"USE_CUDA: {USE_CUDA}, CUDA_DEVICE_NUM: {CUDA_DEVICE_NUM}")
    for g_key in globals().keys():
        if g_key.endswith("params"):
            logger.info(f"{g_key}{globals()[g_key]}")


def add_common_args(parser):
    """
    Adds common command-line arguments to the parser.
    """
    parser.add_argument(
        "--cuda_device_num", type=int, default=0, help="CUDA device number"
    )
    parser.add_argument(
        "--use_cuda",
        type=_str_to_bool,
        default=USE_CUDA,
        help="Whether to use CUDA when it is available",
    )
    parser.add_argument(
        "--problem_size", type=int, default=0, help="The size of the problem"
    )
    parser.add_argument(
        "--test_in_tsplib",
        type=_str_to_bool,
        default=True,
        help="Whether to test on TSPLib instances",
    )
    parser.add_argument(
        "--tsp_data_path",
        type=str,
        default=None,
        help="TSP data file or directory, relative to lehd/TSP/data unless absolute",
    )
    parser.add_argument(
        "--test_episodes",
        type=int,
        default=None,
        help="Override number of test instances",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=None,
        help="Override test batch size",
    )
    parser.add_argument(
        "--tsplib_opt_costs",
        type=str,
        default=None,
        help="Optional comma-separated TSPLIB best costs, e.g. ch150=6528,pr226=80369",
    )
    parser.add_argument(
        "--RRC_budget", type=int, default=0, help="Budget for Ruin and Recreate"
    )
    parser.add_argument(
        "--RRC_range", type=int, default=1000, help="Range for Ruin and Recreate"
    )
    parser.add_argument(
        "--random_insertion",
        type=_str_to_bool,
        default=False,
        help="Whether to use random insertion",
    )
    parser.add_argument(
        "--knearest",
        type=_str_to_bool,
        default=True,
        help="Whether to use k-nearest neighbors",
    )
    parser.add_argument(
        "--k_nearest_nodes",
        type=int,
        default=100,
        help="Number of nearest nodes to consider",
    )
    parser.add_argument(
        "--coor_projection",
        type=_str_to_bool,
        default=True,
        help="Whether to use coordinate projection",
    )
    parser.add_argument(
        "--counter_current", type=int, default=0, help="Current counter for logging"
    )
    parser.add_argument(
        "--projection",
        type=str,
        default="projection_1k",
        help="Projection method to use",
    )
    parser.add_argument(
        "--MVDF",
        type=_str_to_bool,
        default=True,
        help="Whether to use the MVDF projection method",
    )
    parser.add_argument(
        "--model_load_epoch",
        type=int,
        default=model_load_epoch,
        help="Epoch number of the model to load",
    )
    parser.add_argument(
        "--model_load_path",
        type=str,
        default="result/TSP100_model",
        help="Path to the model to load",
    )
    parser.add_argument(
        "--inference_backend",
        type=str,
        default="auto",
        choices=("auto", "ttpl", "pomo"),
        help="Inference path to use; auto selects POMO for PolyNet checkpoints",
    )
    parser.add_argument(
        "--pomo_aug_factor",
        type=int,
        default=8,
        choices=(1, 8),
        help="POMO augmentation factor for TSPLIB inference",
    )
    parser.add_argument(
        "--pomo_log_dist_bias",
        type=_str_to_bool,
        default=False,
        help="Whether to add fixed log-distance bias to POMO decoder logits",
    )
    parser.add_argument(
        "--pomo_log_dist_topk",
        type=int,
        default=20,
        help="Top-K nearest candidates using -log(distance); others use -distance",
    )
    parser.add_argument(
        "--pomo_log_dist_alpha",
        type=float,
        default=1.0,
        help="Scale for fixed POMO log-distance bias",
    )
    parser.add_argument(
        "--pomo_log_dist_eps",
        type=float,
        default=1e-6,
        help="Distance epsilon for fixed POMO log-distance bias",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for TSP")
    add_common_args(parser)
    args = parser.parse_args()

    main_test(args)
