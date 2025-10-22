import argparse
import logging
import os
import sys

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
USE_CUDA = True
CUDA_DEVICE_NUM = 1


# Parameters for loading the pre-trained model
model_load_path = "result/TSP100_model"
model_load_epoch = 150

# Test parameters for different problem sizes
# Format: {problem_size: [test_file, test_episodes, batch_size]}
test_paras = {
    1000: ["test/MCTS_tsp1000_test_concorde.txt", 128, 128],
    5000: ["test/test_tsp5000_lkh3_n16.txt", 16, 16],
    10000: ["test/MCTS_tsp10000_test_concorde.txt", 16, 16],
    50000: ["test/test_tsp50000_lkh3_n16.txt", 16, 16],
    100000: ["test/test_tsp100000_lkh3_n16.txt", 16, 16],
    0: ["test/TSPlib_scale_ge_1K_n33_ascending.txt", 33, 1],
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

    # Configure logger description based on arguments
    logger_params["log_file"]["desc"] = (
        f"test_counter_{args.counter_current}_tsplib{args.test_in_tsplib}_tsp{args.problem_size}_"
        f"RRC{args.RRC_budget}_range{args.RRC_range}_knearest{args.knearest}_"
        f"num{args.k_nearest_nodes}_RI_{args.random_insertion}_projection_{args.coor_projection}"
    )

    # Update parameters from arguments
    tester_params["cuda_device_num"] = args.cuda_device_num
    tester_params["test_episodes"] = test_paras[args.problem_size][1]
    tester_params["test_batch_size"] = test_paras[args.problem_size][2]
    model_params["k_nearest_nodes"] = args.k_nearest_nodes
    model_params["knearest"] = args.knearest
    model_params["coor_projection"] = args.coor_projection

    # Set data paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_filename = test_paras[args.problem_size][0]
    env_params["data_path"] = os.path.join(script_dir, "data", data_filename)
    env_params["tsplib_path"] = os.path.join(script_dir, "data", data_filename)

    # Update environment parameters from arguments
    env_params["test_in_tsplib"] = args.test_in_tsplib
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
        "--problem_size", type=int, default=1000, help="The size of the problem"
    )
    parser.add_argument(
        "--test_in_tsplib",
        type=bool,
        default=False,
        help="Whether to test on TSPLib instances",
    )
    parser.add_argument(
        "--RRC_budget", type=int, default=0, help="Budget for Ruin and Recreate"
    )
    parser.add_argument(
        "--RRC_range", type=int, default=1000, help="Range for Ruin and Recreate"
    )
    parser.add_argument(
        "--random_insertion",
        type=bool,
        default=False,
        help="Whether to use random insertion",
    )
    parser.add_argument(
        "--knearest", type=bool, default=True, help="Whether to use k-nearest neighbors"
    )
    parser.add_argument(
        "--k_nearest_nodes",
        type=int,
        default=100,
        help="Number of nearest nodes to consider",
    )
    parser.add_argument(
        "--coor_projection",
        type=bool,
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
        type=bool,
        default=True,
        help="Whether to use the MVDF projection method",
    )
    parser.add_argument(
        "--model_load_epoch",
        type=int,
        default=150,
        help="Epoch number of the model to load",
    )
    parser.add_argument(
        "--model_load_path",
        type=str,
        default="result/TSP100_model",
        help="Path to the model to load",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for TSP")
    add_common_args(parser)
    args = parser.parse_args()

    main_test(args)
