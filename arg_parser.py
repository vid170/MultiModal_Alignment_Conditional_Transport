import argparse


def parse_arguments():
    # Training settings
    parser = argparse.ArgumentParser(
        description="MultiModal ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=128,
        metavar="B",
        help="Batch size",
    )
 
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=40,
        metavar="N",
        help="Total number of epochs to train",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="test_run",
        help="optional identifier of experiment",
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        default=False,
        help="Disable wandb logging",
    )
 
    parser.add_argument(
        "--datapoints",
        type=int,
        default=3000,
        help="Number of samples of training data allotted to each client",
    )

    parser.add_argument(
        "--loss",
        type=str,
        default="ct_bce",
        help="Loss function to use",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default="ct_bce",
        help="Regularization Parameter",
    )
   
    args = parser.parse_args()
    return args