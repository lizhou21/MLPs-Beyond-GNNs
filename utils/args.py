import argparse


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", default="/data/jiangh/ExplainMLP/datasets/parse", type=str, help="SentEval or EdgeProb"
    )
    parser.add_argument(
        "--log_dir", default="./logs", type=str, help="SentEval or EdgeProb"
    )
    parser.add_argument(
        "--task_type", default="EdgeProb", type=str, help="SentEval or EdgeProb"
    )
    parser.add_argument(
        "--data_name", default="spr2", type=str, help="coordination_inversion.txt"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--model_name",
        default="/data1/public/bert/bert-base-uncased",
        type=str,
    )  # original bert-base-uncased
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated.",
    )
    parser.add_argument(
        "--train_batch_size", default=64, type=int, help="Batch size for training."
    )
    parser.add_argument(
        "--test_batch_size", default=64, type=int, help="Batch size for testing."
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=10,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--do_middle", action="store_true", help="Whether to run middle component", default=False
    )
    parser.add_argument(
        "--middle_format", default="KAN", type=str, help="in [MLP, GNN, Message, KAN]"
    )
    parser.add_argument(
        "--token_format", default="mean", type=str, help="in [max, mean]"
    )
    parser.add_argument("--mid_layers", default=2, type=int, help="Middle Layers")
    parser.add_argument(
        "--prob_layers",
        default=-1,
        type=int,
        help="0: embedding layer; -1: last layer before LMH",
    )
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument(
        "--warmup_ratio", default=0.1, type=float, help="Warm up ratio for Adam."
    )
    parser.add_argument(
        "--evaluation_steps",
        type=int,
        default=2,
        help="Number of steps to evaluate the model",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument("--runs", default=0, type=int, help="")
    # parser.add_argument("--random_graph", action="store_true", help="random create edge", default=True)

    args = parser.parse_args()
    return args
