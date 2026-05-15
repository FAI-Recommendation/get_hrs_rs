"""
parser.py — Argument parser cho CombiGCN (PyG)
================================================
Convert từ hr/utility/parser.py — giữ tương thích args cũ + thêm --sim_type.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train CombiGCN (PyG)")

    # ── Data ──
    parser.add_argument("--data_path", nargs="?", default="Data/",
                        help="Input data path.")
    parser.add_argument("--dataset", nargs="?", default="clip_10k_sample",
                        help="Choose a dataset folder name.")
    parser.add_argument("--sim_type", type=str, default="img_only",
                        choices=["none", "multimodal", "img_only", "tfidf", "bert", "full_bert"],
                        help="Similarity type: none=LightGCN thuần, khác=CombiGCN")

    # ── Model ──
    parser.add_argument("--embed_size", type=int, default=64,
                        help="Embedding size.")
    parser.add_argument("--layer_size", nargs="?", default="[64, 64, 64]",
                        help="Output sizes of every layer (dùng len() để xác định n_layers)")
    parser.add_argument("--regs", nargs="?", default="[1e-5]",
                        help="Regularizations.")
    parser.add_argument("--node_dropout_flag", type=int, default=0,
                        help="0: Disable node dropout, 1: Activate node dropout")
    parser.add_argument("--node_dropout", nargs="?", default="[0.1]",
                        help="Keep probability w.r.t. node dropout for each deep layer.")
    parser.add_argument("--mess_dropout", nargs="?", default="[0.1]",
                        help="Keep probability w.r.t. message dropout for each deep layer.")

    # ── Training ──
    parser.add_argument("--epoch", type=int, default=1000,
                        help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate.")
    parser.add_argument("--eval_batch_size", type=int, default=1024,
                        help="Evaluation user batch size.")

    # ── Evaluation ──
    parser.add_argument("--Ks", nargs="?", default="[1,5,10,20,50]",
                        help="Top k(s) recommend")
    parser.add_argument("--eval_interval", type=int, default=10,
                        help="Evaluate every N epochs")
    parser.add_argument("--early_stop_steps", type=int, default=5,
                        help="Early stopping patience (in eval intervals)")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Print training loss every N epochs.")
    parser.add_argument("--test_flag", nargs="?", default="part",
                        help="Specify the test type from {part, full}")

    # ── Save / Load ──
    parser.add_argument("--save_flag", type=int, default=1,
                        help="0: Disable model saver, 1: Activate model saver")
    parser.add_argument("--checkpoint_interval", type=int, default=0,
                        help="Luu checkpoint moi N epoch (0 = tat). VD: 50 = luu moi 50 epoch")
    parser.add_argument("--weights_path", nargs="?", default="weights/",
                        help="Store model path.")
    parser.add_argument("--output_path", type=str, default="output/",
                        help="Directory to save results")
    parser.add_argument("--pretrain", type=int, default=0,
                        help="0: No pretrain, 1: Load pretrained weights")
    parser.add_argument("--proj_path", nargs="?", default="",
                        help="Project path.")

    # ── Device ──
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU device ID")

    # ── Wandb (optional) ──
    parser.add_argument("--use_wandb", type=int, default=0,
                        help="0: Disable wandb, 1: Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="combigcn-rs",
                        help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default="",
                        help="Wandb entity/team name (de trong neu dung personal account)")
    parser.add_argument("--wandb_run_name", type=str, default="",
                        help="Wandb run name (auto-generated neu de trong)")

    # ── HuggingFace Hub (optional) ──
    parser.add_argument("--use_hf", type=int, default=0,
                        help="0: Disable HF push, 1: Push best_model.pt to HuggingFace Hub")
    parser.add_argument("--hf_repo_id", type=str, default="",
                        help="HuggingFace repo id, vd: YourOrg/combigcn-fashion-rs")

    # ── Compat (giữ lại cho tương thích) ──
    parser.add_argument("--model_type", nargs="?", default="combigcn",
                        help="Specify the name of model.")
    parser.add_argument("--alg_type", nargs="?", default="combigcn",
                        help="Specify the type of the graph convolutional layer.")
    parser.add_argument("--adj_type", nargs="?", default="mean",
                        help="Specify the type of the adjacency matrix from {plain, norm, mean}.")
    parser.add_argument("--is_norm", type=int, default=1,
                        help="Normalization flag.")

    return parser.parse_args()
