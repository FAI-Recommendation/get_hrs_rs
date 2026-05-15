"""
train.py — Training script cho CombiGCN (PyG)
===============================================
Usage:
  python train.py --data_path ../get10k_data/clip_10k_sample --sim_type img_only
  python train.py --data_path ../get10k_data/clip_10k_sample --sim_type multimodal
  python train.py --data_path ../get10k_data/clip_10k_sample --sim_type none
  python train.py --data_path ../get10k_data/clip_10k_sample --sim_type tfidf

  # Voi wandb:
  python train.py ... --use_wandb 1 --wandb_project combigcn-rs --wandb_entity your-team

  # Voi HuggingFace Hub push:
  python train.py ... --use_hf 1 --hf_repo_id YourOrg/combigcn-fashion-rs
"""

import os
import sys
from time import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utility.load_data import Data
from utility.parser import parse_args
from utility.helper import early_stopping, ensureDir
from utility.batch_test import test
from model import CombiGCN, scipy_to_sparse_tensor

# Optional dependencies
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from huggingface_hub import HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# ─────────────────────────────────────────────
# Format metrics for printing
# ─────────────────────────────────────────────
def format_metrics(ret, Ks):
    lines = []
    for metric in ["recall", "precision", "ndcg", "map", "mrr", "hit_ratio"]:
        vals = ", ".join([f"{v:.5f}" for v in ret[metric]])
        lines.append(f"  {metric:>10s}@{Ks} = [{vals}]")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Push best model to HuggingFace Hub
# ─────────────────────────────────────────────
def push_to_hf(local_path, hf_repo_id, epoch, recall_score):
    """Push best_model.pt to HuggingFace Hub. No-op if HF not available."""
    if not HF_AVAILABLE:
        print("   ⚠️  huggingface_hub not installed. Skipping HF push.")
        return
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("   ⚠️  HF_TOKEN not set in environment. Skipping HF push.")
        return
    try:
        api = HfApi(token=hf_token)
        api.create_repo(
            repo_id=hf_repo_id,
            repo_type="model",
            exist_ok=True,
            private=True,
        )
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo="best_model.pt",
            repo_id=hf_repo_id,
            repo_type="model",
            commit_message=f"Best model epoch={epoch} recall@K={recall_score:.5f}",
        )
        print(f"   🤗 Pushed to HF Hub: https://huggingface.co/{hf_repo_id}")
    except Exception as e:
        print(f"   ⚠️  HF push failed: {e}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    Ks = eval(args.Ks)
    n_layers = len(eval(args.layer_size))
    decay = eval(args.regs)[0]

    # ── Auto-detect wandb / HF from environment variables ──
    # Neu .env co WANDB_API_KEY → tu dong bat wandb (khong can --use_wandb 1)
    if not args.use_wandb and os.environ.get("WANDB_API_KEY", "").strip():
        args.use_wandb = 1
    if not args.wandb_project:
        args.wandb_project = os.environ.get("WANDB_PROJECT", "combigcn-rs")
    if not args.wandb_entity:
        args.wandb_entity = os.environ.get("WANDB_ENTITY", "")
    # Neu .env co HF_TOKEN + HF_REPO_ID → tu dong bat HF push
    if not args.use_hf and os.environ.get("HF_TOKEN", "").strip():
        args.use_hf = 1
    if not args.hf_repo_id:
        args.hf_repo_id = os.environ.get("HF_REPO_ID", "")

    # ── Device ──
    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")
    print(f"🖥️  Device: {device}")

    # ── Load data ──
    data_path = os.path.join(args.data_path, args.dataset) if args.dataset else args.data_path
    print(f"\n📂 Loading data from: {data_path}")
    t0 = time()
    data = Data(data_path, args.batch_size)
    n_users, n_items = data.get_num_users_items()
    print(f"   Loaded in {time() - t0:.1f}s")

    # ── Build adjacency matrices ──
    print(f"\n🔨 Building adjacency matrices ...")
    t1 = time()
    matrices = data.get_norm_adj_mat(sim_type=args.sim_type)
    interaction_adj = scipy_to_sparse_tensor(matrices[0], device=device)

    if args.sim_type == "none":
        similarity_adj = None
        print(f"   Mode: LightGCN thuan (no similarity graph)")
    else:
        sim_map = {
            "tfidf":      matrices[3],
            "bert":       matrices[4],
            "full_bert":  matrices[5],
            "multimodal": matrices[6],
            "img_only":   matrices[7],
        }
        similarity_adj = scipy_to_sparse_tensor(sim_map[args.sim_type], device=device)
        print(f"   Mode: CombiGCN with similarity_type={args.sim_type}")
    print(f"   Built in {time() - t1:.1f}s")

    # ── Create model ──
    model = CombiGCN(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=args.embed_size,
        n_layers=n_layers,
        decay=decay,
        node_dropout=eval(args.node_dropout)[0] if args.node_dropout_flag else 0.0,
    ).to(device)
    print(f"\n🏗️  Model: {model}")

    # ── torch.compile disabled: không tương thích với torch_sparse custom kernels ──
    # torch_sparse dùng ind2ptr (custom CUDA op) không thể trace bằng TorchDynamo

    # ── AMP: Automatic Mixed Precision fp16 (~2x faster, ít VRAM hơn) ──
    use_amp = torch.cuda.is_available()
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"🔥 AMP (fp16): {'ON' if use_amp else 'OFF'}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── Run name (dung cho ca TensorBoard, wandb) ──
    run_name = args.wandb_run_name or (
        f"{args.sim_type}_layers{n_layers}_dim{args.embed_size}_lr{args.lr}_reg{decay}"
    )

    # ── TensorBoard (luon bat) ──
    # PyTorch dung torch.utils.tensorboard.SummaryWriter — cung format voi TF
    # Chay: tensorboard --logdir rs/lightgcn_pyg/tensorboard/
    log_dir = f"tensorboard/{run_name}"
    writer = SummaryWriter(log_dir)
    print(f"📊 TensorBoard: {log_dir}")

    # ── Wandb (optional) ──
    use_wandb = bool(args.use_wandb)
    if use_wandb:
        if not WANDB_AVAILABLE:
            print("⚠️  wandb chua duoc cai. Tat wandb. Chay: pip install wandb")
            use_wandb = False
        else:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                name=run_name,
                config={
                    "sim_type":   args.sim_type,
                    "embed_size": args.embed_size,
                    "n_layers":   n_layers,
                    "lr":         args.lr,
                    "decay":      decay,
                    "batch_size": args.batch_size,
                    "n_users":    n_users,
                    "n_items":    n_items,
                },
            )
            print(f"📈 wandb: project={args.wandb_project}, run={run_name}")

    # ── Save path ──
    weights_dir = None
    if args.save_flag:
        weights_dir = os.path.join(
            args.weights_path, args.sim_type,
            f"layers_{n_layers}_dim_{args.embed_size}",
            f"lr_{args.lr}_reg_{decay}",
        )
        os.makedirs(weights_dir, exist_ok=True)

    # ── Training loop ──
    print(f"\n🚀 Training starts — {args.epoch} epochs, batch={args.batch_size}")
    print(f"   Eval every {args.eval_interval} epochs, early_stop={args.early_stop_steps}")
    print(f"   Ks = {Ks}")
    print(f"   TensorBoard: ON | wandb: {'ON' if use_wandb else 'OFF'} | HF push: {'ON' if args.use_hf else 'OFF'}\n")

    cur_best_pre_0 = []
    stopping_step = 0
    best_epoch = 0

    loss_loger = []
    rec_loger, pre_loger, ndcg_loger = [], [], []
    hit_loger, mrr_loger = [], []

    epoch_bar = tqdm(range(1, args.epoch + 1), desc="Training", unit="epoch")

    for epoch in epoch_bar:
        t_epoch = time()
        model.train()

        epoch_loss = 0.0
        epoch_mf_loss = 0.0
        epoch_reg_loss = 0.0
        n_batch = data.n_train // args.batch_size + 1

        for _ in range(n_batch):
            users, pos_items, neg_items = data.sample()

            users_t = torch.LongTensor(users).to(device)
            pos_t = torch.LongTensor(pos_items).to(device)
            neg_t = torch.LongTensor(neg_items).to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, mf_loss, reg_loss = model(
                    interaction_adj, similarity_adj,
                    users_t, pos_t, neg_t,
                )

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item() / n_batch
            epoch_mf_loss += mf_loss.item() / n_batch
            epoch_reg_loss += reg_loss.item() / n_batch

        if np.isnan(epoch_loss):
            print("❌ ERROR: loss is nan. Stopping.")
            break

        # ── Log loss ──
        writer.add_scalar("train/loss", epoch_loss, epoch)
        writer.add_scalar("train/mf_loss", epoch_mf_loss, epoch)
        writer.add_scalar("train/reg_loss", epoch_reg_loss, epoch)

        if use_wandb:
            wandb.log({
                "train/loss":     epoch_loss,
                "train/mf_loss":  epoch_mf_loss,
                "train/reg_loss": epoch_reg_loss,
                "epoch":          epoch,
            })

        epoch_bar.set_postfix(loss=f"{epoch_loss:.4f}", mf=f"{epoch_mf_loss:.4f}")

        if args.verbose > 0 and epoch % args.verbose == 0:
            print(f"Epoch {epoch:4d} [{time() - t_epoch:.1f}s]: "
                  f"loss={epoch_loss:.5f} (mf={epoch_mf_loss:.5f} + reg={epoch_reg_loss:.5f})")

        # ── Evaluation ──
        if epoch % args.eval_interval != 0:
            continue

        t_eval = time()

        # Evaluate on train set
        ret_train = test(
            model, interaction_adj, similarity_adj, data,
            Ks, device, batch_size=args.eval_batch_size, train_set_flag=1,
        )
        print(f"\n📈 Epoch {epoch} — TRAIN metrics:")
        print(format_metrics(ret_train, Ks))

        # Evaluate on test set
        ret_test = test(
            model, interaction_adj, similarity_adj, data,
            Ks, device, batch_size=args.eval_batch_size, train_set_flag=0,
        )
        print(f"📊 Epoch {epoch} — TEST metrics:")
        print(format_metrics(ret_test, Ks))
        print(f"   (eval took {time() - t_eval:.1f}s)")

        # ── TensorBoard: metrics ──
        for metric in ["recall", "precision", "ndcg", "mrr", "hit_ratio"]:
            if len(ret_test[metric]) > 0:
                writer.add_scalar(f"test/{metric}@{Ks[0]}", ret_test[metric][0], epoch)
                writer.add_scalar(f"test/{metric}@{Ks[-1]}", ret_test[metric][-1], epoch)

        # ── Wandb: metrics ──
        if use_wandb:
            log_dict = {"epoch": epoch}
            for metric in ["recall", "precision", "ndcg", "mrr", "hit_ratio", "map"]:
                for i, k in enumerate(Ks):
                    if i < len(ret_test[metric]):
                        log_dict[f"test/{metric}@{k}"] = ret_test[metric][i]
            wandb.log(log_dict)

        # ── Log results ──
        loss_loger.append(epoch_loss)
        rec_loger.append(ret_test["recall"])
        pre_loger.append(ret_test["precision"])
        ndcg_loger.append(ret_test["ndcg"])
        hit_loger.append(ret_test["hit_ratio"])
        mrr_loger.append(ret_test["mrr"])

        # ── Early stopping ──
        cur_best_pre_0, stopping_step, should_stop = early_stopping(
            ret_test["recall"], cur_best_pre_0,
            stopping_step, expected_order="acc", flag_step=args.early_stop_steps,
        )

        # ── Save best model ──
        if (args.save_flag and weights_dir
                and len(ret_test["recall"]) > 0
                and ret_test["recall"][0] == cur_best_pre_0[0]):
            best_epoch = epoch
            best_model_path = os.path.join(weights_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
                "metrics": ret_test,
                "args": vars(args),
            }, best_model_path)
            print(f"   💾 Saved best model at epoch {epoch}")

            # ── HuggingFace Hub push ──
            if args.use_hf and args.hf_repo_id:
                push_to_hf(
                    local_path=best_model_path,
                    hf_repo_id=args.hf_repo_id,
                    epoch=epoch,
                    recall_score=ret_test["recall"][0],
                )

        if should_stop:
            print(f"\n⏹️  Early stopping triggered at epoch {epoch}.")
            break

    # ─────────────────────────────────────────
    # Final summary
    # ─────────────────────────────────────────
    writer.close()

    if len(rec_loger) > 0:
        recs = np.array(rec_loger)
        pres = np.array(pre_loger)
        ndcgs = np.array(ndcg_loger)
        hits = np.array(hit_loger)
        mrrs = np.array(mrr_loger)

        best_idx = np.argmax(recs[:, 0])
        best_eval_epoch = (best_idx + 1) * args.eval_interval

        print(f"\n{'='*70}")
        print(f"🏆 Best Iter = epoch {best_eval_epoch}  [{time() - t0:.1f}s total]")
        print(f"   recall    = [{', '.join([f'{r:.5f}' for r in recs[best_idx]])}]")
        print(f"   precision = [{', '.join([f'{r:.5f}' for r in pres[best_idx]])}]")
        print(f"   ndcg      = [{', '.join([f'{r:.5f}' for r in ndcgs[best_idx]])}]")
        print(f"   hit_ratio = [{', '.join([f'{r:.5f}' for r in hits[best_idx]])}]")
        print(f"   mrr       = [{', '.join([f'{r:.5f}' for r in mrrs[best_idx]])}]")
        print(f"{'='*70}")

        # ── Wandb: summary ──
        if use_wandb:
            wandb.summary["best_epoch"]     = best_eval_epoch
            wandb.summary["best_recall@K0"] = float(recs[best_idx][0])
            wandb.summary["best_ndcg@K0"]   = float(ndcgs[best_idx][0])
            wandb.finish()

        # ── Save result to file ──
        result_dir = os.path.join(args.output_path, args.sim_type)
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, "combigcn.result")
        with open(result_path, "a") as f:
            f.write(
                f"embed_size={args.embed_size}, lr={args.lr}, "
                f"n_layers={n_layers}, node_dropout={args.node_dropout}, "
                f"regs={args.regs}, sim_type={args.sim_type}\n"
                f"  recall=[{', '.join([f'{r:.5f}' for r in recs[best_idx]])}]\n"
                f"  precision=[{', '.join([f'{r:.5f}' for r in pres[best_idx]])}]\n"
                f"  ndcg=[{', '.join([f'{r:.5f}' for r in ndcgs[best_idx]])}]\n"
                f"  hit_ratio=[{', '.join([f'{r:.5f}' for r in hits[best_idx]])}]\n"
                f"  mrr=[{', '.join([f'{r:.5f}' for r in mrrs[best_idx]])}]\n\n"
            )
        print(f"📄 Results saved to: {result_path}")
    else:
        if use_wandb:
            wandb.finish()
        print("\n⚠️  No evaluation was performed.")


if __name__ == "__main__":
    main()
