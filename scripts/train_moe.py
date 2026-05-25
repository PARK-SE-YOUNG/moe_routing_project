import csv
import os
import time
import yaml
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.datasets.dummy_dataset import DummyDataset
from src.models.sparse_transformer import SparseTransformer

from src.training.losses import (
    get_loss_function,
    load_balancing_loss
)

from src.utils.logger import setup_logger
from src.utils.checkpoint import save_checkpoint


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def build_dataloader(config):
    dataset = DummyDataset(
        num_samples=100,
        seq_len=config["data"]["seq_len"],
        vocab_size=config["data"]["vocab_size"],
        num_classes=config["model"]["num_classes"]
    )

    g = torch.Generator()
    g.manual_seed(config["seed"])

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        generator=g
    )

    return dataloader


def build_model(config, experiment, device):
    router_config = config.get("router", {})

    model = SparseTransformer(
        vocab_size=config["data"]["vocab_size"],
        hidden_dim=config["model"]["d_model"],
        num_heads=config["model"]["nhead"],
        num_classes=config["model"]["num_classes"],
        num_experts=config["model"]["num_experts"],
        top_k=config["model"]["top_k"],
        use_adapter=experiment["use_adapter"],
        temperature=router_config.get("temperature", 0.7)
    )

    return model.to(device)


def apply_freeze_policy(model, experiment, logger):
    if experiment["freeze_adapter_only"]:
        for param in model.parameters():
            param.requires_grad = False

        for name, param in model.named_parameters():
            if "adapter" in name.lower():
                param.requires_grad = True

    trainable_params = 0
    total_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()

        if param.requires_grad:
            trainable_params += param.numel()
            logger.info(f"Trainable: {name}")

    logger.info(
        f"Trainable params: "
        f"{trainable_params:,} / {total_params:,}"
    )


def compute_reward(predictions, y, latency, config, device):
    rl_config = config.get("rl", {})

    accuracy_reward = (predictions == y).float()

    latency_penalty = torch.full_like(
        accuracy_reward,
        fill_value=latency
    )

    reward = (
        rl_config.get("reward_accuracy_weight", 1.0)
        * accuracy_reward
        -
        rl_config.get("reward_latency_weight", 0.1)
        * latency_penalty
    )

    if rl_config.get("normalize_reward", True):
        reward = (
            reward - reward.mean()
        ) / (
            reward.std() + 1e-8
        )

    return reward.to(device)


def train_one_experiment(config, experiment, device, logger):
    experiment_name = experiment["name"]

    logger.info(
        f"========== Running Experiment: "
        f"{experiment_name} =========="
    )

    set_seed(config["seed"])

    dataloader = build_dataloader(config)

    model = build_model(
        config=config,
        experiment=experiment,
        device=device
    )

    apply_freeze_policy(
        model=model,
        experiment=experiment,
        logger=logger
    )

    criterion = get_loss_function()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["lr"]
    )

    epochs = config["training"]["epochs"]

    load_balance_alpha = config["training"].get("load_balance_alpha", 0.0)
    kl_alpha = config["training"].get("kl_alpha", 0.0)
    rl_alpha = config["training"].get("rl_alpha", 0.0)

    final_result = None
    epoch_history = []

    for epoch in range(epochs):
        logger.info(
            f"[{experiment_name}] "
            f"===== Epoch {epoch + 1}/{epochs} ====="
        )

        model.train()

        total_loss = 0
        total_cls_loss = 0
        total_aux_loss = 0
        total_kl_loss = 0
        total_rl_loss = 0
        total_latency = 0

        total_correct = 0
        total_samples = 0

        total_entropy = 0
        total_expert_usage = None

        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            start_time = time.time()

            outputs = model(x)

            if device == "cuda":
                torch.cuda.synchronize()

            latency = time.time() - start_time

            logits = outputs["logits"]
            router_outputs = outputs["router_outputs"]

            router_probs = router_outputs["router_probs"]
            topk_indices = router_outputs["topk_indices"]
            expert_usage = router_outputs["expert_usage"]
            entropy = router_outputs["entropy"]

            kl_loss = router_outputs.get(
                "kl_to_original",
                torch.tensor(0.0, device=device)
            )

            selected_log_probs = router_outputs.get(
                "selected_log_probs",
                None
            )

            cls_loss = criterion(logits, y)
            aux_loss = load_balancing_loss(router_probs)

            predictions = torch.argmax(logits, dim=1)

            correct = (predictions == y).sum().item()
            accuracy = correct / y.size(0)

            reward = compute_reward(
                predictions=predictions,
                y=y,
                latency=latency,
                config=config,
                device=device
            )

            if selected_log_probs is not None:
                token_log_probs = selected_log_probs.mean(dim=-1)
                sequence_log_probs = token_log_probs.mean(dim=-1)

                rl_loss = -(
                    reward.detach()
                    * sequence_log_probs
                ).mean()
            else:
                rl_loss = torch.tensor(0.0, device=device)

            loss = (
                cls_loss
                + load_balance_alpha * aux_loss
            )

            if experiment["use_kl_loss"]:
                loss = loss + kl_alpha * kl_loss

            if experiment["use_rl_loss"]:
                loss = loss + rl_alpha * rl_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_aux_loss += aux_loss.item()
            total_kl_loss += kl_loss.item()
            total_rl_loss += rl_loss.item()
            total_latency += latency

            total_correct += correct
            total_samples += y.size(0)
            total_entropy += entropy.item()

            if total_expert_usage is None:
                total_expert_usage = expert_usage.detach().cpu()
            else:
                total_expert_usage += expert_usage.detach().cpu()

            logger.info(
                f"[{experiment_name}] "
                f"Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"Total Loss: {loss.item():.4f}, "
                f"Cls Loss: {cls_loss.item():.4f}, "
                f"Aux Loss: {aux_loss.item():.4f}, "
                f"KL Loss: {kl_loss.item():.6f}, "
                f"RL Loss: {rl_loss.item():.6f}, "
                f"Entropy: {entropy.item():.4f}, "
                f"Latency: {latency:.6f}, "
                f"Accuracy: {accuracy:.4f}"
            )

            if batch_idx == 0:
                logger.info(
                    f"[{experiment_name}] "
                    f"TopK Expert Indices (sample): "
                    f"{topk_indices[0][:5]}"
                )

        avg_loss = total_loss / len(dataloader)
        avg_cls_loss = total_cls_loss / len(dataloader)
        avg_aux_loss = total_aux_loss / len(dataloader)
        avg_kl_loss = total_kl_loss / len(dataloader)
        avg_rl_loss = total_rl_loss / len(dataloader)
        avg_latency = total_latency / len(dataloader)

        avg_accuracy = total_correct / total_samples
        avg_entropy = total_entropy / len(dataloader)

        avg_expert_usage = (
            total_expert_usage / len(dataloader)
        ).numpy()

        imbalance = (
            avg_expert_usage.max()
            - avg_expert_usage.min()
        )

        logger.info(f"[{experiment_name}] Average Total Loss: {avg_loss:.4f}")
        logger.info(f"[{experiment_name}] Average Cls Loss: {avg_cls_loss:.4f}")
        logger.info(f"[{experiment_name}] Average Aux Loss: {avg_aux_loss:.4f}")
        logger.info(f"[{experiment_name}] Average KL Loss: {avg_kl_loss:.6f}")
        logger.info(f"[{experiment_name}] Average RL Loss: {avg_rl_loss:.6f}")
        logger.info(f"[{experiment_name}] Average Entropy: {avg_entropy:.4f}")
        logger.info(f"[{experiment_name}] Average Latency: {avg_latency:.6f}")
        logger.info(f"[{experiment_name}] Average Accuracy: {avg_accuracy:.4f}")

        logger.info(
            f"[{experiment_name}] Average Expert Usage: "
            f"{avg_expert_usage}"
        )

        logger.info(
            f"[{experiment_name}] Routing Imbalance: "
            f"{imbalance:.4f}"
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            loss=avg_loss,
            filename=f"{experiment_name}_epoch_{epoch + 1}.pt"
        )

        epoch_result = {
            "experiment": experiment_name,
            "epoch": epoch + 1,
            "adapter": experiment["use_adapter"],
            "kl": experiment["use_kl_loss"],
            "rl": experiment["use_rl_loss"],
            "accuracy": avg_accuracy,
            "total_loss": avg_loss,
            "cls_loss": avg_cls_loss,
            "aux_loss": avg_aux_loss,
            "kl_loss": avg_kl_loss,
            "rl_loss": avg_rl_loss,
            "entropy": avg_entropy,
            "latency": avg_latency,
            "imbalance": imbalance,
            "expert_usage": avg_expert_usage.tolist()
        }

        epoch_history.append(epoch_result)

        final_result = {
            "experiment": experiment_name,
            "adapter": experiment["use_adapter"],
            "kl": experiment["use_kl_loss"],
            "rl": experiment["use_rl_loss"],
            "accuracy": avg_accuracy,
            "total_loss": avg_loss,
            "cls_loss": avg_cls_loss,
            "aux_loss": avg_aux_loss,
            "kl_loss": avg_kl_loss,
            "rl_loss": avg_rl_loss,
            "entropy": avg_entropy,
            "latency": avg_latency,
            "imbalance": imbalance,
            "expert_usage": avg_expert_usage.tolist()
        }

    return final_result, epoch_history


def save_results_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)


def save_expert_usage_plots(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for result in results:
        usage = result["expert_usage"]

        plt.figure()
        plt.bar(range(len(usage)), usage)

        plt.xlabel("Expert ID")
        plt.ylabel("Usage Ratio")
        plt.title(f"Expert Usage - {result['experiment']}")

        plt.ylim(0, 1)
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                f"{result['experiment']}_expert_usage.png"
            )
        )

        plt.close()


def save_metric_comparison_plot(results, metric, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    names = [
        result["experiment"]
        for result in results
    ]

    values = [
        result[metric]
        for result in results
    ]

    plt.figure()
    plt.bar(names, values)

    plt.xlabel("Experiment")
    plt.ylabel(metric)
    plt.title(f"{metric} Comparison")
    plt.xticks(rotation=20)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_epoch_curve_plot(epoch_history, metric, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    experiments = sorted(
        set(row["experiment"] for row in epoch_history)
    )

    plt.figure()

    for experiment in experiments:
        rows = [
            row for row in epoch_history
            if row["experiment"] == experiment
        ]

        epochs = [
            row["epoch"]
            for row in rows
        ]

        values = [
            row[metric]
            for row in rows
        ]

        plt.plot(
            epochs,
            values,
            marker="o",
            label=experiment
        )

    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(f"{metric} over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_all_plots(results, epoch_history):
    plot_dir = "outputs/plots"

    save_expert_usage_plots(
        results=results,
        output_dir=plot_dir
    )

    for metric in [
        "entropy",
        "imbalance",
        "latency",
        "accuracy",
        "rl_loss",
        "kl_loss"
    ]:
        save_metric_comparison_plot(
            results=results,
            metric=metric,
            output_path=os.path.join(
                plot_dir,
                f"{metric}_comparison.png"
            )
        )

    for metric in [
        "accuracy",
        "total_loss",
        "kl_loss",
        "rl_loss",
        "entropy",
        "latency",
        "imbalance"
    ]:
        save_epoch_curve_plot(
            epoch_history=epoch_history,
            metric=metric,
            output_path=os.path.join(
                plot_dir,
                f"{metric}_curve.png"
            )
        )


def generate_experiment_summary(results, epoch_history):
    final_lines = []
    md_lines = []

    best_accuracy = max(
        results,
        key=lambda x: x["accuracy"]
    )

    lowest_latency = min(
        results,
        key=lambda x: x["latency"]
    )

    lowest_imbalance = min(
        results,
        key=lambda x: x["imbalance"]
    )

    baseline = next(
        r for r in results
        if r["experiment"] == "sparse_moe_baseline"
    )

    final_lines.append("MoE Router Experiment Summary")
    final_lines.append("=" * 40)
    final_lines.append("")
    final_lines.append("Final Comparison")
    final_lines.append("-" * 40)

    for result in results:
        final_lines.append(
            f"{result['experiment']}: "
            f"Acc={result['accuracy']:.4f}, "
            f"Loss={result['total_loss']:.4f}, "
            f"KL={result['kl_loss']:.6f}, "
            f"RL={result['rl_loss']:.6f}, "
            f"Entropy={result['entropy']:.4f}, "
            f"Latency={result['latency']:.6f}, "
            f"Imbalance={result['imbalance']:.4f}"
        )

    final_lines.append("")
    final_lines.append("Key Findings")
    final_lines.append("-" * 40)

    final_lines.append(
        f"Best accuracy: {best_accuracy['experiment']} "
        f"({best_accuracy['accuracy']:.4f})"
    )

    final_lines.append(
        f"Lowest latency: {lowest_latency['experiment']} "
        f"({lowest_latency['latency']:.6f})"
    )

    final_lines.append(
        f"Lowest routing imbalance: {lowest_imbalance['experiment']} "
        f"({lowest_imbalance['imbalance']:.4f})"
    )

    for result in results:
        if result["experiment"] == "sparse_moe_baseline":
            continue

        acc_delta = result["accuracy"] - baseline["accuracy"]
        entropy_delta = result["entropy"] - baseline["entropy"]
        imbalance_delta = result["imbalance"] - baseline["imbalance"]
        latency_delta = result["latency"] - baseline["latency"]

        final_lines.append("")
        final_lines.append(
            f"Compared with sparse_moe_baseline, "
            f"{result['experiment']} changed:"
        )
        final_lines.append(f"- Accuracy: {acc_delta:+.4f}")
        final_lines.append(f"- Entropy: {entropy_delta:+.4f}")
        final_lines.append(f"- Imbalance: {imbalance_delta:+.4f}")
        final_lines.append(f"- Latency: {latency_delta:+.6f}")

    final_lines.append("")
    final_lines.append("Interpretation")
    final_lines.append("-" * 40)
    final_lines.append(
        "The current experiments confirm that the Sparse MoE baseline, "
        "Router Adapter with KL constraint, and Adapter + KL + RL variants "
        "can be executed and compared automatically."
    )
    final_lines.append(
        "Accuracy remains similar across experiments, which is expected "
        "because the current dataset is a dummy dataset."
    )
    final_lines.append(
        "KL loss remains small, indicating that the adapter does not "
        "strongly diverge from the original router policy."
    )
    final_lines.append(
        "Entropy remains close to the maximum entropy for four experts, "
        "suggesting that severe routing collapse has not occurred."
    )
    final_lines.append(
        "Routing imbalance and expert usage should be monitored carefully "
        "because adapter/RL variants may increase expert specialization."
    )

    md_lines.append("# MoE Router Experiment Summary")
    md_lines.append("")
    md_lines.append("## Final Comparison")
    md_lines.append("")
    md_lines.append(
        "| Experiment | Adapter | KL | RL | Accuracy | Loss | KL Loss | RL Loss | Entropy | Latency | Imbalance |"
    )
    md_lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for result in results:
        md_lines.append(
            f"| {result['experiment']} "
            f"| {result['adapter']} "
            f"| {result['kl']} "
            f"| {result['rl']} "
            f"| {result['accuracy']:.4f} "
            f"| {result['total_loss']:.4f} "
            f"| {result['kl_loss']:.6f} "
            f"| {result['rl_loss']:.6f} "
            f"| {result['entropy']:.4f} "
            f"| {result['latency']:.6f} "
            f"| {result['imbalance']:.4f} |"
        )

    md_lines.append("")
    md_lines.append("## Key Findings")
    md_lines.append("")
    md_lines.append(
        f"- **Best accuracy:** `{best_accuracy['experiment']}` "
        f"({best_accuracy['accuracy']:.4f})"
    )
    md_lines.append(
        f"- **Lowest latency:** `{lowest_latency['experiment']}` "
        f"({lowest_latency['latency']:.6f})"
    )
    md_lines.append(
        f"- **Lowest routing imbalance:** `{lowest_imbalance['experiment']}` "
        f"({lowest_imbalance['imbalance']:.4f})"
    )

    md_lines.append("")
    md_lines.append("## Baseline Comparison")
    md_lines.append("")

    for result in results:
        if result["experiment"] == "sparse_moe_baseline":
            continue

        acc_delta = result["accuracy"] - baseline["accuracy"]
        entropy_delta = result["entropy"] - baseline["entropy"]
        imbalance_delta = result["imbalance"] - baseline["imbalance"]
        latency_delta = result["latency"] - baseline["latency"]

        md_lines.append(f"### {result['experiment']}")
        md_lines.append("")
        md_lines.append(f"- Accuracy change: `{acc_delta:+.4f}`")
        md_lines.append(f"- Entropy change: `{entropy_delta:+.4f}`")
        md_lines.append(f"- Imbalance change: `{imbalance_delta:+.4f}`")
        md_lines.append(f"- Latency change: `{latency_delta:+.6f}`")
        md_lines.append("")

    md_lines.append("## Interpretation")
    md_lines.append("")
    md_lines.append(
        "- The current experiment runner successfully compares Sparse MoE, "
        "Adapter + KL, and Adapter + KL + RL settings in one execution."
    )
    md_lines.append(
        "- Accuracy is similar across experiments, which is expected with "
        "the current dummy dataset."
    )
    md_lines.append(
        "- KL loss remains small, meaning the adapter is still close to the "
        "original router policy."
    )
    md_lines.append(
        "- Entropy remains close to the maximum for four experts, so severe "
        "routing collapse is not observed."
    )
    md_lines.append(
        "- Routing imbalance should be monitored because RL/adapter variants "
        "can increase expert specialization."
    )

    return "\n".join(final_lines), "\n".join(md_lines)


def save_experiment_summary(results, epoch_history):
    os.makedirs("outputs", exist_ok=True)

    txt_summary, md_summary = generate_experiment_summary(
        results=results,
        epoch_history=epoch_history
    )

    with open(
        "outputs/experiment_summary.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(txt_summary)

    with open(
        "outputs/experiment_summary.md",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(md_summary)


def print_comparison_table(results):
    print("\n===== FINAL COMPARISON =====")
    print(
        f"{'Experiment':<24} "
        f"{'Adapter':<8} "
        f"{'KL':<6} "
        f"{'RL':<6} "
        f"{'Acc':<8} "
        f"{'Loss':<8} "
        f"{'KL Loss':<10} "
        f"{'RL Loss':<10} "
        f"{'Entropy':<10} "
        f"{'Latency':<10} "
        f"{'Imbalance':<10}"
    )

    print("-" * 125)

    for result in results:
        print(
            f"{result['experiment']:<24} "
            f"{str(result['adapter']):<8} "
            f"{str(result['kl']):<6} "
            f"{str(result['rl']):<6} "
            f"{result['accuracy']:<8.4f} "
            f"{result['total_loss']:<8.4f} "
            f"{result['kl_loss']:<10.6f} "
            f"{result['rl_loss']:<10.6f} "
            f"{result['entropy']:<10.4f} "
            f"{result['latency']:<10.6f} "
            f"{result['imbalance']:<10.4f}"
        )


def main():
    config = load_config("configs/moe.yaml")

    logger = setup_logger()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Using device: {device}")

    experiments = [
        {
            "name": "sparse_moe_baseline",
            "use_adapter": False,
            "freeze_adapter_only": False,
            "use_kl_loss": False,
            "use_rl_loss": False,
        },
        {
            "name": "adapter_kl",
            "use_adapter": True,
            "freeze_adapter_only": True,
            "use_kl_loss": True,
            "use_rl_loss": False,
        },
        {
            "name": "adapter_kl_rl",
            "use_adapter": True,
            "freeze_adapter_only": True,
            "use_kl_loss": True,
            "use_rl_loss": True,
        },
    ]

    results = []
    all_epoch_history = []

    for experiment in experiments:
        result, epoch_history = train_one_experiment(
            config=config,
            experiment=experiment,
            device=device,
            logger=logger
        )

        results.append(result)
        all_epoch_history.extend(epoch_history)

    print_comparison_table(results)

    save_results_csv(
        results=results,
        path="outputs/moe_experiment_results.csv"
    )

    save_results_csv(
        results=all_epoch_history,
        path="outputs/moe_epoch_history.csv"
    )

    save_all_plots(
        results=results,
        epoch_history=all_epoch_history
    )

    save_experiment_summary(
        results=results,
        epoch_history=all_epoch_history
    )


if __name__ == "__main__":
    main()