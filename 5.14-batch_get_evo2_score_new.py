import torch
import pandas as pd
import numpy as np
from pathlib import Path
from evo2 import Evo2


# ==========================
# 路径设置
# ==========================
STEPS_DIR = Path("/home/ziyan/enzyme_pipeline/data/steps")

# 所有 step 的 Evo2 结果统一输出到这里
EVO2_OUTPUT_DIR = Path("/home/ziyan/enzyme_pipeline/data/evo2_new")

# 汇总文件
SUMMARY_FILE = EVO2_OUTPUT_DIR / "all_steps_evo2_summary1.csv"

MODEL_NAME = "evo2_7b"
DEVICE = "cuda:0"


# ==========================
# 基础函数
# ==========================
def read_fasta(fasta_file):
    records = []
    header = None
    seq_parts = []

    with open(fasta_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_parts).upper()))

                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)

        if header is not None:
            records.append((header, "".join(seq_parts).upper()))

    return records


def write_fasta(records, output_file):
    with open(output_file, "w") as f:
        for header, seq in records:
            f.write(f">{header}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")


def clean_dna(seq):
    seq = seq.upper().replace("U", "T")
    valid = set("ACGTN")
    return "".join([x for x in seq if x in valid])


def get_seq_id(header):
    parts = header.split("|")

    if parts[0].lower() == "known" and len(parts) >= 2:
        return parts[1]

    return parts[0]


def is_known(header):
    return "known" in header.lower()


def evo2_nll_per_base(model, sequence):
    sequence = clean_dna(sequence)

    if len(sequence) < 10:
        return np.nan

    input_ids = torch.tensor(
        model.tokenizer.tokenize(sequence),
        dtype=torch.long
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs, _ = model(input_ids)
        logits = outputs[0]

        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = input_ids[:, 1:].contiguous().long()

        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean"
        )

    return float(loss.detach().cpu())


# ==========================
# Evo2 分类函数
# ==========================
def classify_evo2(candidate_nll, known_mean, known_std):
    """
    Z ≤ 1        -> Evo2-supported
    1 < Z ≤ 2    -> Evo2-borderline
    Z > 2        -> Evo2-weak
    """
    if np.isnan(candidate_nll):
        return "Evo2-invalid", np.nan

    if known_std == 0:
        z_score = 0 if candidate_nll <= known_mean else 999
    else:
        z_score = (candidate_nll - known_mean) / known_std

    if z_score <= 1:
        return "Evo2-supported", z_score
    elif z_score <= 2:
        return "Evo2-borderline", z_score
    else:
        return "Evo2-weak", z_score


# ==========================
# 合并 Known + Predicted CDS
# ==========================
def merge_step_fastas(step_dir):
    step_name = step_dir.name

    clustered_fasta = step_dir / "cds_cluster95.fasta"
    known_fasta = step_dir / "known_cds_from_uniprot.fasta"
    all_fasta = step_dir / "all_sequences.fasta"

    if not clustered_fasta.exists():
        raise FileNotFoundError(f"缺少文件: {clustered_fasta}")

    if not known_fasta.exists():
        raise FileNotFoundError(f"缺少文件: {known_fasta}")

    known_records = read_fasta(known_fasta)
    predicted_records = read_fasta(clustered_fasta)

    merged_records = known_records + predicted_records

    write_fasta(merged_records, all_fasta)

    return all_fasta


# ==========================
# 单个 step 打分
# ==========================
def score_one_step(model, step_dir):
    step_name = step_dir.name

    print("\n" + "=" * 80)
    print(f"Processing {step_name}")
    print("=" * 80)

    all_fasta = merge_step_fastas(step_dir)
    records = read_fasta(all_fasta)

    known_records = []
    predicted_records = []

    for header, seq in records:
        if is_known(header):
            known_records.append((header, seq))
        else:
            predicted_records.append((header, seq))

    print(f"Known sequence count: {len(known_records)}")
    print(f"Predicted sequence count: {len(predicted_records)}")

    if len(known_records) == 0:
        raise ValueError(
            f"{step_name} 没有识别到 Known 序列，请检查 header 中是否包含 Known。"
        )

    if len(predicted_records) == 0:
        raise ValueError(f"{step_name} 没有识别到 predicted 序列。")

    # --------------------------
    # 1. 计算 Known NLL
    # --------------------------
    known_results = []

    print(f"\nScoring Known sequences for {step_name}...")
    for header, seq in known_records:
        cleaned_seq = clean_dna(seq)
        nll = evo2_nll_per_base(model, cleaned_seq)

        known_results.append({
            "step": step_name,
            "candidate_id": get_seq_id(header),
            "header": header,
            "sequence_type": "Known",
            "length": len(cleaned_seq),
            "evo2_nll_per_base": nll,
        })

        print("Known:", header, "NLL:", nll)

    known_nlls = np.array([
        row["evo2_nll_per_base"]
        for row in known_results
        if not np.isnan(row["evo2_nll_per_base"])
    ])

    if len(known_nlls) == 0:
        raise ValueError(
            f"{step_name} 的 Known 序列没有有效 NLL，无法建立 Evo2 参考分布。"
        )

    known_mean = float(np.mean(known_nlls))
    known_std = float(np.std(known_nlls))

    print("\nKnown Evo2 distribution:")
    print("Mean NLL:", known_mean)
    print("Std NLL:", known_std)

    # --------------------------
    # 2. 计算 Predicted NLL
    # --------------------------
    predicted_results = []

    print(f"\nScoring all predicted sequences for {step_name}...")
    for header, seq in predicted_records:
        cleaned_seq = clean_dna(seq)
        nll = evo2_nll_per_base(model, cleaned_seq)

        evo2_label, z_score = classify_evo2(
            candidate_nll=nll,
            known_mean=known_mean,
            known_std=known_std
        )

        predicted_results.append({
            "step": step_name,
            "candidate_id": get_seq_id(header),
            "header": header,
            "sequence_type": "Predicted",
            "length": len(cleaned_seq),
            "evo2_nll_per_base": nll,
            "known_mean_nll": known_mean,
            "known_std_nll": known_std,
            "z_score_vs_known": z_score,
            "evo2_label": evo2_label,
        })

        print(
            "Pred:",
            header,
            "NLL:",
            nll,
            "Z:",
            z_score,
            "Label:",
            evo2_label
        )

    # --------------------------
    # 3. 保存结果
    # --------------------------
    result_df = pd.DataFrame(predicted_results)

    result_df["candidate_id"] = (
        result_df["candidate_id"]
        .astype(str)
        .str.split("|")
        .str[0]
    )

    EVO2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    step_output_file = EVO2_OUTPUT_DIR / f"{step_name}_evo2_results.csv"
    result_df.to_csv(step_output_file, index=False)

    print(f"\n{step_name} Evo2 results saved:")
    print(step_output_file)

    return result_df


# ==========================
# 主函数
# ==========================
def main():
    print("Loading Evo2 model...")
    model = Evo2(MODEL_NAME)
    model.model.eval()

    step_dirs = sorted([
        p for p in STEPS_DIR.iterdir()
        if p.is_dir() and p.name.startswith("step")
    ])

    print(f"Found {len(step_dirs)} step folders.")

    all_results = []
    failed_steps = []

    for step_dir in step_dirs:
        try:
            result_df = score_one_step(model, step_dir)
            all_results.append(result_df)

        except Exception as e:
            print(f"\n[ERROR] {step_dir.name} failed:")
            print(e)

            failed_steps.append({
                "step": step_dir.name,
                "error": str(e)
            })

            continue

    EVO2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if all_results:
        summary_df = pd.concat(all_results, ignore_index=True)
        summary_df.to_csv(SUMMARY_FILE, index=False)

        print("\nAll step Evo2 summary saved:")
        print(SUMMARY_FILE)

    if failed_steps:
        failed_file = EVO2_OUTPUT_DIR / "failed_steps.csv"
        pd.DataFrame(failed_steps).to_csv(failed_file, index=False)

        print("\nSome steps failed. Failed step list saved:")
        print(failed_file)

    print("\nBatch Evo2 scoring finished.")


if __name__ == "__main__":
    main()