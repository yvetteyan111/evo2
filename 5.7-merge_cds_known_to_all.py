from pathlib import Path


# ==========================
# 路径设置
# ==========================
STEPS_DIR = Path("/home/ziyan/enzyme_pipeline/data/steps")

PRED_CDS_FASTA_NAME = "cds_cluster95.fasta"
KNOWN_FASTA_NAME = "known_proteins.fasta"
OUTPUT_FASTA_NAME = "all_sequences.fasta"

SUMMARY_FILE = STEPS_DIR / "merge_all_sequences_summary.csv"


def read_fasta_text(fasta_file):
    with open(fasta_file, "r") as f:
        text = f.read().strip()
    return text


def count_fasta_records(fasta_file):
    if not fasta_file.exists():
        return 0

    count = 0
    with open(fasta_file, "r") as f:
        for line in f:
            if line.startswith(">"):
                count += 1
    return count


def main():
    step_dirs = sorted([
        p for p in STEPS_DIR.iterdir()
        if p.is_dir() and p.name.startswith("step")
    ])

    print(f"找到 {len(step_dirs)} 个 step 文件夹")

    summary = []

    for step_dir in step_dirs:
        step_name = step_dir.name

        pred_fasta = step_dir / PRED_CDS_FASTA_NAME
        known_fasta = step_dir / KNOWN_FASTA_NAME
        output_fasta = step_dir / OUTPUT_FASTA_NAME

        print("\n" + "=" * 80)
        print(f"处理 {step_name}")
        print("=" * 80)

        if not pred_fasta.exists():
            print(f"[跳过] 缺少文件: {pred_fasta}")
            summary.append([step_name, "missing_cds_cluster95", 0, 0, 0])
            continue

        if not known_fasta.exists():
            print(f"[跳过] 缺少文件: {known_fasta}")
            summary.append([step_name, "missing_known_proteins", 0, 0, 0])
            continue

        pred_text = read_fasta_text(pred_fasta)
        known_text = read_fasta_text(known_fasta)

        with open(output_fasta, "w") as out:
            if known_text:
                out.write(known_text)
                out.write("\n")

            if pred_text:
                out.write(pred_text)
                out.write("\n")

        known_count = count_fasta_records(known_fasta)
        pred_count = count_fasta_records(pred_fasta)
        total_count = count_fasta_records(output_fasta)

        print(f"Known 序列数: {known_count}")
        print(f"Predicted CDS 序列数: {pred_count}")
        print(f"合并后序列数: {total_count}")
        print(f"输出文件: {output_fasta}")

        summary.append([
            step_name,
            "done",
            known_count,
            pred_count,
            total_count
        ])

    with open(SUMMARY_FILE, "w") as f:
        f.write("step,status,known_count,predicted_cds_count,total_count\n")
        for row in summary:
            f.write(",".join(map(str, row)) + "\n")

    print("\n全部完成")
    print("汇总文件:", SUMMARY_FILE)


if __name__ == "__main__":
    main()