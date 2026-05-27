from pathlib import Path
import time
import random
import requests
import pandas as pd
from Bio import SeqIO
from io import StringIO
from Bio.Seq import UndefinedSequenceError


# =========================
# 1. 路径设置
# =========================
BASE_DIR = Path("/home/ziyan/enzyme_pipeline/data/steps")

KNOWN_FASTA_NAMES = [
    "known_proteins.fasta",
    "known_proteins.fa",
]

OUT_NAME = "known_cds_from_uniprot.fasta"
MAPPING_NAME = "known_cds_mapping.csv"
FAILED_NAME = "known_cds_failed.csv"


# =========================
# 2. API
# =========================
UNIPROT_JSON_API = "https://rest.uniprot.org/uniprotkb/{acc}.json"
ENA_EMBL_API = "https://www.ebi.ac.uk/ena/browser/api/embl/{nuc_acc}"


def read_known_headers(fasta_file):
    records = []

    with open(fasta_file) as f:
        for line in f:
            line = line.strip()

            if not line.startswith(">"):
                continue

            header = line[1:]
            parts = header.split("|")

            if len(parts) < 4:
                print(f"[WARN] header 格式异常，跳过：{header}")
                continue

            if parts[0] != "Known":
                continue

            records.append({
                "original_header": header,
                "uniprot_acc": parts[1],
                "species": parts[2],
                "ec_numbers": "|".join(parts[3:]),
            })

    return records


def get_uniprot_nucleotide_xrefs(uniprot_acc):
    url = UNIPROT_JSON_API.format(acc=uniprot_acc)

    try:
        r = requests.get(url, timeout=60)
    except Exception as e:
        print(f"  [WARN] UniProt 请求失败：{e}")
        return []

    if r.status_code != 200:
        print(f"  [WARN] UniProt HTTP 状态码：{r.status_code}")
        return []

    try:
        data = r.json()
    except Exception as e:
        print(f"  [WARN] UniProt JSON 解析失败：{e}")
        return []

    xrefs = data.get("uniProtKBCrossReferences", [])
    results = []

    for x in xrefs:
        db = x.get("database")
        nuc_acc = x.get("id")

        if db not in ["EMBL", "GenBank", "DDBJ", "RefSeq"]:
            continue

        props = {}
        for p in x.get("properties", []):
            key = p.get("key")
            value = p.get("value")
            if key and value:
                props[key] = value

        protein_id = (
            props.get("ProteinId")
            or props.get("Protein ID")
            or props.get("protein_id")
            or ""
        )

        molecule_type = (
            props.get("MoleculeType")
            or props.get("Molecule type")
            or ""
        )

        results.append({
            "database": db,
            "nucleotide_accession": nuc_acc,
            "protein_id": protein_id,
            "molecule_type": molecule_type,
        })

    return results


def fetch_embl_record(nuc_acc):
    url = ENA_EMBL_API.format(nuc_acc=nuc_acc)

    try:
        r = requests.get(url, timeout=90)
    except Exception as e:
        print(f"  [WARN] ENA 请求失败：{e}")
        return None

    if r.status_code != 200:
        print(f"  [WARN] ENA HTTP 状态码：{r.status_code}")
        return None

    text = r.text.strip()

    if not text.startswith("ID"):
        print("  [WARN] ENA 返回内容不是 EMBL 格式")
        return None

    return text


def safe_extract_feature_seq(feature, record):
    try:
        cds_seq = feature.extract(record.seq)
        cds_seq = str(cds_seq)

        if not cds_seq:
            return None, "EMPTY_CDS_SEQUENCE"

        seq_upper = cds_seq.upper()

        if set(seq_upper) <= {"N"}:
            return None, "CDS_ONLY_N"

        return cds_seq, "OK"

    except UndefinedSequenceError:
        return None, "UNDEFINED_SEQUENCE_IN_EMBL_RECORD"

    except Exception as e:
        return None, f"FEATURE_EXTRACT_FAILED: {e}"


def extract_cds_from_embl(embl_text, target_protein_id=None):
    handle = StringIO(embl_text)

    try:
        record = SeqIO.read(handle, "embl")
    except Exception as e:
        return None, f"EMBL_PARSE_FAILED: {e}"

    cds_features = [
        f for f in record.features
        if f.type == "CDS"
    ]

    if not cds_features:
        return None, "NO_CDS_FEATURE"

    if target_protein_id:
        for feature in cds_features:
            protein_ids = feature.qualifiers.get("protein_id", [])

            if target_protein_id in protein_ids:
                cds_seq, status = safe_extract_feature_seq(feature, record)

                if cds_seq:
                    return cds_seq, "MATCH_BY_PROTEIN_ID"

                return None, status

    if len(cds_features) == 1:
        cds_seq, status = safe_extract_feature_seq(cds_features[0], record)

        if cds_seq:
            return cds_seq, "SINGLE_CDS_USED"

        return None, status

    return None, "MULTIPLE_CDS_NO_MATCH"


def get_cds_for_uniprot(uniprot_acc):
    xrefs = get_uniprot_nucleotide_xrefs(uniprot_acc)

    if not xrefs:
        return None, {
            "status": "FAILED",
            "reason": "NO_NUCLEOTIDE_XREF",
            "nucleotide_database": "",
            "nucleotide_accession": "",
            "protein_id": "",
            "molecule_type": "",
        }

    last_reason = "ALL_XREF_FAILED"

    for x in xrefs:
        db = x["database"]
        nuc_acc = x["nucleotide_accession"]
        protein_id = x["protein_id"]
        molecule_type = x["molecule_type"]

        print(f"  尝试：{uniprot_acc} -> {db}:{nuc_acc}, protein_id={protein_id}")

        embl_text = fetch_embl_record(nuc_acc)

        if embl_text is None:
            last_reason = "EMBL_RECORD_FETCH_FAILED"
            time.sleep(random.uniform(1.5, 4.0))
            continue

        cds_seq, reason = extract_cds_from_embl(
            embl_text=embl_text,
            target_protein_id=protein_id
        )

        if cds_seq:
            return cds_seq, {
                "status": "OK",
                "reason": reason,
                "nucleotide_database": db,
                "nucleotide_accession": nuc_acc,
                "protein_id": protein_id,
                "molecule_type": molecule_type,
            }

        print(f"  [WARN] CDS 提取失败：{reason}")
        last_reason = reason
        time.sleep(random.uniform(1.5, 4.0))

    return None, {
        "status": "FAILED",
        "reason": last_reason,
        "nucleotide_database": "",
        "nucleotide_accession": "",
        "protein_id": "",
        "molecule_type": "",
    }


def write_fasta_record(handle, header, seq, width=60):
    handle.write(f">{header}\n")
    for i in range(0, len(seq), width):
        handle.write(seq[i:i + width] + "\n")


def main():
    step_dirs = [
        p for p in BASE_DIR.iterdir()
        if p.is_dir() and p.name.startswith("step")
    ]

    print(f"检测到 step 文件夹数量：{len(step_dirs)}")

    for step_dir in sorted(step_dirs):
        print("\n" + "=" * 80)
        print(f"处理 step：{step_dir.name}")

        known_fasta = None

        for name in KNOWN_FASTA_NAMES:
            candidate = step_dir / name
            if candidate.exists():
                known_fasta = candidate
                break

        if known_fasta is None:
            print(f"[SKIP] 没有找到 known fasta：{step_dir}")
            continue

        known_records = read_known_headers(known_fasta)

        if not known_records:
            print(f"[SKIP] 没有解析到 Known header：{known_fasta}")
            continue

        out_fasta = step_dir / OUT_NAME
        mapping_csv = step_dir / MAPPING_NAME
        failed_csv = step_dir / FAILED_NAME

        mapping_records = []
        failed_records = []

        with open(out_fasta, "w") as fout:
            for i, rec in enumerate(known_records, start=1):
                uniprot_acc = rec["uniprot_acc"]
                species = rec["species"]
                ec_numbers = rec["ec_numbers"]

                print(f"\n[{i}/{len(known_records)}] {uniprot_acc}")

                try:
                    cds_seq, info = get_cds_for_uniprot(uniprot_acc)

                except Exception as e:
                    cds_seq = None
                    info = {
                        "status": "FAILED",
                        "reason": f"UNEXPECTED_ERROR: {e}",
                        "nucleotide_database": "",
                        "nucleotide_accession": "",
                        "protein_id": "",
                        "molecule_type": "",
                    }

                row = {
                    "step": step_dir.name,
                    "original_header": rec["original_header"],
                    "uniprot_acc": uniprot_acc,
                    "species": species,
                    "ec_numbers": ec_numbers,
                    **info,
                }

                if cds_seq:
                    nuc_acc = info["nucleotide_accession"]
                    protein_id = info["protein_id"]

                    new_header = (
                        f"Known|{uniprot_acc}|{species}|{ec_numbers}"
                        f"|nuc={nuc_acc}|protein_id={protein_id}"
                    )

                    write_fasta_record(fout, new_header, cds_seq)

                    mapping_records.append(row)
                    print(f"  [OK] CDS 长度：{len(cds_seq)}")

                else:
                    failed_records.append(row)
                    print(f"  [FAILED] {info['reason']}")

                time.sleep(random.uniform(3, 8))

        pd.DataFrame(mapping_records).to_csv(mapping_csv, index=False)
        pd.DataFrame(failed_records).to_csv(failed_csv, index=False)

        print(f"\n输出 CDS fasta：{out_fasta}")
        print(f"输出 mapping 表：{mapping_csv}")
        print(f"输出 failed 表：{failed_csv}")


if __name__ == "__main__":
    main()