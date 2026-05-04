#!/usr/bin/env python
"""Write a curated inventory of public QEC datasets for P10 evidence planning."""

from __future__ import annotations

import csv
from pathlib import Path


FIELDS = [
    "priority",
    "dataset_id",
    "title",
    "provider",
    "hardware_or_platform",
    "source_url",
    "doi",
    "file_name",
    "file_size_gb",
    "expected_size_bytes",
    "checksum_md5",
    "download_url",
    "local_path",
    "local_status",
    "failureops_compatibility",
    "recommended_use",
    "caveat",
]


def gb(num_bytes: int) -> str:
    return f"{num_bytes / 1_000_000_000:.6f}"


DATASETS = [
    {
        "priority": 1,
        "dataset_id": "google_rl_qec_v2_2026",
        "title": 'Data for "Reinforcement Learning Control of Quantum Error Correction"',
        "provider": "Google Quantum AI",
        "hardware_or_platform": "Willow superconducting processor",
        "source_url": "https://zenodo.org/records/18896801",
        "doi": "10.5281/zenodo.18896801",
        "file_name": "google_reinforcement_learning_qec.zip",
        "expected_size_bytes": 7786791716,
        "file_size_gb": gb(7786791716),
        "checksum_md5": "ca54323082fcd0e3671d5b90ce45d85c",
        "download_url": "https://zenodo.org/api/records/18896801/files/google_reinforcement_learning_qec.zip/content",
        "local_path": "data/raw/google_rl_qec_v2/google_reinforcement_learning_qec.zip",
        "failureops_compatibility": "high: public real QEC records; likely closest to existing Google RL QEC adapter",
        "recommended_use": "First new dataset to inspect; test whether current P7 adapter can generalize to v2 surface/color-code layout.",
        "caveat": "Still Google-family evidence; large enough to inspect schema before full analysis.",
    },
    {
        "priority": 2,
        "dataset_id": "google_below_threshold_105q_surface_2024",
        "title": 'Data for "Quantum error correction below the surface code threshold"',
        "provider": "Google Quantum AI",
        "hardware_or_platform": "Willow superconducting processor",
        "source_url": "https://zenodo.org/records/13273331",
        "doi": "10.5281/zenodo.13273331",
        "file_name": "google_105Q_surface_code_d3_d5_d7.zip",
        "expected_size_bytes": 5716907033,
        "file_size_gb": gb(5716907033),
        "checksum_md5": "21fa6ad35b395d838ebcdbc92e364a12",
        "download_url": "https://zenodo.org/api/records/13273331/files/google_105Q_surface_code_d3_d5_d7.zip/content",
        "local_path": "data/raw/google_below_threshold/google_105Q_surface_code_d3_d5_d7.zip",
        "failureops_compatibility": "high if archive contains shot-level detection events and decoder predictions",
        "recommended_use": "Strengthen surface-code real-record evidence with d3/d5/d7 scaling data without downloading the 112.5 GB full record.",
        "caveat": "Still Google-family evidence; adapter work likely needed.",
    },
    {
        "priority": 3,
        "dataset_id": "google_decoder_prior_surface_2024",
        "title": 'Data for "Optimization of decoder priors for accurate quantum error correction"',
        "provider": "Google Quantum AI",
        "hardware_or_platform": "Sycamore superconducting processor",
        "source_url": "https://zenodo.org/records/11403595",
        "doi": "10.5281/zenodo.11403595",
        "file_name": "google_sycamore_surface_code_d3_d5.zip",
        "expected_size_bytes": 6216793617,
        "file_size_gb": gb(6216793617),
        "checksum_md5": "c1205ecff28c51f8cc74c2de5ee73261",
        "download_url": "https://zenodo.org/api/records/11403595/files/google_sycamore_surface_code_d3_d5.zip/content",
        "local_path": "data/raw/google_decoder_priors/google_sycamore_surface_code_d3_d5.zip",
        "failureops_compatibility": "high if priors/DEM variants and shot-level outcomes are exposed",
        "recommended_use": "Best candidate for broadening the current P7.5 decoder-prior intervention beyond one condition.",
        "caveat": "Still Google-family evidence; need schema inspection before claiming paired prior interventions.",
    },
    {
        "priority": 4,
        "dataset_id": "google_dynamic_surface_codes_2024",
        "title": 'Data for "Demonstrating dynamic surface codes"',
        "provider": "Google Quantum AI",
        "hardware_or_platform": "Willow superconducting processor",
        "source_url": "https://zenodo.org/records/14238907",
        "doi": "10.5281/zenodo.14238907",
        "file_name": "google_dynamic_circuits_d3_d5.zip",
        "expected_size_bytes": 2081742123,
        "file_size_gb": gb(2081742123),
        "checksum_md5": "3981b6f8ca8e093c705684ff7c9f6697",
        "download_url": "https://zenodo.org/api/records/14238907/files/google_dynamic_circuits_d3_d5.zip/content",
        "local_path": "data/raw/google_dynamic_surface_codes/google_dynamic_circuits_d3_d5.zip",
        "failureops_compatibility": "medium-high: public real surface-code memory experiments, schema unknown locally",
        "recommended_use": "Potential robustness extension for surface-code memory data with different experiment semantics.",
        "caveat": "Dynamic-code semantics may require a narrower FailureOps interpretation.",
    },
    {
        "priority": 5,
        "dataset_id": "google_color_code_2024",
        "title": 'Data for "Scaling and logic in the color code on a superconducting quantum processor"',
        "provider": "Google Quantum AI",
        "hardware_or_platform": "Willow superconducting processor",
        "source_url": "https://zenodo.org/records/14238944",
        "doi": "10.5281/zenodo.14238944",
        "file_name": "google_superdense_color_code_d3_d5.zip",
        "expected_size_bytes": 738541607,
        "file_size_gb": gb(738541607),
        "checksum_md5": "a17f410331ddddcf8b786cd9da2e456d",
        "download_url": "https://zenodo.org/api/records/14238944/files/google_superdense_color_code_d3_d5.zip/content",
        "local_path": "data/raw/google_color_code/google_superdense_color_code_d3_d5.zip",
        "failureops_compatibility": "medium: likely public real QEC records, but not the same code family as current surface-code adapter",
        "recommended_use": "Smallest new real-record archive; useful for testing whether FailureOps generalizes beyond surface code.",
        "caveat": "Adding color-code support could dilute P10 unless framed as optional robustness.",
    },
    {
        "priority": 6,
        "dataset_id": "google_qec3v5_2022",
        "title": 'Data for "Suppressing quantum errors by scaling a surface code logical qubit"',
        "provider": "Google Quantum AI Team",
        "hardware_or_platform": "Google superconducting processor",
        "source_url": "https://zenodo.org/records/6804040",
        "doi": "10.5281/zenodo.6804040",
        "file_name": "google_qec3v5_experiment_data.zip",
        "expected_size_bytes": 315490804,
        "file_size_gb": gb(315490804),
        "checksum_md5": "a7fd8b481c3087090093106382dc217d",
        "download_url": "https://zenodo.org/api/records/6804040/files/google_qec3v5_experiment_data.zip/content",
        "local_path": "data/raw/google_qec3v5/google_qec3v5_experiment_data.zip",
        "failureops_compatibility": "high: already used by P10 qec3v5 external replication",
        "recommended_use": "Expand current qec3v5 sweep over more distances/centers/decoder pathways before adding a new adapter.",
        "caveat": "Already part of P10; extra runs have lower marginal novelty than new records.",
    },
    {
        "priority": 7,
        "dataset_id": "daqec_ibm_2025",
        "title": "DAQEC-Benchmark: Drift-Aware Quantum Error Correction Dataset with IBM Hardware Validation",
        "provider": "Independent Researcher",
        "hardware_or_platform": "IBM Quantum backends",
        "source_url": "https://zenodo.org/records/17881116",
        "doi": "10.5281/zenodo.17881116",
        "file_name": "master.parquet",
        "expected_size_bytes": 109439,
        "file_size_gb": gb(109439),
        "checksum_md5": "850582441564640ec5c191ba4b33bdfe",
        "download_url": "https://zenodo.org/api/records/17881116/files/master.parquet/content",
        "local_path": "data/raw/daqec_ibm/master.parquet",
        "failureops_compatibility": "low-medium: independent hardware evidence, but aggregate/session-level rather than shot-level detector records",
        "recommended_use": "Keep as independent boundary/sanity evidence, not as primary FailureOps paired attribution.",
        "caveat": "Does not replace shot-level public detector records.",
    },
    {
        "priority": 8,
        "dataset_id": "multiplatform_daqec_2025",
        "title": "Multi-Platform Cross-Validation of Interaction Effect of Drift-Aware Quantum Error Correction",
        "provider": "Independent Researcher",
        "hardware_or_platform": "IBM, IQM, IonQ, Rigetti",
        "source_url": "https://zenodo.org/records/18087905",
        "doi": "10.5281/zenodo.18087905",
        "file_name": "MULTI_PLATFORM_CROSS_VALIDATION_README.md",
        "expected_size_bytes": 8196,
        "file_size_gb": gb(8196),
        "checksum_md5": "bbbf8f5e9012d7718b31d3048c1ac733",
        "download_url": "https://zenodo.org/api/records/18087905/files/MULTI_PLATFORM_CROSS_VALIDATION_README.md/content",
        "local_path": "data/raw/multiplatform_daqec/MULTI_PLATFORM_CROSS_VALIDATION_README.md",
        "failureops_compatibility": "low: hardware summaries and task identifiers, not shot-level QEC detector records",
        "recommended_use": "Cite only as broader hardware-boundary context if inspected; avoid core attribution claims.",
        "caveat": "Very small summary files; not a replacement for raw syndrome/detector records.",
    },
]


def main() -> None:
    output = Path("data/results/p10_public_qec_data_candidates.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in DATASETS:
        enriched = dict(row)
        local_path = Path(enriched["local_path"])
        if not local_path.exists():
            enriched["local_status"] = "missing"
        elif local_path.stat().st_size == int(enriched["expected_size_bytes"]):
            enriched["local_status"] = "complete"
        else:
            enriched["local_status"] = f"partial:{local_path.stat().st_size}"
        rows.append(enriched)

    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} public QEC data candidate rows to {output}")


if __name__ == "__main__":
    main()
