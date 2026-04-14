import os
import re
import csv
import argparse

METRIC_KEYS = [
    "Successful requests",
    "Failed requests",
    "Maximum request concurrency",
    "Benchmark duration (s)",
    "Total input tokens",
    "Total generated tokens",
    "Request throughput (req/s)",
    "Output token throughput (tok/s)",
    "Peak output token throughput (tok/s)",
    "Peak concurrent requests",
    "Total token throughput (tok/s)",
    "Mean TTFT (ms)",
    "Median TTFT (ms)",
    "P99 TTFT (ms)",
    "Mean TPOT (ms)",
    "Median TPOT (ms)",
    "P99 TPOT (ms)",
    "Mean ITL (ms)",
    "Median ITL (ms)",
    "P99 ITL (ms)",
]

FILENAME_RE = re.compile(r"bench_c(\d+)_in(\d+)_out(\d+)_rep(\d+)\.log")


def parse_log(log_path):
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    metrics = {}
    for line in content.splitlines():
        line = line.strip()
        for key in METRIC_KEYS:
            if line.startswith(key + ":"):
                value = line.split(":", 1)[1].strip()
                metrics[key] = value
                break
    return metrics


def read_cmd(cmd_path):
    with open(cmd_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def main():
    parser = argparse.ArgumentParser(
        description="Parse vLLM benchmark serve logs and generate CSV."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Path to directory containing log files"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to output CSV file"
    )

    args = parser.parse_args()

    LOG_DIR = args.log_dir
    OUTPUT_CSV = args.output_csv

    rows = []

    for fname in os.listdir(LOG_DIR):
        m = FILENAME_RE.match(fname)
        if not m:
            continue

        concurrency = int(m.group(1))
        input_len = int(m.group(2))
        output_len = int(m.group(3))
        rep = int(m.group(4))

        log_path = os.path.join(LOG_DIR, fname)
        cmd_file = fname.replace(".log", "_cmd.txt")
        cmd_path = os.path.join(LOG_DIR, cmd_file)

        metrics = parse_log(log_path)
        command = read_cmd(cmd_path) if os.path.exists(cmd_path) else ""

        row = {
            "File Name": fname,
            "Concurrency": concurrency,
            "Input Length": input_len,
            "Output Length": output_len,
            "Rep": rep,
            "Command": command,
        }

        for key in METRIC_KEYS:
            row[key] = metrics.get(key, "")

        rows.append(row)

    rows.sort(key=lambda r: (r["Concurrency"], r["Input Length"], r["Output Length"], r["Rep"]))

    fieldnames = [
        "File Name",
        "Concurrency",
        "Input Length",
        "Output Length",
        "Rep",
    ] + METRIC_KEYS + ["Command"]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV written to {OUTPUT_CSV} with {len(rows)} rows.")


if __name__ == "__main__":
    main()
