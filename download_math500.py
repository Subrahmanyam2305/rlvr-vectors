"""Download and prepare the MATH500 evaluation dataset."""

import json
import urllib.request
import re
from pathlib import Path


def extract_boxed_answer(text):
    """Extract answer from \\boxed{...} with nested brace handling."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    idx += len("\\boxed{")
    depth = 1
    result = []
    while idx < len(text) and depth > 0:
        c = text[idx]
        if c == '{':
            depth += 1
            result.append(c)
        elif c == '}':
            depth -= 1
            if depth > 0:
                result.append(c)
        else:
            result.append(c)
        idx += 1
    return "".join(result).strip()


def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / "math500.json"

    if output_file.exists():
        print(f"Dataset already exists at {output_file}")
        return

    url = "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl"
    tmp_file = data_dir / "math500_raw.jsonl"

    print(f"Downloading MATH500 from {url}...")
    urllib.request.urlretrieve(url, tmp_file)

    problems = []
    with open(tmp_file) as f:
        for line in f:
            item = json.loads(line.strip())
            answer = extract_boxed_answer(item.get("solution", item.get("answer", "")))
            if not answer and "answer" in item:
                answer = item["answer"]
            problems.append({
                "problem": item["problem"],
                "answer": answer,
                "subject": item.get("subject", ""),
                "level": item.get("level", ""),
            })

    with open(output_file, "w") as f:
        json.dump(problems, f, indent=2)

    tmp_file.unlink()
    print(f"Saved {len(problems)} problems to {output_file}")


if __name__ == "__main__":
    main()
