#!/usr/bin/env python
import json
from pathlib import Path

out = Path("data/toy")
out.mkdir(parents=True, exist_ok=True)
rows = [
    ("A robin is a bird. All birds are animals.", "Is a robin an animal?", "true", 2),
    ("A salmon is a fish. No fish is a mammal.", "Is a salmon a mammal?", "false", 2),
    ("Mira is tall. Tall people may be athletes.", "Is Mira a doctor?", "unknown", 1),
    ("A is B. B is C. C is D. D is E.", "Does A imply E?", "true", 4),
]
options = [
    {"id": "true", "text": "true"}, {"id": "false", "text": "false"}, {"id": "unknown", "text": "unknown"}
]
for split in ["train", "validation"]:
    with (out / f"{split}.jsonl").open("w") as f:
        for _ in range(64 if split == "train" else 16):
            for state, q, label, depth in rows:
                f.write(json.dumps({
                    "state": state, "instruction": q, "type": "choice", "options": options,
                    "label": label, "metadata": {"depth": depth}
                }) + "\n")
print(out)
