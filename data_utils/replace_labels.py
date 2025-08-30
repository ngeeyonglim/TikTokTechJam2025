import os
from pathlib import Path

# Set your directory path here
image_dir = Path("./data/labels")

# Loop through all .txt files
for txt_file in image_dir.glob("*.txt"):
    new_lines = []
    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            parts[0] = "80"  # replace first label
            new_lines.append(" ".join(parts))

    # Overwrite file with updated lines
    with open(txt_file, "w") as f:
        f.write("\n".join(new_lines) + "\n")

    print(f"Updated: {txt_file}")
