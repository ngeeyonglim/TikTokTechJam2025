#!/usr/bin/env python3
import sys
from pathlib import Path

# --- paths ---
OLD_LABELS = Path("data/valid/labels")   # input dir with .txt YOLO labels
NEW_LABELS = Path("data/new_labels")   # output dir

NEW_LABELS.mkdir(parents=True, exist_ok=True)

# Mapping from Roboflow (old) indices -> your COCO(+face) indices
# (Derived by normalizing name differences: aeroplane->airplane, motorbike->motorcycle,
#  diningtable->dining table, pottedplant->potted plant, sofa->couch, tvmonitor->tv)
old_to_new = {
     0:  4,   # aeroplane → airplane
     1: 47,   # apple
     2: 24,   # backpack
     3: 46,   # banana
     4: 34,   # baseball bat
     5: 35,   # baseball glove
     6: 21,   # bear
     7: 59,   # bed
     8: 13,   # bench
     9:  1,   # bicycle
    10: 14,   # bird
    11:  8,   # boat
    12: 73,   # book
    13: 39,   # bottle
    14: 45,   # bowl
    15: 50,   # broccoli
    16:  5,   # bus
    17: 55,   # cake
    18:  2,   # car
    19: 51,   # carrot
    20: 15,   # cat
    21: 67,   # cell phone
    22: 56,   # chair
    23: 74,   # clock
    24: 19,   # cow
    25: 41,   # cup
    26: 60,   # diningtable → dining table
    27: 16,   # dog
    28: 54,   # donut
    29: 20,   # elephant
    30: 10,   # fire hydrant
    31: 42,   # fork
    32: 29,   # frisbee
    33: 23,   # giraffe
    34: 78,   # hair drier
    35: 26,   # handbag
    36: 17,   # horse
    37: 52,   # hot dog
    38: 66,   # keyboard
    39: 33,   # kite
    40: 43,   # knife
    41: 63,   # laptop
    42: 68,   # microwave
    43:  3,   # motorbike → motorcycle
    44: 64,   # mouse
    45: 49,   # orange
    46: 69,   # oven
    47: 12,   # parking meter
    48:  0,   # person
    49: 53,   # pizza
    50: 58,   # pottedplant → potted plant
    51: 72,   # refrigerator
    52: 65,   # remote
    53: 48,   # sandwich
    54: 76,   # scissors
    55: 18,   # sheep
    56: 71,   # sink
    57: 36,   # skateboard
    58: 30,   # skis
    59: 31,   # snowboard
    60: 57,   # sofa → couch
    61: 44,   # spoon
    62: 32,   # sports ball
    63: 11,   # stop sign
    64: 28,   # suitcase
    65: 37,   # surfboard
    66: 77,   # teddy bear
    67: 38,   # tennis racket
    68: 27,   # tie
    69: 70,   # toaster
    70: 61,   # toilet
    71: 79,   # toothbrush
    72:  9,   # traffic light
    73:  6,   # train
    74:  7,   # truck
    75: 62,   # tvmonitor → tv
    76: 25,   # umbrella
    77: 75,   # vase
    78: 40,   # wine glass
    79: 22    # zebra
}

# Note: There is no "face" in the old set; your final "face" is index 80.

def remap_file(in_path: Path, out_path: Path):
    if not in_path.read_text().strip():
        # empty file -> write empty to keep structure
        out_path.write_text("")
        return

    lines_out = []
    with in_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                old_id = int(parts[0])
            except ValueError:
                # non-standard line; skip safely
                continue
            if old_id not in old_to_new:
                # unseen class -> skip or raise; here we skip the box
                continue
            parts[0] = str(old_to_new[old_id])
            lines_out.append(" ".join(parts))
    out_path.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))

def main():
    srcs = sorted(OLD_LABELS.glob("**/*.txt"))
    if not srcs:
        print(f"No .txt files found under {OLD_LABELS.resolve()}")
        sys.exit(1)

    for src in srcs:
        rel = src.relative_to(OLD_LABELS)
        dst = NEW_LABELS / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        remap_file(src, dst)

    print(f"Done. Remapped {len(srcs)} files into {NEW_LABELS.resolve()}.")

if __name__ == "__main__":
    main()
