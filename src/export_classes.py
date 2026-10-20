import json
from pathlib import Path

train_txt = Path("AML Assignment 1/train.txt")
classes = {}

with open(train_txt, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        img_name, label = line.rsplit(" ", 1)
        label_idx = int(label)
        if label_idx not in classes:
            # Format "Black_footed_Albatross_0019_..." -> "Black-footed Albatross"
            species_raw = "_".join(img_name.split("_")[:-2])
            clean_name = species_raw.replace("_", " ")
            classes[label_idx] = clean_name

# Sort in order 0 to 199
sorted_classes = [classes[i] for i in range(len(classes))]
with open("classes.json", "w") as f:
    json.dump(sorted_classes, f, indent=2)

print(f"Exported {len(sorted_classes)} classes to classes.json")