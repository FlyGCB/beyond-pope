"""
Parse and filter Visual Genome annotations for X-POPE construction.

Downloads needed (put in data/raw/visual_genome/):
  attributes.json    - from http://visualgenome.org/static/data/dataset/attributes.json.zip
  relationships.json - from http://visualgenome.org/static/data/dataset/relationships.json.zip
  image_data.json    - from http://visualgenome.org/static/data/dataset/image_data.json.zip

Usage:
    python -m src.dataset.parse_vg \
        --vg-dir data/raw/visual_genome \
        --coco-dir data/raw/coco \
        --output-dir data/processed/vg_parsed
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict


# ── Filter config ─────────────────────────────────────────────────────────────

VALID_ATTRIBUTE_TYPES = {
    "color": {
        "red", "blue", "green", "yellow", "white", "black", "brown",
        "orange", "purple", "pink", "gray", "grey", "silver", "gold",
        "beige", "tan", "navy",
    },
    "brightness": {
        "dark", "light", "bright", "dim",
    },
    "material": {
        "wooden", "metal", "metallic", "plastic", "glass", "concrete",
        "stone", "brick", "leather", "fabric", "cloth", "paper",
        "rubber", "steel", "iron", "wood",
    },
    "size": {
        "large", "small", "big", "tiny", "huge", "tall", "short",
        "wide", "narrow", "long", "little",
    },
    "shape": {
        "round", "circular", "square", "rectangular", "oval",
        "triangular", "flat", "curved", "straight",
    },
}

ALL_VALID_ATTRS = {a for attrs in VALID_ATTRIBUTE_TYPES.values() for a in attrs}

SUBJECTIVE_ATTRS = {
    "beautiful", "pretty", "nice", "good", "bad", "old", "new", "young",
    "clean", "dirty", "open", "closed", "empty", "full",
    "happy", "sad", "calm", "busy", "simple", "complex", "natural",
}

COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
}

# Extended mapping: common VG natural-language words → COCO class
# This dramatically increases relation coverage since VG uses natural language
EXTENDED_CLASS_MAP = {
    # person variants
    "man": "person", "woman": "person", "boy": "person", "girl": "person",
    "child": "person", "kid": "person", "baby": "person", "people": "person",
    "player": "person", "rider": "person", "skater": "person", "surfer": "person",
    "biker": "person", "cyclist": "person", "pedestrian": "person",
    "guy": "person", "lady": "person", "male": "person", "female": "person",
    "skier": "person", "snowboarder": "person", "swimmer": "person",
    "hiker": "person", "runner": "person", "walker": "person",
    # vehicle variants
    "vehicle": "car", "automobile": "car", "sedan": "car", "suv": "car",
    "van": "car", "pickup": "truck", "lorry": "truck",
    "motorbike": "motorcycle", "moped": "motorcycle", "scooter": "motorcycle",
    "jet": "airplane", "plane": "airplane", "aircraft": "airplane",
    "vessel": "boat", "sailboat": "boat", "canoe": "boat", "kayak": "boat",
    # animal variants
    "dogs": "dog", "puppy": "dog", "cats": "cat", "kitten": "cat",
    "horses": "horse", "cows": "cow", "birds": "bird",
    "elephants": "elephant", "bears": "bear", "zebras": "zebra",
    "giraffes": "giraffe",
    # furniture / objects
    "sofa": "couch", "settee": "couch", "loveseat": "couch",
    "table": "dining table", "desk": "dining table",
    "monitor": "tv", "television": "tv", "screen": "tv",
    "computer": "laptop", "notebook": "laptop",
    "phone": "cell phone", "cellphone": "cell phone", "smartphone": "cell phone",
    "mug": "cup", "glass": "cup",
    "bag": "backpack", "purse": "handbag",
    "plant": "potted plant",
    "fridge": "refrigerator",
}

# Sorted by length descending — longer phrases match before substrings
_COCO_CLASSES_SORTED = sorted(COCO_CLASSES, key=len, reverse=True)

VALID_RELATIONS = {
    # Spatial
    "on", "in", "on top of", "next to", "beside", "near", "above",
    "below", "behind", "in front of", "under", "underneath", "inside",
    "outside", "across from", "along",
    # Functional
    "holding", "wearing", "carrying", "sitting on", "standing on",
    "riding", "eating", "drinking", "using", "playing with",
    "attached to", "hanging from", "leaning on", "lying on",
    # Extended — unambiguous spatial/functional meaning
    "touching", "facing", "looking at", "walking on", "running on",
    "parked on", "mounted on", "flying over", "jumping over",
    "pulling", "pushing", "kicking", "throwing", "catching",
    "adjacent to", "surrounding", "covering",
}


# ── COCO image ID lookup ──────────────────────────────────────────────────────

def load_coco_image_ids(coco_dir: Path) -> set[int]:
    ann_file = coco_dir / "annotations" / "instances_val2014.json"
    if not ann_file.exists():
        print(f"[warn] COCO annotations not found at {ann_file}")
        print("       Skipping COCO filter — all VG images will be used.")
        return set()
    with open(ann_file) as f:
        coco = json.load(f)
    ids = {img["id"] for img in coco["images"]}
    print(f"Loaded {len(ids):,} COCO val2014 image IDs")
    return ids


def load_vg_coco_mapping(vg_dir: Path) -> dict[int, int]:
    image_data_file = vg_dir / "image_data.json"
    if not image_data_file.exists():
        print(f"[warn] image_data.json not found — skipping COCO alignment")
        return {}
    with open(image_data_file) as f:
        image_data = json.load(f)
    mapping = {}
    for img in image_data:
        if img.get("coco_id"):
            mapping[img["image_id"]] = img["coco_id"]
    print(f"VG→COCO mapping: {len(mapping):,} images have COCO IDs")
    return mapping


# ── Attribute parser ──────────────────────────────────────────────────────────

def parse_attributes(
    vg_dir: Path,
    coco_image_ids: set[int],
    vg_coco_map: dict[int, int],
) -> list[dict]:
    attr_file = vg_dir / "attributes.json"
    print(f"Loading {attr_file} ...")
    with open(attr_file) as f:
        raw = json.load(f)

    records = []
    skipped = defaultdict(int)

    for image in raw:
        vg_id = image["image_id"]
        coco_id = vg_coco_map.get(vg_id, -1)

        if coco_image_ids and coco_id not in coco_image_ids:
            skipped["not_in_coco"] += 1
            continue

        for obj in image.get("attributes", []):
            obj_name = (obj.get("name") or (obj.get("names") or [""])[0]).lower().strip()
            coco_class = _match_coco_class(obj_name)
            if coco_class is None:
                skipped["no_coco_class"] += 1
                continue

            attrs = obj.get("attributes", [])
            if not attrs:
                skipped["no_attributes"] += 1
                continue

            try:
                bbox = [obj["x"], obj["y"], obj["w"], obj["h"]]
            except KeyError:
                bbox = None

            for attr in attrs:
                attr = attr.lower().strip()
                if attr in SUBJECTIVE_ATTRS:
                    skipped["subjective"] += 1
                    continue
                attr_type = _get_attribute_type(attr)
                if attr_type is None:
                    skipped["invalid_attr"] += 1
                    continue

                records.append({
                    "vg_image_id":    vg_id,
                    "coco_image_id":  coco_id,
                    "object_name":    coco_class,
                    "attribute":      attr,
                    "attribute_type": attr_type,
                    "object_id":      obj["object_id"],
                    "bbox":           bbox,
                })

    print(f"Parsed {len(records):,} valid attribute records")
    print(f"Skipped: { {k: v for k, v in skipped.items()} }")
    return records


# ── Relation parser ───────────────────────────────────────────────────────────

def parse_relations(
    vg_dir: Path,
    coco_image_ids: set[int],
    vg_coco_map: dict[int, int],
) -> list[dict]:
    rel_file = vg_dir / "relationships.json"
    print(f"Loading {rel_file} ...")
    with open(rel_file) as f:
        raw = json.load(f)

    records = []
    skipped = defaultdict(int)

    for image in raw:
        vg_id = image["image_id"]
        coco_id = vg_coco_map.get(vg_id, -1)

        if coco_image_ids and coco_id not in coco_image_ids:
            skipped["not_in_coco"] += 1
            continue

        for rel in image.get("relationships", []):
            predicate = rel.get("predicate", "").lower().strip()
            if predicate not in VALID_RELATIONS:
                skipped["invalid_relation"] += 1
                continue

            subj = rel["subject"]
            obj  = rel["object"]

            subj_name = (subj.get("name") or (subj.get("names") or [""])[0]).lower().strip()
            obj_name  = (obj.get("name")  or (obj.get("names")  or [""])[0]).lower().strip()

            subj_coco = _match_coco_class(subj_name)
            obj_coco  = _match_coco_class(obj_name)

            if subj_coco is None or obj_coco is None:
                skipped["no_coco_class"] += 1
                continue

            if subj["object_id"] == obj["object_id"]:
                skipped["self_relation"] += 1
                continue

            records.append({
                "vg_image_id":     vg_id,
                "coco_image_id":   coco_id,
                "subject_name":    subj_coco,
                "object_name":     obj_coco,
                "relation":        predicate,
                "subject_id":      subj["object_id"],
                "object_id":       obj["object_id"],
                "relationship_id": rel["relationship_id"],
            })

    print(f"Parsed {len(records):,} valid relation records")
    print(f"Skipped: { {k: v for k, v in skipped.items()} }")
    return records


# ── Helper functions ──────────────────────────────────────────────────────────

def _match_coco_class(name: str) -> str | None:
    """
    Match a VG object name to a COCO class.

    Priority:
    1. Direct match in COCO_CLASSES
    2. Extended mapping (e.g. "man" → "person")
    3. Word-boundary regex against COCO class names (longest first)
    4. Extended map words inside name
    """
    name = name.lower().strip()
    if not name:
        return None

    if name in COCO_CLASSES:
        return name

    if name in EXTENDED_CLASS_MAP:
        return EXTENDED_CLASS_MAP[name]

    for coco_class in _COCO_CLASSES_SORTED:
        pattern = r'\b' + re.escape(coco_class) + r'\b'
        if re.search(pattern, name):
            return coco_class

    for ext_word, coco_class in EXTENDED_CLASS_MAP.items():
        pattern = r'\b' + re.escape(ext_word) + r'\b'
        if re.search(pattern, name):
            return coco_class

    return None


def _get_attribute_type(attr: str) -> str | None:
    for attr_type, attr_set in VALID_ATTRIBUTE_TYPES.items():
        if attr in attr_set:
            return attr_type
    return None


# ── Statistics ────────────────────────────────────────────────────────────────

def print_attribute_stats(records: list[dict]):
    type_counts   = defaultdict(int)
    object_counts = defaultdict(int)
    attr_counts   = defaultdict(int)

    for r in records:
        type_counts[r["attribute_type"]] += 1
        object_counts[r["object_name"]] += 1
        attr_counts[r["attribute"]] += 1

    print("\n── Attribute type distribution ──")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:<12} {c:>6,}")

    print("\n── Top 15 objects ──")
    for obj, c in sorted(object_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {obj:<20} {c:>6,}")

    print("\n── Top 20 attributes ──")
    for attr, c in sorted(attr_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {attr:<20} {c:>6,}")


def print_relation_stats(records: list[dict]):
    rel_counts  = defaultdict(int)
    subj_counts = defaultdict(int)
    obj_counts  = defaultdict(int)

    for r in records:
        rel_counts[r["relation"]] += 1
        subj_counts[r["subject_name"]] += 1
        obj_counts[r["object_name"]] += 1

    print("\n── Top 20 relations ──")
    for rel, c in sorted(rel_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {rel:<25} {c:>6,}")

    print("\n── Top 10 subjects ──")
    for s, c in sorted(subj_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {s:<20} {c:>6,}")

    print("\n── Top 10 objects ──")
    for o, c in sorted(obj_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {o:<20} {c:>6,}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parse Visual Genome for X-POPE")
    parser.add_argument("--vg-dir",     default="data/raw/visual_genome")
    parser.add_argument("--coco-dir",   default="data/raw/coco")
    parser.add_argument("--output-dir", default="data/processed/vg_parsed")
    args = parser.parse_args()

    vg_dir     = Path(args.vg_dir)
    coco_dir   = Path(args.coco_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coco_ids    = load_coco_image_ids(coco_dir)
    vg_coco_map = load_vg_coco_mapping(vg_dir)

    print("\n=== Parsing attributes ===")
    attr_records = parse_attributes(vg_dir, coco_ids, vg_coco_map)
    print_attribute_stats(attr_records)
    attr_out = output_dir / "vg_attributes_filtered.json"
    with open(attr_out, "w") as f:
        json.dump(attr_records, f, separators=(',', ':'))
    print(f"\nSaved {len(attr_records):,} attribute records → {attr_out}")

    print("\n=== Parsing relations ===")
    rel_records = parse_relations(vg_dir, coco_ids, vg_coco_map)
    print_relation_stats(rel_records)
    rel_out = output_dir / "vg_relations_filtered.json"
    with open(rel_out, "w") as f:
        json.dump(rel_records, f, separators=(',', ':'))
    print(f"\nSaved {len(rel_records):,} relation records → {rel_out}")

    print("\nDone. Next step: run src/dataset/build_xpope.py")


if __name__ == "__main__":
    main()
