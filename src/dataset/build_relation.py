"""
Build the relation split of X-POPE.

Input:  data/processed/vg_parsed/vg_relations_filtered.json
Output: data/processed/xpope_relation.jsonl

Each question:
{
    "question_id": "rel_0001",
    "image": "COCO_val2014_000000XXXXXX.jpg",
    "question": "Is the person sitting on the chair? Please answer Yes or No.",
    "label": "yes" | "no",
    "question_type": "relation",
    "subject_name": "person",
    "object_name": "chair",
    "relation": "sitting on",
    "negative_type": null | "reversed" | "wrong_pair" | "cross_image",
    "coco_image_id": int,
}

Negative sample strategy:
  Type A (reversed)     - swap subject and object
                          e.g. person ON chair → "Is the chair on the person?"
  Type B (wrong_pair)   - use a real relation from image but with wrong object
                          e.g. person-chair + dog-mat → "Is the person on the mat?"
  Type C (cross_image)  - borrow an object from a different image
                          e.g. person ON chair, borrow "skateboard" → "Is the person on the skateboard?"
                          Fallback when Type B fails (single-relation images)
  Ratio: 1/3 Type A, 1/3 Type B, 1/3 Type C
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict


random.seed(42)


# ── Config ────────────────────────────────────────────────────────────────────

TARGET_TOTAL  = 2700    # increased from 1800 to aim for ~900 pos + 900 neg
TARGET_POS    = 900
TARGET_NEG_A  = 300     # reversed
TARGET_NEG_B  = 300     # wrong pair (same image)
TARGET_NEG_C  = 300     # cross-image object substitution
MAX_PER_IMAGE = 6

COCO_FILENAME = "COCO_val2014_{:012d}.jpg"


# ── Load ──────────────────────────────────────────────────────────────────────

def load_relations(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def group_by_image(records: list[dict]) -> dict[int, list[dict]]:
    groups = defaultdict(list)
    for r in records:
        if r["coco_image_id"] != -1:
            groups[r["coco_image_id"]].append(r)
    return groups


def build_object_pool(records: list[dict]) -> dict[str, list[str]]:
    """
    Build {object_class: [list of other object classes that co-occur with it
    in different images]} for Type C negatives.

    We collect all object names per relation type so we can find
    plausible-but-wrong substitutes.
    """
    # relation → set of object names used with that relation
    rel_objects: dict[str, set[str]] = defaultdict(set)
    for r in records:
        rel_objects[r["relation"]].add(r["object_name"])
    return {rel: list(objs) for rel, objs in rel_objects.items()}


# ── Question builders ─────────────────────────────────────────────────────────

def build_question(subj: str, rel: str, obj: str) -> str:
    return (
        f"Is the {subj} {rel} the {obj} in the image? "
        f"Please answer Yes or No."
    )


def make_positive(record: dict, qid: int) -> dict:
    return {
        "question_id":  f"rel_{qid:05d}",
        "image":        COCO_FILENAME.format(record["coco_image_id"]),
        "question":     build_question(
            record["subject_name"], record["relation"], record["object_name"]
        ),
        "label":        "yes",
        "question_type":"relation",
        "subject_name": record["subject_name"],
        "object_name":  record["object_name"],
        "relation":     record["relation"],
        "negative_type":None,
        "coco_image_id":record["coco_image_id"],
    }


def make_negative_type_a(record: dict, qid: int) -> dict:
    """Type A: reversed — swap subject and object."""
    return {
        "question_id":  f"rel_{qid:05d}",
        "image":        COCO_FILENAME.format(record["coco_image_id"]),
        "question":     build_question(
            record["object_name"], record["relation"], record["subject_name"]
        ),
        "label":        "no",
        "question_type":"relation",
        "subject_name": record["object_name"],
        "object_name":  record["subject_name"],
        "relation":     record["relation"],
        "negative_type":"reversed",
        "coco_image_id":record["coco_image_id"],
    }


def make_negative_type_b(
    record: dict,
    image_records: list[dict],
    qid: int,
) -> dict | None:
    """
    Type B: wrong object pair from the same image.
    Subject stays the same, object comes from a different relation in the image.
    """
    # Prefer same subject, different object
    candidates = [
        r for r in image_records
        if r["relationship_id"] != record["relationship_id"]
        and r["object_name"] != record["object_name"]
        and r["subject_name"] == record["subject_name"]
    ]
    # Fallback: any other relation with a different object
    if not candidates:
        candidates = [
            r for r in image_records
            if r["relationship_id"] != record["relationship_id"]
            and r["object_name"] != record["object_name"]
        ]
    if not candidates:
        return None

    donor = random.choice(candidates)
    return {
        "question_id":  f"rel_{qid:05d}",
        "image":        COCO_FILENAME.format(record["coco_image_id"]),
        "question":     build_question(
            record["subject_name"], record["relation"], donor["object_name"]
        ),
        "label":        "no",
        "question_type":"relation",
        "subject_name": record["subject_name"],
        "object_name":  donor["object_name"],
        "relation":     record["relation"],
        "negative_type":"wrong_pair",
        "coco_image_id":record["coco_image_id"],
    }


def make_negative_type_c(
    record: dict,
    rel_object_pool: dict[str, list[str]],
    qid: int,
) -> dict | None:
    """
    Type C: cross-image object substitution.

    Keep the same image, subject, and relation.
    Replace the object with one that appears with this relation in OTHER images
    but is NOT the correct object for this image.

    e.g. "person sitting on chair" → borrow "skateboard" from another image
         → "Is the person sitting on the skateboard?" (label: no)

    This is harder than Type A/B because the substituted object is visually
    plausible (it genuinely appears with this relation somewhere).
    """
    pool = rel_object_pool.get(record["relation"], [])
    # Filter out the actual object in this image
    candidates = [o for o in pool if o != record["object_name"]]
    if not candidates:
        return None

    wrong_obj = random.choice(candidates)
    return {
        "question_id":  f"rel_{qid:05d}",
        "image":        COCO_FILENAME.format(record["coco_image_id"]),
        "question":     build_question(
            record["subject_name"], record["relation"], wrong_obj
        ),
        "label":        "no",
        "question_type":"relation",
        "subject_name": record["subject_name"],
        "object_name":  wrong_obj,
        "relation":     record["relation"],
        "negative_type":"cross_image",
        "coco_image_id":record["coco_image_id"],
    }


# ── Main builder ──────────────────────────────────────────────────────────────

def build_relation_split(
    records: list[dict],
    target_pos: int,
    target_neg_a: int,
    target_neg_b: int,
    target_neg_c: int,
    max_per_image: int,
) -> list[dict]:

    by_image      = group_by_image(records)
    rel_obj_pool  = build_object_pool(records)
    image_ids     = list(by_image.keys())
    random.shuffle(image_ids)

    pos_records   = []
    neg_a_records = []
    neg_b_records = []
    neg_c_records = []
    image_counts  = defaultdict(int)

    for img_id in image_ids:
        img_records = by_image[img_id]

        for record in img_records:
            if image_counts[img_id] >= max_per_image:
                break

            if len(pos_records) < target_pos:
                pos_records.append(record)
                image_counts[img_id] += 1
            elif len(neg_a_records) < target_neg_a:
                neg_a_records.append(record)
                image_counts[img_id] += 1
            elif len(neg_b_records) < target_neg_b:
                neg_b_records.append((record, img_records))
                image_counts[img_id] += 1
            elif len(neg_c_records) < target_neg_c:
                neg_c_records.append(record)
                image_counts[img_id] += 1

        if (len(pos_records)   >= target_pos   and
                len(neg_a_records) >= target_neg_a and
                len(neg_b_records) >= target_neg_b and
                len(neg_c_records) >= target_neg_c):
            break

    questions = []
    qid = 1

    for record in pos_records[:target_pos]:
        questions.append(make_positive(record, qid))
        qid += 1

    for record in neg_a_records[:target_neg_a]:
        questions.append(make_negative_type_a(record, qid))
        qid += 1

    for record, img_records in neg_b_records[:target_neg_b]:
        q = make_negative_type_b(record, img_records, qid)
        if q:
            questions.append(q)
            qid += 1

    for record in neg_c_records[:target_neg_c]:
        q = make_negative_type_c(record, rel_obj_pool, qid)
        if q:
            questions.append(q)
            qid += 1

    random.shuffle(questions)
    return questions


# ── Stats ─────────────────────────────────────────────────────────────────────

def print_stats(questions: list[dict]):
    total   = len(questions)
    pos     = sum(1 for q in questions if q["label"] == "yes")
    neg_a   = sum(1 for q in questions if q["negative_type"] == "reversed")
    neg_b   = sum(1 for q in questions if q["negative_type"] == "wrong_pair")
    neg_c   = sum(1 for q in questions if q["negative_type"] == "cross_image")

    rel_counts   = defaultdict(int)
    image_counts = defaultdict(int)
    for q in questions:
        rel_counts[q["relation"]] += 1
        image_counts[q["coco_image_id"]] += 1

    print(f"\n── X-POPE Relation Split Statistics ──")
    print(f"  Total questions  : {total:,}")
    print(f"  Positive (yes)   : {pos:,}")
    print(f"  Negative (no)    : {total - pos:,}  (ratio {pos/(total-pos):.2f}:1)")
    print(f"    Type A (reversed)    : {neg_a:,}")
    print(f"    Type B (wrong_pair)  : {neg_b:,}")
    print(f"    Type C (cross_image) : {neg_c:,}")
    print(f"  Unique images    : {len(image_counts):,}")

    print(f"\n── Top 15 relations ──")
    for rel, c in sorted(rel_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {rel:<25} {c:>5,}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build X-POPE relation split")
    parser.add_argument(
        "--input",
        default="data/processed/vg_parsed/vg_relations_filtered.json",
    )
    parser.add_argument(
        "--output",
        default="data/processed/xpope_relation.jsonl",
    )
    parser.add_argument("--target-pos",   type=int, default=TARGET_POS)
    parser.add_argument("--target-neg-a", type=int, default=TARGET_NEG_A)
    parser.add_argument("--target-neg-b", type=int, default=TARGET_NEG_B)
    parser.add_argument("--target-neg-c", type=int, default=TARGET_NEG_C)
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path} ...")
    records = load_relations(input_path)
    print(f"Loaded {len(records):,} relation records")

    print(f"\nBuilding relation split "
          f"(pos={args.target_pos}, "
          f"neg_A={args.target_neg_a}, "
          f"neg_B={args.target_neg_b}, "
          f"neg_C={args.target_neg_c}) ...")

    questions = build_relation_split(
        records      = records,
        target_pos   = args.target_pos,
        target_neg_a = args.target_neg_a,
        target_neg_b = args.target_neg_b,
        target_neg_c = args.target_neg_c,
        max_per_image= MAX_PER_IMAGE,
    )

    print_stats(questions)

    with open(output_path, "w") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(questions):,} questions → {output_path}")


if __name__ == "__main__":
    main()
