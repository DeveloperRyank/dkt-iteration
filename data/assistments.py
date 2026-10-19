"""Loader for ASSISTments 2009-2010 Skill Builder (data/raw/skill_builder_data_corrected.csv).

Preprocessing decisions (each is a documented approximation):
1. Keep rows with correct in {0,1} and a non-empty skill_name (rows with no
   skill tag are dropped).
2. One interaction = one unique (user_id, order_id). Multi-skill problems
   appear as duplicate rows sharing an order_id, one row per skill; we keep
   the FIRST skill listed per interaction so every step has exactly one skill,
   as DKT requires.
3. Per student, interactions are ordered by numeric order_id.
4. Skills with fewer than min_skill_count interactions are dropped, then
   survivors are remapped to contiguous indices 0..num_skills-1.
5. Students with fewer than min_seq_len interactions are dropped.
"""

import csv
import gzip
import json
import os
from collections import Counter, defaultdict

SKILL_COL = "skill_name"
CACHE_PATH = "data/processed/assistments2009.json.gz"


def _cache_path_for(min_skill_count, min_seq_len):
    base, ext = os.path.splitext(CACHE_PATH)
    return f"{base}.s{min_skill_count}.l{min_seq_len}{ext}"


def _read_events(path):
    """Return [(user_id, order_id, skill, correct)], first skill per interaction."""
    seen = set()
    events = []
    dropped_bad_correct = 0
    dropped_blank_skill = 0
    dropped_dup_skill = 0
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            c = row["correct"]
            if c not in ("0", "1"):
                dropped_bad_correct += 1
                continue
            skill = (row[SKILL_COL] or "").strip()
            if skill in ("", "None", "nan"):
                dropped_blank_skill += 1
                continue
            key = (row["user_id"], int(row["order_id"]))
            if key in seen:
                dropped_dup_skill += 1
                continue
            seen.add(key)
            events.append((row["user_id"], key[1], skill, int(c)))
    return events, {
        "dropped_bad_correct": dropped_bad_correct,
        "dropped_blank_skill": dropped_blank_skill,
        "dropped_duplicate_skill_rows": dropped_dup_skill,
    }


def load_assistments(path="data/raw/skill_builder_data_corrected.csv",
                     min_skill_count=1000, min_seq_len=5, verbose=True,
                     use_cache=True):
    cache = _cache_path_for(min_skill_count, min_seq_len)
    if use_cache and os.path.exists(cache):
        with gzip.open(cache, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        if verbose:
            print(f"loaded cached processed data: {cache}")
        return ({"num_skills": payload["num_skills"], "sequences": payload["sequences"]},
                payload["skill_index"])

    events, drop_stats = _read_events(path)
    stats = {"unique_interactions": len(events), **drop_stats}

    skill_counts = Counter(skill for _, _, skill, _ in events)
    kept_skills = {s for s, c in skill_counts.items() if c >= min_skill_count}
    events = [e for e in events if e[2] in kept_skills]
    stats.update(skills_total=len(skill_counts), skills_kept=len(kept_skills),
                 interactions_after_skill_filter=len(events))

    skill_index = {s: i for i, s in enumerate(sorted(kept_skills))}

    by_user = defaultdict(list)
    for user, order, skill, c in events:
        by_user[user].append((order, skill_index[skill], c))

    sequences = []
    dropped_short = 0
    for user, evs in sorted(by_user.items()):
        if len(evs) < min_seq_len:
            dropped_short += 1
            continue
        evs.sort(key=lambda t: t[0])
        sequences.append({"skills": [s for _, s, _ in evs],
                          "correct": [c for _, _, c in evs]})
    stats.update(students_total=len(by_user), students_dropped_short=dropped_short,
                 students_kept=len(sequences),
                 interactions_kept=sum(len(s["skills"]) for s in sequences))

    if verbose:
        print("preprocessing stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        lens = sorted(len(s["skills"]) for s in sequences)
        print(f"  sequence lengths: min={lens[0]} median={lens[len(lens)//2]} "
              f"p90={lens[int(len(lens)*0.9)]} max={lens[-1]}")
        all_c = [c for s in sequences for c in s["correct"]]
        print(f"  overall accuracy: {sum(all_c)/len(all_c):.4f} "
              f"over {len(all_c)} interactions")
        print(f"  kept skills: {sorted(skill_index, key=skill_index.get)}")
    if use_cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with gzip.open(cache, "wt", encoding="utf-8") as f:
            json.dump({"num_skills": len(skill_index), "sequences": sequences,
                       "skill_index": skill_index}, f)
    return {"num_skills": len(skill_index), "sequences": sequences}, skill_index


if __name__ == "__main__":
    data, skill_index = load_assistments()
    from data.dataset import make_loader
    loader = make_loader(data["sequences"][:8], batch_size=8)
    batch = next(iter(loader))
    print("smoke batch shapes:", {k: tuple(v.shape) for k, v in batch.items()},
          "num_skills =", data["num_skills"])
