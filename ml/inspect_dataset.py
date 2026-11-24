import json
from collections import Counter
from pathlib import Path

files = sorted((Path(__file__).resolve().parent / "data" / "raw").rglob("*.json"))

for path in files:
    with path.open(encoding="utf-8-sig") as file:
        data = json.load(file)

    print(f"\n=== {path.name} ===")
    if not isinstance(data, list) or not data:
        print("Top-level type:", type(data).__name__)
        if isinstance(data, dict):
            print("Top-level keys:", list(data))
        continue

    first = data[0]
    print("Records:", len(data))
    print("Record keys:", list(first))

    profile = first.get("profile") or {}
    emotion = profile.get("emotion") or {}
    talk = first.get("talk") or {}
    content = talk.get("content") or {}

    print("Profile keys:", list(profile))
    print("Emotion keys:", list(emotion))
    print("Talk keys:", list(talk))
    print("Utterance keys:", list(content))

    label_counts = Counter()
    first_utterance_count = 0

    for record in data:
        profile = record.get("profile") or {}
        emotion = profile.get("emotion") or {}
        content = (record.get("talk") or {}).get("content") or {}

        label_counts[emotion.get("type") or "(missing)"] += 1
        text = content.get("HS01")
        if isinstance(text, str) and text.strip():
            first_utterance_count += 1

    print("Records with non-empty HS01:", first_utterance_count)
    print("Label counts:", dict(sorted(label_counts.items())))