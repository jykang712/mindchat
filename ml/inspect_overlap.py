from collections import Counter

from train_baseline import load_rows

train = load_rows("Training")
validation = load_rows("Validation")

train_ids = {talk_id for talk_id, _, _ in train if talk_id}
train_texts = {text for _, text, _ in train}

counts = Counter(
    (talk_id in train_ids, text in train_texts)
    for talk_id, text, _ in validation
)

print("Training rows:", len(train))
print("Unique training IDs:", len(train_ids))
print("Unique training texts:", len(train_texts))
print("Validation rows:", len(validation))
print("ID match only:", counts[True, False])
print("Text match only:", counts[False, True])
print("Both match:", counts[True, True])
print("Neither matches:", counts[False, False])