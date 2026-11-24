"""같은 데이터에서 첫 발화와 첫 두 발화의 감정 분류 성능을 비교한다."""

import json
import re
import unicodedata
from pathlib import Path
from typing import NamedTuple
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from random import Random

RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
EMOTIONS = {
    "E1": "분노",
    "E2": "슬픔",
    "E3": "불안",
    "E4": "상처",
    "E5": "당황",
    "E6": "기쁨",
}
LABELS = list(EMOTIONS.values())


class Dialogue(NamedTuple):
    talk_id: str | int | None
    hs01: str
    hs02: str
    label: str


def normalize_text(value):
    if not isinstance(value, str):
        return ""
    return " ".join(unicodedata.normalize("NFC", value).split())


def load_rows(split):
    paths = sorted(RAW_DIR.rglob(f"*_{split}.json"))
    if len(paths) != 1:
        raise ValueError(f"{RAW_DIR}에서 {split} JSON이 1개여야 합니다. 발견: {len(paths)}개")

    with paths[0].open(encoding="utf-8-sig") as file:
        records = json.load(file)

    rows = []
    for index, record in enumerate(records, start=1):
        code = record["profile"]["emotion"]["type"]
        if not isinstance(code, str) or not re.fullmatch(r"E[1-6][0-9]", code):
            raise ValueError(f"{split} {index}번 레코드의 감정 코드 확인: {code!r}")

        talk = record["talk"]
        content = talk["content"]
        hs01 = normalize_text(content.get("HS01"))
        if not hs01:
            continue

        talk_id = talk.get("id")
        if isinstance(talk_id, dict):
            talk_id = json.dumps(talk_id, sort_keys=True) if talk_id else None

        rows.append(Dialogue(talk_id, hs01, normalize_text(content.get("HS02")), EMOTIONS[code[:2]]))

    return rows


def evaluate(x_train, y_train, x_val, y_val):
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            min_df=3,
            max_features=100_000,
        )),
        ("classifier", LogisticRegression(max_iter=500, class_weight="balanced")),
    ])
    model.fit(x_train, y_train)
    predictions = model.predict(x_val)
    class_f1 = dict(zip(LABELS, f1_score(y_val, predictions, labels=LABELS, average=None, zero_division=0)))
    matrix = confusion_matrix(y_val, predictions, labels=LABELS)
    hurt = LABELS.index("상처")
    sadness = LABELS.index("슬픔")
    hurt_as_sadness = matrix[hurt, sadness]
    sadness_as_hurt = matrix[sadness, hurt]
    # return {
    metrics = {
        "Accuracy": accuracy_score(y_val, predictions),
        "Macro F1": f1_score(y_val, predictions, labels=LABELS, average="macro", zero_division=0),
        "상처 F1": class_f1["상처"],
        "슬픔 F1": class_f1["슬픔"],
        "상처 → 슬픔": int(hurt_as_sadness),
        "슬픔 → 상처": int(sadness_as_hurt),
        "양방향 혼동 합계": int(hurt_as_sadness + sadness_as_hurt),
    }
    return metrics, predictions, model


def main():
    train = load_rows("Training")
    validation = load_rows("Validation")

    # 기존 기준 모델과 동일하게 검증에 나온 대화 ID 또는 HS01 문장을 학습에서 제외한다.
    validation_ids = {row.talk_id for row in validation if row.talk_id}
    validation_texts = {row.hs01 for row in validation}
    clean_train = [
        row for row in train
        if row.talk_id not in validation_ids and row.hs01 not in validation_texts
    ]
    if not clean_train or not validation:
        raise ValueError("중복 제거 후 학습 데이터 또는 검증 데이터가 비어 있습니다.")

    print(f"학습: {len(clean_train):,}건 (원본 {len(train):,}건, 중복 제외 {len(train) - len(clean_train):,}건)")
    print(f"검증: {len(validation):,}건")
    print(f"HS02 없음: 학습 {sum(not row.hs02 for row in clean_train):,}건, "
          f"검증 {sum(not row.hs02 for row in validation):,}건")

    y_train = [row.label for row in clean_train]
    y_val = [row.label for row in validation]

    # results = []
    # for use_context in (False, True):
    #     x_train = [input_text(row, use_context) for row in clean_train]
    #     x_val = [input_text(row, use_context) for row in validation]
    #     results.append(evaluate(x_train, y_train, x_val, y_val))

    # baseline, context = results

    results = []
    predictions_by_model = []

    for use_context in (False, True):
        x_train = [input_text(row, use_context) for row in clean_train]
        x_val = [input_text(row, use_context) for row in validation]

        metrics, predictions, model = evaluate(x_train, y_train, x_val, y_val)
        results.append(metrics)
        predictions_by_model.append(predictions)

        if use_context:
            model_path = (
                Path(__file__).resolve().parent / "models" / "emotion_hs01_hs02.joblib"
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            print(f"두 발화 모델 저장: {model_path}")

    baseline, context = results
    baseline_predictions, context_predictions = predictions_by_model
    print("\n지표 | HS01 | HS01 + HS02 | 변화 (두 번째 - 첫 번째)")
    for metric, before in baseline.items():
        after = context[metric]
        if isinstance(before, int):
            print(f"{metric} | {before:,}건 | {after:,}건 | {after - before:+,}건")
        else:
            print(f"{metric} | {before:.4f} | {after:.4f} | {after - before:+.4f}")

    improved = []
    regressed = []

    for row, before, after in zip(
        validation, baseline_predictions, context_predictions, strict=True
    ):
        if before != row.label and after == row.label:
            improved.append((row, before, after))
        elif before == row.label and after != row.label:
            regressed.append((row, before, after))

    rng = Random(42)

    for title, cases in [
        ("개선 · 오답 → 정답", improved),
        ("악화 · 정답 → 오답", regressed),
    ]:
        samples = rng.sample(cases, k=min(10, len(cases)))
        print(f"\n[{title}] 전체 {len(cases):,}건 중 {len(samples)}건")

        for i, (row, before, after) in enumerate(samples, start=1):
            print(f"\n{i:02d}. 실제 감정: {row.label}")
            print(f"    HS01 모델 예측: {before}")
            print(f"    HS01 + HS02 모델 예측: {after}")
            print(f"    HS01: {row.hs01}")
            print(f"    HS02: {row.hs02 or '(없음)'}")


def input_text(row, use_context):
    if use_context and row.hs02:
        return f"{row.hs01}\n{row.hs02}"
    return row.hs01


if __name__ == "__main__":
    main()
