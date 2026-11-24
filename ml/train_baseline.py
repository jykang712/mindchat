import json
import re
import unicodedata
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

ML_DIR = Path(__file__).resolve().parent
RAW_DIR = ML_DIR / "data" / "raw"

EMOTIONS = {
    "E1": "분노",
    "E2": "슬픔",
    "E3": "불안",
    "E4": "상처",
    "E5": "당황",
    "E6": "기쁨",
}


# def load_rows(split):
def load_rows(split, *, review_details=None):
    files = sorted(RAW_DIR.rglob(f"*_{split}.json"))
    if len(files) != 1:
        raise ValueError(f"{split} JSON이 1개여야 합니다. 발견: {len(files)}개")

    with files[0].open(encoding="utf-8-sig") as file:
        records = json.load(file)

    rows = []
    for index, record in enumerate(records, start=1):
        code = record["profile"]["emotion"]["type"]
        if not isinstance(code, str) or not re.fullmatch(r"E[1-6][0-9]", code):
            raise ValueError(f"{split} {index}번 레코드의 감정 코드 확인: {code!r}")

        talk = record["talk"]
        text = talk["content"]["HS01"]
        if isinstance(text, str) and text.strip():
            text = " ".join(unicodedata.normalize("NFC", text).split())
            talk_id = talk.get("id")
            if isinstance(talk_id, dict):
                talk_id = json.dumps(talk_id, sort_keys=True) if talk_id else None

            rows.append((talk_id, text, EMOTIONS[code[:2]]))

            #--------------------------------------
            if review_details is not None:
                content = talk["content"]
                review_details.append({
                    "감정 코드": code,
                    "HS02": content.get("HS02") or "",
                    "HS03": content.get("HS03") or "",
                })

    return rows


def main():
    train = load_rows("Training")
    # validation = load_rows("Validation")
    validation_details = []
    validation = load_rows("Validation", review_details=validation_details)

    # train_ids = {talk_id for talk_id, _, _ in train if talk_id}
    # train_texts = {text for _, text, _ in train}

    # # 학습 데이터에 이미 나온 대화 ID나 동일 문장을 검증에서 제외합니다.
    # clean_validation = [
    #     (talk_id, text, label)
    #     for talk_id, text, label in validation
    #     if talk_id not in train_ids and text not in train_texts
    # ]

    # print(f"학습: {len(train):,}건")
    # print(f"검증: {len(clean_validation):,}건")
    # print(f"검증에서 제외한 중복: {len(validation) - len(clean_validation):,}건")

    # if not train or not clean_validation:
    #     raise ValueError("학습 또는 검증 데이터가 비어 있습니다.")

    # _, x_train, y_train = zip(*train)
    # _, x_val, y_val = zip(*clean_validation)

    validation_ids = {talk_id for talk_id, _, _ in validation if talk_id}
    validation_texts = {text for _, text, _ in validation}

    clean_train = [
        (talk_id, text, label)
        for talk_id, text, label in train
        if talk_id not in validation_ids and text not in validation_texts
    ]

    print(f"원본 학습: {len(train):,}건")
    print(f"학습에서 제외한 중복: {len(train) - len(clean_train):,}건")
    print(f"실제 학습: {len(clean_train):,}건")
    print(f"검증: {len(validation):,}건")

    if not clean_train or not validation:
        raise ValueError("학습 또는 검증 데이터가 비어 있습니다.")

    _, x_train, y_train = zip(*clean_train)
    _, x_val, y_val = zip(*validation)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            min_df=3,
            max_features=100_000,
        )),
        ("classifier", LogisticRegression(
            max_iter=500,
            class_weight="balanced",
        )),
    ])

    model.fit(x_train, y_train)
    predictions = model.predict(x_val)
    labels = list(EMOTIONS.values())

    print(f"\nAccuracy: {accuracy_score(y_val, predictions):.4f}")
    print(f"Macro F1: {f1_score(y_val, predictions, labels=labels, average='macro', zero_division=0):.4f}")
    print(classification_report(
        y_val, predictions, labels=labels, digits=4, zero_division=0
    ))

    #혼동행렬 출력
    labels = ["분노", "슬픔", "불안", "상처", "당황", "기쁨"]

    matrix = confusion_matrix(y_val, predictions, labels=labels)

    matrix_df = pd.DataFrame(
        matrix,
        index=pd.Index(labels, name="실제 감정"),
        columns=pd.Index(labels, name="예측 감정"),
    )

    print("\n혼동행렬 (건수)")
    print(matrix_df.to_string())

    print("\n실제 '상처' 문장의 오분류 건수")
    print(
        matrix_df.loc["상처"]
        .drop("상처")
        .sort_values(ascending=False)
        .to_string()
    )

    # X_val은 벡터화하기 전의 검증 문장 목록
    # y_val과 predictions는 confusion_matrix()의 첫 번째, 두 번째 인자
    review_df = pd.DataFrame({
        "문장": list(x_val),
        "실제 감정": list(y_val),
        "예측 감정": list(predictions),
    }).join(pd.DataFrame(validation_details))

    review_groups = [
        ("정분류", "상처", "상처"),
        ("오분류", "상처", "슬픔"),
        ("정분류", "슬픔", "슬픔"),
        ("오분류", "슬픔", "상처"),
    ]

    for group, actual, predicted in review_groups:
        matched = review_df.loc[
            review_df["실제 감정"].eq(actual)
            & review_df["예측 감정"].eq(predicted)
        ]
        samples = matched.sample(n=min(20, len(matched)), random_state=42)

        print(
            f"\n[{group} | {actual} → {predicted}] "
            f"전체 {len(matched)}건 중 {len(samples)}건"
        )
        #--------------------------------------------------------------------------
        sample_rows = samples[["문장", "감정 코드", "HS02", "HS03"]]

        for i, (sentence, code, hs02, hs03) in enumerate(
            sample_rows.itertuples(index=False, name=None), start=1
        ):
            print(f"{i:02d}. [원본 코드 {code}] {sentence}")
            print(f"    HS02 | {hs02 or '(없음)'}")
            print(f"    HS03 | {hs03 or '(없음)'}")

    # train: 51628, valid: 261
    # model_path = ML_DIR / "models" / "emotion_baseline.joblib"

    # train: 45246, valid: 6640
    model_path = ML_DIR / "models" / "emotion_baseline_v2.joblib"
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    print(f"모델 저장: {model_path}")

    sample = "내일 발표인데 실수할까 봐 너무 걱정돼."
    print(f"예시 예측: {sample} → {model.predict([sample])[0]}")


if __name__ == "__main__":
    main()