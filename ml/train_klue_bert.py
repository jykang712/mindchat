"""MindChat의 기존 분할로 KLUE-BERT 감정 분류기를 학습한다."""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from compare_emotion_context import LABELS, load_rows

MODEL_ID = "klue/bert-base"
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
LABEL_IDS = list(LABEL_TO_ID.values())


class EmotionDataset(Dataset):
    def __init__(self, rows, tokenizer, use_context, max_length):
        first_turns = [row.hs01 for row in rows]
        second_turns = [row.hs02 for row in rows] if use_context else None
        self.inputs = tokenizer(
            first_turns,
            text_pair=second_turns,
            truncation=True,
            max_length=max_length,
        )
        self.labels = [LABEL_TO_ID[row.label] for row in rows]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: values[index] for key, values in self.inputs.items()}
        item["labels"] = self.labels[index]
        return item


def compute_metrics(eval_prediction):
    actual = eval_prediction.label_ids
    predicted = np.argmax(eval_prediction.predictions, axis=-1)
    class_f1 = f1_score(actual, predicted, labels=LABEL_IDS, average=None, zero_division=0)
    matrix = confusion_matrix(actual, predicted, labels=LABEL_IDS)
    hurt = LABEL_TO_ID["상처"]
    sadness = LABEL_TO_ID["슬픔"]
    hurt_as_sadness = int(matrix[hurt, sadness])
    sadness_as_hurt = int(matrix[sadness, hurt])
    return {
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": float(class_f1.mean()),
        "hurt_f1": float(class_f1[hurt]),
        "sadness_f1": float(class_f1[sadness]),
        "hurt_to_sadness": hurt_as_sadness,
        "sadness_to_hurt": sadness_as_hurt,
        "hurt_sadness_confusions": hurt_as_sadness + sadness_as_hurt,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", choices=("hs01", "hs01-hs02"), required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs와 --batch-size는 1 이상이어야 합니다.")
    if not 4 <= args.max_length <= 512:
        parser.error("--max-length는 4~512여야 합니다.")

    train = load_rows("Training")
    validation = load_rows("Validation")
    # 기존 TF-IDF 기준 모델과 같은 중복 제외 규칙을 적용한다.
    validation_ids = {row.talk_id for row in validation if row.talk_id}
    validation_texts = {row.hs01 for row in validation}
    clean_train = [
        row for row in train
        if row.talk_id not in validation_ids and row.hs01 not in validation_texts
    ]
    if not clean_train or not validation:
        raise ValueError("중복 제거 후 학습 데이터 또는 검증 데이터가 비어 있습니다.")

    use_context = args.input == "hs01-hs02"
    output_dir = Path(__file__).resolve().parent / "models" / f"klue_bert_{args.input.replace('-', '_')}"
    print(f"입력: {args.input}")
    print(f"학습: {len(clean_train):,}건, 검증: {len(validation):,}건")
    print(f"학습 장치: {'CUDA GPU' if torch.cuda.is_available() else 'CPU'}")

    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL_TO_ID,
        use_safetensors=True,
    )
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        report_to="none",
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=EmotionDataset(clean_train, tokenizer, use_context, args.max_length),
        eval_dataset=EmotionDataset(validation, tokenizer, use_context, args.max_length),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    trainer.save_metrics("eval", metrics)

    print(f"\nAccuracy: {metrics['eval_accuracy']:.4f}")
    print(f"Macro F1: {metrics['eval_macro_f1']:.4f}")
    print(f"상처 F1: {metrics['eval_hurt_f1']:.4f}")
    print(f"슬픔 F1: {metrics['eval_sadness_f1']:.4f}")
    print(f"상처 → 슬픔: {metrics['eval_hurt_to_sadness']:,}건")
    print(f"슬픔 → 상처: {metrics['eval_sadness_to_hurt']:,}건")
    print(f"양방향 혼동 합계: {metrics['eval_hurt_sadness_confusions']:,}건")
    print(f"KLUE-BERT 모델과 토크나이저 저장: {output_dir}")


if __name__ == "__main__":
    main()
