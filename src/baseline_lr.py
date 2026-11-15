"""ASAP 方面级情感分析基线：TF-IDF + 逻辑回归（1/0/-1 三分类）。

运行: python src/baseline_lr.py
产物: outputs/baseline_metrics.csv, outputs/model_{aspect}.pkl
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

try:
    from data_loader import load_asap
except ImportError:  # 以 python -m src.baseline_lr 方式运行时
    from src.data_loader import load_asap


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "asap-master" / "data"
OUTPUT_DIR = ROOT / "outputs"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

TEXT_COL = "review"
LABELS = [-1, 0, 1]
NOT_MENTIONED = -2


def build_vectorizer(min_df=2, max_features=50000):
    """中文评论无空格分词，用字符级 1-2 gram，避免依赖第三方分词库。"""
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )


def _filter_aspect(df, aspect):
    """过滤掉 -2（未提及）样本，返回 review 文本与三分类标签。"""
    sub = df.loc[df[aspect] != NOT_MENTIONED, [TEXT_COL, aspect]]
    x_text = sub[TEXT_COL].fillna("").astype(str)
    y = pd.to_numeric(sub[aspect]).astype(int)
    return x_text, y


def train_aspect(aspect, train_df, test_df, output_dir=OUTPUT_DIR):
    """训练单个方面，在 test 上评估，并保存模型与 TfidfVectorizer。"""
    x_train_text, y_train = _filter_aspect(train_df, aspect)
    x_test_text, y_test = _filter_aspect(test_df, aspect)

    if y_train.nunique() < 2:
        raise ValueError(f"{aspect}: 训练集类别少于 2 个，无法训练逻辑回归")
    if len(y_test) == 0:
        raise ValueError(f"{aspect}: 测试集过滤后为空")

    vectorizer = build_vectorizer()
    try:
        x_train = vectorizer.fit_transform(x_train_text)
    except ValueError:  # 小语料下 min_df=2 可能产生空词表，放宽到 1
        vectorizer = build_vectorizer(min_df=1)
        x_train = vectorizer.fit_transform(x_train_text)
    x_test = vectorizer.transform(x_test_text)

    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"model_{aspect}.pkl"
    joblib.dump(
        {
            "aspect": aspect,
            "vectorizer": vectorizer,
            "classifier": clf,
            "labels": LABELS,
        },
        model_path,
        compress=3,
    )

    return {
        "aspect": aspect,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(
            y_test, y_pred, labels=LABELS, average="macro", zero_division=0
        ),
        "micro_f1": f1_score(
            y_test, y_pred, labels=LABELS, average="micro", zero_division=0
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df = load_asap(TRAIN_CSV)
    test_df = load_asap(TEST_CSV)

    aspects = list(train_df.columns[3:21])
    if len(aspects) != 18 or list(test_df.columns[3:21]) != aspects:
        raise ValueError("train/test 的 18 个方面列不一致")
    if TEXT_COL not in train_df.columns or TEXT_COL not in test_df.columns:
        raise ValueError(f"缺少文本列 {TEXT_COL}")

    rows = []
    for i, aspect in enumerate(aspects, start=1):
        print(f"[{i:2d}/{len(aspects)}] {aspect} ...", end=" ", flush=True)
        metrics = train_aspect(aspect, train_df, test_df)
        rows.append(metrics)
        print(
            f"acc={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"micro_f1={metrics['micro_f1']:.4f}",
            flush=True,
        )

    result = pd.DataFrame(
        rows,
        columns=["aspect", "n_train", "n_test", "accuracy", "macro_f1", "micro_f1"],
    )
    metrics_path = OUTPUT_DIR / "baseline_metrics.csv"
    result.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print("\n=== 18 个方面汇总指标 ===")
    print(result.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n总体平均 Macro-F1: {result['macro_f1'].mean():.4f}")
    print(f"指标文件: {metrics_path}")
    print(f"模型目录: {OUTPUT_DIR} (共 {len(result)} 个 model_*.pkl)")


if __name__ == "__main__":
    main()
