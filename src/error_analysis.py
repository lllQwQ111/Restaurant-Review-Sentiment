"""ASAP 基线模型错误分析：聚焦 Macro-F1 最低的 3 个方面。

运行: python src/error_analysis.py
产物: outputs/error_analysis.txt
"""

import re
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

try:
    from data_loader import load_asap
except ImportError:  # 以 python -m src.error_analysis 方式运行时
    from src.data_loader import load_asap


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
TEST_CSV = ROOT / "asap-master" / "data" / "test.csv"
METRICS_CSV = OUTPUT_DIR / "baseline_metrics.csv"
REPORT_PATH = OUTPUT_DIR / "error_analysis.txt"

TOP_K = 3
N_SAMPLES = 5
RANDOM_STATE = 42
TEXT_COL = "review"
NOT_MENTIONED = -2
LABELS = [-1, 0, 1]
LABEL_NAMES = {-1: "负面(-1)", 0: "中性(0)", 1: "正面(1)"}

SEP = "=" * 78
SUB_SEP = "-" * 78


def _confusion_lines(cm):
    """把混淆矩阵格式化为可读文本行。"""
    table = pd.DataFrame(
        cm,
        index=[f"真实={label}" for label in LABELS],
        columns=[f"预测={label}" for label in LABELS],
    )
    lines = [
        "混淆矩阵（行 = 真实标签，列 = 预测标签，对角线为预测正确）",
        table.to_string(),
    ]

    off_diagonal = [
        (true_label, pred_label, int(cm[i, j]))
        for i, true_label in enumerate(LABELS)
        for j, pred_label in enumerate(LABELS)
        if true_label != pred_label
    ]
    off_diagonal.sort(key=lambda item: item[2], reverse=True)
    true_label, pred_label, count = off_diagonal[0]
    lines.append(
        "最常见误判方向: 真实 "
        f"{LABEL_NAMES[true_label]} 被预测为 {LABEL_NAMES[pred_label]}，共 {count} 条"
    )
    return lines


def analyze_aspect(aspect, recorded_macro_f1, test_df, lines):
    """加载模型、复算预测、输出混淆矩阵与随机误判样本。"""
    model_path = OUTPUT_DIR / f"model_{aspect}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"缺少模型文件，请先运行 baseline_lr.py: {model_path}")

    # 与训练保持一致：过滤掉 -2（未提及）样本后再预测
    sub = test_df.loc[
        test_df[aspect] != NOT_MENTIONED, ["id", TEXT_COL, aspect]
    ].reset_index(drop=True)
    x_text = sub[TEXT_COL].fillna("").astype(str)
    y_true = pd.to_numeric(sub[aspect]).astype(int).to_numpy()

    bundle = joblib.load(model_path)
    vectorizer = bundle["vectorizer"]
    classifier = bundle["classifier"]
    y_pred = classifier.predict(vectorizer.transform(x_text))

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    lines.extend(
        [
            "",
            SEP,
            f"[{aspect}]  Macro-F1 = {macro_f1:.4f} （全 18 个方面中最低的 {TOP_K} 个之一）",
            SEP,
            f"模型文件: {model_path}",
            f"评估样本: {len(sub)} 条（已排除 {NOT_MENTIONED} 未提及样本）",
            f"复算指标: accuracy={accuracy:.4f}, macro_f1={macro_f1:.4f}",
            "与 baseline_metrics.csv 对比: "
            + (
                "一致"
                if abs(macro_f1 - recorded_macro_f1) < 1e-6
                else f"不一致（CSV 记录 {recorded_macro_f1:.4f}）"
            ),
            "",
        ]
    )
    lines.extend(_confusion_lines(cm))

    error_mask = y_true != y_pred
    errors = sub.loc[error_mask, ["id", TEXT_COL]].copy()
    errors["y_true"] = y_true[error_mask]
    errors["y_pred"] = y_pred[error_mask]
    n_errors = len(errors)
    n_pick = min(N_SAMPLES, n_errors)

    lines.extend(
        [
            "",
            SUB_SEP,
            f"误判样本: 共 {n_errors} 条 / {len(sub)} 条 "
            f"({n_errors / len(sub):.2%})，随机抽取 {n_pick} 条 "
            f"(random_state={RANDOM_STATE})",
            SUB_SEP,
        ]
    )

    if n_pick == 0:
        lines.append("该方面在测试集上没有被误判的样本。")
        return

    for n, (_, row) in enumerate(
        errors.sample(n=n_pick, random_state=RANDOM_STATE).iterrows(), start=1
    ):
        # 数据中残留的是字面量 \\n，不是真换行；仅做显示层替换，不改变其它内容
        text = re.sub(r"\\+n", " ", str(row[TEXT_COL]))
        text = text.replace("\r", " ").replace("\n", " ").strip()
        lines.extend(
            [
                "",
                f"--- 误判样本 {n}/{n_pick} ---",
                f"id: {row['id']}",
                f"真实标签: {LABEL_NAMES[int(row['y_true'])]}"
                f"    预测标签: {LABEL_NAMES[int(row['y_pred'])]}",
                f"评论原文: {text}",
            ]
        )


def main():
    if not METRICS_CSV.exists():
        raise FileNotFoundError(f"请先运行 baseline_lr.py 生成指标文件: {METRICS_CSV}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(METRICS_CSV, encoding="utf-8-sig")
    if not {"aspect", "macro_f1"}.issubset(metrics.columns):
        raise ValueError(f"指标文件缺少 aspect / macro_f1 列: {METRICS_CSV}")
    worst = metrics.nsmallest(TOP_K, "macro_f1").reset_index(drop=True)

    test_df = load_asap(TEST_CSV)
    if TEXT_COL not in test_df.columns:
        raise ValueError(f"测试集缺少文本列 {TEXT_COL}")

    lines = [
        SEP,
        "ASAP 基线模型（TF-IDF + 逻辑回归）错误分析报告",
        SEP,
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"测试数据: {TEST_CSV}",
        f"筛选依据: {METRICS_CSV} 中 Macro-F1 最低的 {TOP_K} 个方面",
        f"抽样设置: 每个方面各随机抽取 {N_SAMPLES} 条误判评论"
        f"（random_state={RANDOM_STATE}，结果可复现）",
        "标签含义: 1=正面, 0=中性, -1=负面, -2=未提及（评估时已排除）",
        "显示说明: 评论原文保持原样，仅把数据中残留的字面换行符替换为空格",
        "",
        f"Macro-F1 最低的 {TOP_K} 个方面:",
    ]
    for i, row in worst.iterrows():
        lines.append(
            f"  {i + 1}. {row['aspect']:<28} macro_f1={row['macro_f1']:.4f}"
            f"  accuracy={row['accuracy']:.4f}"
        )

    for _, row in worst.iterrows():
        analyze_aspect(row["aspect"], float(row["macro_f1"]), test_df, lines)

    report = "\n".join(lines)
    print(report)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8-sig")
    print(f"\n报告已保存: {REPORT_PATH}")


if __name__ == "__main__":
    main()
