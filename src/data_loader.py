import csv
from pathlib import Path

import pandas as pd


DEFAULT_CSV = Path(__file__).resolve().parents[1] / "asap-master" / "data" / "train.csv"

# ASAP 的 CSV 是带 BOM 的 UTF-8，utf-8-sig 会自动去掉 BOM；
# 其余候选编码用于兼容其它来源的中文 CSV。
ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "latin1")


def _sniff_delimiter(csv_path, encoding):
    """从文件头部探测分隔符，失败时回退为逗号。"""
    with open(csv_path, "r", encoding=encoding, newline="") as f:
        sample = f.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def load_asap(csv_path=DEFAULT_CSV, encoding=None, sep=None):
    """读取 ASAP 数据集 CSV，返回 DataFrame。

    依次尝试候选编码；若逗号解析后只有一列，则自动探测分隔符重读。
    列名统一去掉可能残留的 BOM 和首尾空白。
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到数据文件: {csv_path}")

    encodings = (encoding,) if encoding else ENCODINGS
    last_error = None
    for enc in encodings:
        try:
            delimiter = sep or ","
            df = pd.read_csv(csv_path, encoding=enc, sep=delimiter, header=0)
            if df.shape[1] == 1 and sep is None:
                delimiter = _sniff_delimiter(csv_path, enc)
                df = pd.read_csv(csv_path, encoding=enc, sep=delimiter, header=0)
            df.columns = [str(col).lstrip("\ufeff").strip() for col in df.columns]
            print(f"读取成功: {csv_path}")
            print(f"  encoding={enc}, sep={delimiter!r}, shape={df.shape}")
            return df
        except (UnicodeDecodeError, UnicodeError, pd.errors.ParserError) as exc:
            last_error = exc
            print(f"  跳过 encoding={enc}: {type(exc).__name__}: {exc}")

    raise RuntimeError(f"所有候选编码均读取失败: {csv_path}") from last_error


def inspect_data(df, n=3):
    """逐行打印全部列名，并打印前 n 行数据。"""
    print(f"\n=== df.columns（共 {len(df.columns)} 列）===")
    for i, col in enumerate(df.columns, start=1):
        print(f"{i:2d}. {col}")

    print(f"\n=== df.head({n}) ===")
    with pd.option_context(
        "display.max_columns", None,
        "display.width", None,
        "display.max_colwidth", None,
        "display.unicode.east_asian_width", True,
    ):
        print(df.head(n).to_string())


if __name__ == "__main__":
    df = load_asap()
    inspect_data(df, n=3)
