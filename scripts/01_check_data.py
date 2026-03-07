from pathlib import Path
import pandas as pd

data_dir = Path("data")

print("Checking data files...\n")

files = list(data_dir.glob("*"))

if not files:
    print("data 文件夹目前是空的。")
else:
    for file in files:
        print("=" * 60)
        print("File:", file.name)

        if file.suffix.lower() == ".csv":
            df = pd.read_csv(file)
            print("Shape:", df.shape)
            print("Columns:", df.columns.tolist())
            print(df.head())

        elif file.suffix.lower() == ".tsv":
            df = pd.read_csv(file, sep="\t")
            print("Shape:", df.shape)
            print("Columns:", df.columns.tolist())
            print(df.head())

        else:
            print("暂不处理这种文件类型。")