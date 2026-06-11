#!/bin/bash
# 循环运行 llm_test.py --use_llm --model deepseek-chat，结果按编号保存到 0llm/high/

set -e

PROJECT_DIR="/home/lenovo/Git/DeepRL-Group-Cooperation copy"
OUT_DIR="$PROJECT_DIR/0llm/high"
CSV_FILES=(
    "llm_static.csv"
    "llm_statichigh.csv"
    "llm_random.csv"
    "llm_reactive.csv"
    "llm_graphnet.csv"
    "llm_test_results.csv"
)

cd "$PROJECT_DIR"

# 找到下一个编号
n=0
for d in "$OUT_DIR"/*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    [[ "$name" =~ ^[0-9]+$ ]] && (( name > n )) && n=$name
done

echo "========================================="
echo "  循环: python llm_test.py --use_llm --model deepseek-chat"
echo "  输出: 0llm/high/${n+1}/ ..."
echo "  按 Ctrl+C 停止"
echo "========================================="

while true; do
    n=$((n + 1))
    target="$OUT_DIR/$n"

    echo ""
    echo "===== 第 ${n} 轮 ====="
    echo "[运行] python llm_test.py --use_llm --model deepseek-chat"
    echo "----------------------------------------"
    python llm_test.py --use_llm --model deepseek-chat
    echo "----------------------------------------"
    echo "[运行完成]"

    mkdir -p "$target"
    for f in "${CSV_FILES[@]}"; do
        src="$PROJECT_DIR/$f"
        if [ -f "$src" ]; then
            mv "$src" "$target/"
            echo "  已移动: $f"
        fi
    done
    echo "[保存完成] -> 0llm/high/$n/"
done
