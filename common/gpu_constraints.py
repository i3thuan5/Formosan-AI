"""從 common/requirements.txt 取出 GPU 大型套件，產生 common/gpu-constraints.txt。

    $ python common/gpu_constraints.py           # 重新產生
    $ python common/gpu_constraints.py --check   # 只檢查是否和 requirements.txt 同步（CI 用）

「哪些套件要三個 GPU 服務共用」的定義只在這裡。
不能拿整份 common/requirements.txt 當 constraints：裡面的 numpy 等小套件
各服務需求不同（whisperx 要 numpy>=2.1，f5-tts 要 numpy<=1.26.4），硬鎖會解不出來。
"""
import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCK = HERE / "requirements.txt"
CONSTRAINTS = HERE / "gpu-constraints.txt"

GPU_PACKAGES = re.compile(
    r"^(torch|torchaudio|torchvision|torchcodec|triton|nvidia-[a-z0-9-]+)==([^\s\\]+)",
    re.MULTILINE,
)
HEADER = [
    "# 由 common/gpu_constraints.py 從 common/requirements.txt 產生，不要手改。",
    "# 各服務以 uv pip compile -c common/gpu-constraints.txt 編譯，",
    "# 讓 torch 與 nvidia-* 的版本和 formosan-ai-gpu layer 完全一致。",
]


def render():
    versions = dict(GPU_PACKAGES.findall(LOCK.read_text(encoding="utf-8")))
    if not versions:
        raise SystemExit(f"✗ {LOCK} 找不到任何 GPU 套件")
    lines = list(HEADER)
    for name, version in sorted(versions.items()):
        lines.append(f"{name}=={version}")
    return "\n".join(lines) + "\n", len(versions)


def main():
    parser = argparse.ArgumentParser(description="產生 common/gpu-constraints.txt")
    parser.add_argument("--check", action="store_true", help="不寫檔，只檢查是否同步")
    args = parser.parse_args()

    content, count = render()
    if args.check:
        current = CONSTRAINTS.read_text(encoding="utf-8") if CONSTRAINTS.exists() else ""
        if current != content:
            print("✗ common/gpu-constraints.txt 和 common/requirements.txt 不同步，"
                  "請執行 python common/gpu_constraints.py", file=sys.stderr)
            return 1
        print(f"✓ common/gpu-constraints.txt 與 common/requirements.txt 同步（{count} 個套件）")
        return 0

    CONSTRAINTS.write_text(content, encoding="utf-8")
    print(f"寫出 common/gpu-constraints.txt（{count} 個套件）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
