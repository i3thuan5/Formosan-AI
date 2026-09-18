"""檢查 asr、mt、tts 的 requirements.txt 與 common/gpu-constraints.txt 版本一致。

三個 image 疊在同一個 formosan-ai-gpu layer 上，torch 與 nvidia-* 只裝一次。
服務的 lock 檔如果寫了不同版本，pip 會在服務 layer 重裝一份，
image 就多出好幾 GiB，正式機也要重新 pull，共用就失效了。

    $ python tests/check_lock_consistency.py asr/requirements.txt mt/requirements.txt tts/requirements.txt

規則：constraints 有、服務也有的套件，版本必須相同。服務沒列的套件不算問題
（共用層還是會裝，服務只是沒有直接 import）。

哪些套件算 GPU 套件由 common/gpu_constraints.py 決定，這裡只拿它產生的檔案當標準。
這支腳本也會被 common/Dockerfile 的 gpu stage 複製進 image，讓服務 build 時在
pip install 之前自己檢查一次，所以只用標準函式庫，並用 --constraints 指定位置。
"""
import argparse
import re
import sys
from pathlib import Path

DEFAULT_CONSTRAINTS = Path(__file__).resolve().parent.parent / "common" / "gpu-constraints.txt"
PINNED = re.compile(r"^([A-Za-z0-9._-]+)==([^\s\\]+)", re.MULTILINE)


def read_pins(path):
    with open(path, encoding="utf-8") as f:
        content = f.read()
    pins = {}
    for name, version in PINNED.findall(content):
        pins[name.lower()] = version
    return pins


def main():
    parser = argparse.ArgumentParser(description="檢查 GPU 套件版本是否和共用層一致")
    parser.add_argument("locks", nargs="+", help="要檢查的 requirements.txt")
    parser.add_argument(
        "--constraints",
        default=str(DEFAULT_CONSTRAINTS),
        help="共用層的版本標準（image 裡是 /opt/formosan-ai/gpu-constraints.txt）",
    )
    args = parser.parse_args()

    expected = read_pins(args.constraints)
    if not expected:
        print(f"✗ {args.constraints} 沒有任何套件", file=sys.stderr)
        return 1

    failed = False
    for path in args.locks:
        service = read_pins(path)
        mismatched = {}
        for name, version in expected.items():
            if name in service and service[name] != version:
                mismatched[name] = (version, service[name])
        if mismatched:
            failed = True
            print(f"✗ {path} 有 {len(mismatched)} 個套件和共用層不同：")
            for name, (want, got) in sorted(mismatched.items()):
                print(f"    {name}：共用層是 {want}，這裡是 {got}")
        else:
            print(f"✓ {path}（共用 {len(set(expected) & set(service))} 個套件，版本都一致）")

    if failed:
        print("\n請以 `-c common/gpu-constraints.txt` 重新編譯，"
              "或先改 common/requirements.in 再重編全部。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
