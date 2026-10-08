## ADDED Requirements

### Requirement: torch 系列的版本來源是 common/requirements.in

`common/requirements.in` SHALL 只宣告 `torch`、`torchaudio`、`torchvision`、`torchcodec` 四個套件的精確版本（目前 `torch==2.8.0`、`torchaudio==2.8.0`、`torchvision==0.23.0`、`torchcodec==0.7.0`）。`common/requirements.txt` SHALL 由 `uv pip compile --python-version 3.10 --python-platform linux --generate-hashes` 產生，包含解析出的 `nvidia-*-cu12` 與 `triton` 版本與 hash。

`common/gpu-constraints.txt` SHALL 由 `python common/gpu_constraints.py` 從 `common/requirements.txt` 產生，只含 torch 系列、`nvidia-*`、`triton`。MUST NOT 直接拿 `common/requirements.txt` 當各服務的 constraints：它含 numpy 等小套件，而 whisperx 要 `numpy>=2.1.0`、f5-tts 要 `numpy<=1.26.4`，硬鎖會解不出來。

#### Scenario: 產生 common lock 檔

- **WHEN** 執行 `uv pip compile common/requirements.in --python-version 3.10 --python-platform linux --generate-hashes -o common/requirements.txt`
- **THEN** 輸出檔每個套件都有 `--hash=sha256:` 行，且 `nvidia-cudnn-cu12` 為 9.x、`nvidia-cuda-runtime-cu12` 為 12.8.x

#### Scenario: 產生 constraints

- **WHEN** 執行 `python common/gpu_constraints.py`
- **THEN** 產生的 `common/gpu-constraints.txt` 只含 torch、torchaudio、torchvision、torchcodec、triton 與 `nvidia-*`，不含 numpy

### Requirement: GPU 服務的 lock 檔與 common 一致

asr、mt、tts 的 `requirements.txt` SHALL 以 `-c common/gpu-constraints.txt` 編譯。`common/gpu-constraints.txt` 中的套件若出現在服務的 `requirements.txt`，版本 MUST 相同；服務沒有列出的套件不算問題，共用層仍會安裝。

#### Scenario: 服務以 constraints 編譯

- **WHEN** 執行 `uv pip compile mt/requirements.in --python-version 3.10 --python-platform linux --no-strip-extras -c common/gpu-constraints.txt -o mt/requirements.txt`
- **THEN** `mt/requirements.txt` 的 torch 為 2.8.0，`nvidia-*-cu12` 與 `triton` 版本與 `common/requirements.txt` 完全相同

#### Scenario: 上游硬 pin 與 common 衝突時編譯失敗

- **WHEN** `common/requirements.in` 改為 `torch==2.9.1` 後重新編譯 asr（whisperx 要求 `torch~=2.8.0`）
- **THEN** uv 回報無法解析的錯誤，不產生新的 `asr/requirements.txt`

#### Scenario: 服務沒列出共用層的某些套件

- **WHEN** mt 的 lock 檔沒有 torchaudio、torchvision、torchcodec（mt 沒有直接 import 它們）
- **THEN** 一致性檢查通過，因為共用層仍會安裝這些套件

### Requirement: 一致性檢查在 CI 與 build 兩層執行

產生與檢查 SHALL 分成兩支檔案：`common/gpu_constraints.py` 是「哪些套件算 GPU 套件」的唯一定義，負責產生 `common/gpu-constraints.txt`，加 `--check` 時只檢查它與 `common/requirements.txt` 是否同步；`tests/check_lock_consistency.py` 以 `common/gpu-constraints.txt` 為標準比對指定的服務 lock 檔，兩邊都有但版本不同時 MUST 以非零 exit code 結束並列出差異，且只用標準函式庫。tox SHALL 提供 `lockcheck` env 依序執行這兩項檢查，Travis SHALL 有對應 job。`gpu` stage SHALL 將檢查腳本與 `gpu-constraints.txt` 複製到 `/opt/formosan-ai/`，GPU 服務的 Dockerfile MUST 在 `pip install` 之前以 `--constraints /opt/formosan-ai/gpu-constraints.txt` 執行同一檢查。

#### Scenario: 全部一致時通過

- **WHEN** 三個服務的 lock 檔皆以 `-c common/requirements.txt` 編譯後執行 `tox -e lockcheck`
- **THEN** exit code 為 0，輸出列出比對的套件數

#### Scenario: 版本不一致時 CI 失敗

- **WHEN** 手動把 `tts/requirements.txt` 的 `torch==2.8.0` 改為 `torch==2.7.0` 後執行 `tox -e lockcheck`
- **THEN** exit code 非 0，輸出指出 `torch：common 是 2.8.0，這裡是 2.7.0`

#### Scenario: 版本不一致時本機 build 失敗

- **WHEN** 以同樣被改壞的 `tts/requirements.txt` 執行 `docker build ./tts`
- **THEN** build 在 `pip install` 之前的檢查步驟失敗，不會產生含兩份 torch 的 image

### Requirement: hash 鎖定範圍

`common/requirements.txt` 與 `asr/requirements.txt` SHALL 含 hash，對應的 `pip install` MUST 使用 `--require-hashes`。mt、tts、asr-kaldi 的 `requirements.txt` 由 uv 產生但不含 hash，其 `pip install` MUST NOT 使用 `--require-hashes`。

#### Scenario: gpu stage 驗證 hash

- **WHEN** build `common` 的 `gpu` target
- **THEN** pip 以 hash 檢查模式安裝 torch 系列與 nvidia wheel，任一檔案 hash 不符時 build 失敗

#### Scenario: asr 的 hash 涵蓋非 torch 套件

- **WHEN** build asr image
- **THEN** torch 系列因已安裝而略過，其餘套件皆經 hash 驗證；`--only-binary=:all:` 與 `antlr4-python3-runtime`、`julius` 的例外維持不變

### Requirement: 套件更新流程

更新任一服務的直接依賴時，維護者 SHALL 修改該服務的 `requirements.in`，並以 `uv pip compile`（GPU 服務加 `-c common/gpu-constraints.txt`）重新產生 `requirements.txt`。升級 torch 時 SHALL 先修改 `common/requirements.in`、重編 `common/requirements.txt`、重新產生 `common/gpu-constraints.txt`，再重編三個 GPU 服務。README SHALL 記載此流程與三個 target 的 build 順序。

#### Scenario: 更新單一服務的直接依賴

- **WHEN** 修改 `mt/requirements.in` 的 gradio 版本並重新編譯
- **THEN** `mt/requirements.txt` 的 torch 系列版本不變，`tox -e lockcheck` 通過，只需重建 `ithuan/formosan-ai:mt`

#### Scenario: 升級 torch

- **WHEN** 修改 `common/requirements.in` 的 torch 版本，重編 common 後再重編 asr、mt、tts
- **THEN** 三個服務的 lock 檔 torch 版本同步改變，`tox -e lockcheck` 通過，需重建 `formosan-ai-gpu` 與三個 GPU 服務 image，`formosan-ai-common` 與 asr-kaldi 不需重建
