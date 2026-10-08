# CLAUDE 規定

## Git 操作規定

Claude Code **毋准**執行下底ê git 指令（會影響 staged 檔案、commit 紀錄、分支、抑是遠端）：

- `git commit`
- `git add`
- `git checkout`
- `git switch`
- `git restore --staged`
- `git reset`（任何形式，含 --soft、--hard、--mixed）
- `git branch -d` / `git branch -D`
- `git push`
- `git pull`
- `git fetch`

**准**執行下底ê git 指令（查看抑是暫存，毋影響紀錄）：

- `git stash`、`git stash pop`、`git stash apply`、`git stash list`
- `git log`、`git show`、`git diff`、`git status`
- `git branch`（列出，毋是建立抑是hìnn-sak）
- `git restore <file>`（working tree only，無 `--staged`）

程式碼修改（Edit、Write、等工具）會用得用。

## Python sir-tái-luh

- `for` khǹg 頭前（`for x in ...:`），毋准用 list comprehension 抑是 generator expression kā `for` khǹg tī 後壁（`[... for x in ...]`）。若欲，ài 有特殊理由，而且經過使用者同意。
