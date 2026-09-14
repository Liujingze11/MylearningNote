# microduck_rl 来源说明

本机路径 `07_AI_Robot/microduck_rl/` 是第三方仓库的克隆，**已在外层仓库 `.gitignore` 中忽略**，不随本笔记仓库推送。

## 来源

- 仓库地址: https://github.com/pollen-robotics/microduck_rl
- 分支: `develop`（默认分支，开发活跃）
- 克隆时间: 2026-09-11

## 找回方式

```bash
git clone https://github.com/pollen-robotics/microduck_rl 07_AI_Robot/microduck_rl
cd 07_AI_Robot/microduck_rl && uv sync
```

## 大小说明

- GitHub 上仓库本体约 56 MB（git 对象约 57MB，234 个跟踪文件）
- 本地 7.9 GB 几乎全部来自 `.venv/` 虚拟环境（已被其自身 `.gitignore` 排除，与仓库无关）

## 修改代码注意

- 该仓库无推送权限，如需留存自己的改动需先 fork 到本人账号再改 remote
- 实验产物（`wandb/`、`logdir/`、`artifacts/`、`agents/` 等）不被任何仓库跟踪，重要结果需自行备份
