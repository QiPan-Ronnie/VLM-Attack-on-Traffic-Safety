# GitHub 与 SSH 环境说明（给后续 agent）

本文档用于说明这台本机当前的 GitHub / SSH / git 推送环境，方便后续 agent 继续维护仓库、push 更新、迁移到 AutoDL。

## 1. 当前 GitHub 仓库

- GitHub 用户名：`QiPan-Ronnie`
- 仓库名：`VLM-Attack-on-Traffic-Safety`
- 仓库 URL：
  - HTTPS: `https://github.com/QiPan-Ronnie/VLM-Attack-on-Traffic-Safety`
  - SSH: `git@github.com:QiPan-Ronnie/VLM-Attack-on-Traffic-Safety.git`

## 2. 本地仓库位置

当前已经整理好的、可直接用于 git push 的干净仓库目录：

- `D:\BaiduSyncdisk\USC\2026-Semester 3\USC CSCI 699\projects\codes\VLM-Attack-on-Traffic-Safety`

注意：

- 这个目录是“GitHub 可上传版”
- 它不是原始实验工作区的全部内容
- 原始本地实验仍保留在：
  - `projects/codes/safety_attack_dada_bundle/`
  - `projects/work/baseline_llava/`

## 3. 当前本机 SSH key

本机当前用于 GitHub 的 SSH key：

- 私钥：
  - `C:\Users\14294\.ssh\id_ed25519`
- 公钥：
  - `C:\Users\14294\.ssh\id_ed25519.pub`
- 指纹：
  - `SHA256:2oZQ7w08yyJ+O3yIg7+gS0tcsQAls+JKksORN7w5qq8`

这把 key 已经加到 GitHub 账号中，可用于该账号有权限的仓库。

## 4. 重要说明：这把 key 有 passphrase

这把 SSH key 不是无密码 key，而是**带 passphrase** 的。

这意味着：

- 新终端 / 新会话里，agent 如果直接 `git push`，可能会失败
- 在 push 之前，通常要先把 key 加载到 `ssh-agent`

推荐先执行：

```powershell
ssh-add $HOME\.ssh\id_ed25519
```

如果 `ssh-agent` 还没启动，则执行：

```powershell
Start-Service ssh-agent
ssh-add $HOME\.ssh\id_ed25519
```

如果服务从未配置过，可能还要先：

```powershell
Set-Service ssh-agent -StartupType Manual
Start-Service ssh-agent
ssh-add $HOME\.ssh\id_ed25519
```

## 5. 如何验证 SSH 是否正常

执行：

```powershell
ssh -T git@github.com
```

如果输出类似：

```text
Hi QiPan-Ronnie! You've successfully authenticated, but GitHub does not provide shell access.
```

说明：

- GitHub 已接受当前 SSH key
- 当前终端会话里这把 key 也已可用

## 6. 这次为什么一开始 git push 失败

之前已经定位过一次问题，根因有两层：

1. 这把 key 带 passphrase  
   如果没先 `ssh-add`，自动 push 会失败。

2. Windows 下 `git push` 和终端里手动执行的 `ssh -T`，默认不一定走同一套 SSH 程序  
   在这台机器上，最后是通过显式使用系统 `OpenSSH` 才稳定打通。

因此，这个仓库已经固定了：

```text
core.sshCommand = C:/Windows/System32/OpenSSH/ssh.exe
```

后续在该仓库里继续 push，会比默认配置更稳定。

## 7. 当前仓库的 git 状态

当前仓库已经：

- 初始化 git
- 完成首次提交
- 成功 push 到 GitHub

首次上传对应 commit：

- `aa90ee0c07f684fc60e4814e4c98be73db2863ec`

远端分支：

- `origin/main`

## 8. 当前仓库里上传了什么

当前 GitHub 仓库包含：

- `scripts/`
- `prompts/`
- `configs/`
- `tools/`
- `docs/`
- `results/local_summary/`

也就是说，它适合：

- 继续开发代码
- 继续改 prompt
- 继续改 benchmark 配置
- 在 AutoDL 上 clone 后继续环境搭建和实验

## 9. 当前仓库里没有上传什么

为了避免仓库过大、路径太本地化，以下内容没有上传：

- `hf_cache/`
- 大量原始 `.jsonl` 预测文件
- 长日志 `.log`
- 生成的 attack 图像
- OOM 备份文件
- 暂停状态文件
- 带绝对路径的大型本地 manifest / 产物

这些内容仍然保留在本机，不应被误删。

## 10. 后续 agent 的推荐操作顺序

后续如果需要继续维护这个仓库，建议顺序如下：

1. 进入仓库目录：

```powershell
cd "D:\BaiduSyncdisk\USC\2026-Semester 3\USC CSCI 699\projects\codes\VLM-Attack-on-Traffic-Safety"
```

2. 先验证 SSH：

```powershell
ssh -T git@github.com
```

3. 如果失败，先 `ssh-add`：

```powershell
ssh-add $HOME\.ssh\id_ed25519
```

4. 查看仓库状态：

```powershell
git status
```

5. 修改、提交、推送：

```powershell
git add .
git commit -m "your message"
git push
```

## 11. 给后续 agent 的一句话摘要

这台机器已经配置好 GitHub SSH key，当前项目仓库在：

- `D:\BaiduSyncdisk\USC\2026-Semester 3\USC CSCI 699\projects\codes\VLM-Attack-on-Traffic-Safety`

仓库远端为：

- `git@github.com:QiPan-Ronnie/VLM-Attack-on-Traffic-Safety.git`

当前 key 为：

- `C:\Users\14294\.ssh\id_ed25519`

但这把 key 有 passphrase，所以若 push 失败，优先先做：

```powershell
ssh-add $HOME\.ssh\id_ed25519
```
