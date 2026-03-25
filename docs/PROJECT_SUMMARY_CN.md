# CSCI 699 项目总总结与交接文档（中文）

更新时间：2026-03-24  
适用范围：本地代码、已跑实验、队友截图结论、下一阶段建议  
本文档目标：让后续任何 agent / 队友接手时，都能快速知道“这个项目为什么做、已经做了什么、各实验设置是什么、当前卡在哪里、下一步应该怎么接着做”。

---

## 0. 先看这三句话

1. 这个项目的目标，是做一个面向事故前帧预警任务的 **VLM overlay attack benchmark**。  
2. 当前本地已经把 `DADA -> attack benchmark -> baseline -> eval` 这条链路跑通，并且已经找到一个当前最可用的 LLaVA-family 候选：`LLaVA-OneVision Qwen2 7B`。  
3. 当前的主要瓶颈已经不再是“模型接不进来”，而是“当前 attack 设计和 prompt 还不够强、不够稳定”。

---

## 1. 项目目标与研究问题

### 1.1 我们到底想证明什么

本项目想研究的是：

- 对于 **事故发生前的关键视频帧**，VLM 在 `clean` 条件下能否判断“未来约 1 秒内会发生事故”。
- 当这些关键帧加入视觉干扰后，例如：
  - `text_watermark`
  - `timestamp_box`
  - `logo_patch`
  - `semi_transparent_bar`
  - 后续可能扩展到 `mosaic / blur / targeted occlusion`
- 模型是否会出现：
  - 本来该报 `yes`，攻击后变成 `no`
  - 或者预警明显变晚

这意味着本项目不是在训练一个新模型，而是在做一个 **针对现有视觉模型的鲁棒性 benchmark**。

### 1.2 当前阶段的真正目标

当前阶段不是最终论文收尾阶段，而是 **pilot baseline bring-up + victim model identification** 阶段。

当前阶段要完成的是：

1. 数据与 benchmark 跑通  
2. baseline 与 eval 跑通  
3. 找到至少一个 clean 上“有一定能力”的 victim model  
4. 初步验证 overlay attack 是否能让它退化  

当前阶段不要求一次性得到最终论文最强结果，但必须把实验链路和判断逻辑建立起来。

---

## 2. 证据来源说明

后续接手时一定要区分证据来源，避免把“本地已验证结果”和“队友口头/截图结论”混在一起。

| 证据类型 | 说明 | 当前可信度 |
| --- | --- | --- |
| 本地可复现实验 | 有代码、manifest、predictions、eval 文件 | 最高 |
| 队友截图结论 | 目前只有图片，没有本地原始结果文件 | 中等，适合写“已观察到/队友报告” |
| 计划或口头结论 | 例如 work plan 里提到某模型有 `all-yes` 倾向 | 中等偏低，需要后续原始结果补证 |

本文档里会明确写出哪些是“本地结果”，哪些是“队友截图给出的结果”。

---

## 3. 数据与 benchmark 设置

### 3.1 数据来源

- 原始数据集：`DADA`
- 当前使用的数据，不是从零开始重新抽帧，而是基于队友已整理好的 clean manifest
- 路径根目录：
  - `projects/datasets/master_pack/master_pack`

### 3.2 当前 benchmark 版本

当前本地 pilot benchmark 的 clean/attack 配置如下：

| 项目 | 当前配置 |
| --- | --- |
| clean 帧数 | `6935` |
| total rows | `62415` |
| overlay 类型 | `text_watermark`, `timestamp_box`, `logo_patch`, `semi_transparent_bar` |
| severity | `2`, `4` |
| placement | `random` |
| include clean | 是 |
| 当前定位 | pilot benchmark |

公式如下：

```text
6935 clean
+ 6935 × 4 overlays × 2 severities × 1 placement
= 62415 rows
```

### 3.3 标签含义

本项目当前最重要的标签是：

- `label_risk = 1`：这帧属于“事故即将在约 1 秒内发生”的正样本
- `label_risk = 0`：不是该任务的正样本

目前 full clean `6935` 中：

- positive：`5549`
- negative：`1386`

这意味着数据集是明显偏向正样本的，所以“全预测 no”时 accuracy 会很低。

---

## 4. 代码与运行入口

### 4.1 核心脚本

| 脚本 | 作用 |
| --- | --- |
| `scripts/03b_add_attention_roi_from_maps.py` | 从 attention map 提取 ROI |
| `scripts/04_build_safety_attack_benchmark.py` | 生成 overlay attack benchmark |
| `scripts/05_run_clip_attack_baseline.py` | 跑 CLIP baseline |
| `scripts/06_run_vlm_attack_hf.py` | 跑 Hugging Face VLM/LLaVA |
| `scripts/07_eval_safety_attack.py` | 统一计算 clean / frame ASR / delay 等指标 |

### 4.2 已做过的关键工程改动

为了让 LLaVA / OneVision 在本机上真正能跑，本地已经做过这些改动：

- `06_run_vlm_attack_hf.py` 已支持：
  - `--cache-dir`
  - `--local-files-only`
  - `--clear-cuda-cache`
  - `--load-in-4bit`
  - `--skip-existing`
  - `--log-file`
- 已支持自动从本地 HF cache 解析模型 snapshot
- 已支持 `4-bit NF4` 加载
- 已支持长日志和断点续跑

### 4.3 当前 prompt 文件

已经试过的 prompt 版本：

| prompt 文件 | 用途 |
| --- | --- |
| `prompts/binary_imminent_risk_prompt.txt` | 原始版本 |
| `prompts/binary_imminent_risk_prompt_v2.txt` | 更激进的二分类试探版 |
| `prompts/binary_imminent_risk_prompt_v3_json.txt` | 当前主用 JSON 输出版 |

当前 OneVision 7B 主线使用的是：

- `prompts/binary_imminent_risk_prompt_v3_json.txt`

补充更新（2026-03-24）：

- 旧主线 `prompts/binary_imminent_risk_prompt_v3_json.txt` 的 full clean `6935` 运行已经保留并暂停，停在 `258 / 6935`
- 新主线已经切换到：
  - `projects/codes/safety_attack_dada_bundle/safety_attack_dada/prompts/binary_imminent_risk_prompt.txt`
- 当前新的 OneVision full clean 运行标签为：
  - `prompt_r1_20260324`
- 新旧 prompt 的角色区分：
  - `binary_imminent_risk_prompt_v3_json.txt`：上一轮主用 prompt，用于旧的 OneVision probe / clean100 / attack80 / attack160 / paused clean6935
  - `binary_imminent_risk_prompt.txt`：当前最新 prompt，用于重新启动 OneVision full clean `6935` 评测

---

## 5. 本地可复现的实验时间线

### 5.1 阶段时间线

| 阶段 | 模型 / 对象 | 数据范围 | 结果结论 |
| --- | --- | --- | --- |
| 环境打通 | CUDA + PyTorch + HF | 本机 | 完成 |
| pilot benchmark | clean + attacks | `62415` rows | 完成 |
| CLIP baseline | CLIP | full pilot clean `6935` + full pilot attack | clean 上最稳定 |
| LLaVA 0.5B | OneVision 0.5B | subset clean `100` | 几乎不可用 |
| LLaVA 1.5 7B | old LLaVA 7B | subset clean `100` | 几乎不可用 |
| OneVision 7B probe | OneVision 7B 4-bit | clean probe `20` | 首次出现非塌缩 yes/no |
| OneVision clean100 | OneVision 7B 4-bit | balanced clean `100` | 当前最可用 LLaVA-family 候选 |
| attack subset 80 | OneVision 7B 4-bit | attacked `80` | severity 4 才开始有效 |
| severity4 all 160 | OneVision 7B 4-bit | attacked `160` | 随机攻击仍偏弱，但有效 |
| full clean 6935 | OneVision 7B 4-bit | full clean `6935` | 已启动后暂停，保留 `258` 条 |

---

## 6. 本地实验详细结果

### 6.1 CLIP full clean（本地结果）

结果文件：

- `projects/work/baseline_llava/clip/eval/clean_metrics.json`

clean 指标：

| 指标 | 值 |
| --- | ---: |
| `n` | `6935` |
| `parse_success_rate` | `1.0` |
| `risk_accuracy` | `0.7384` |
| `risk_f1` | `0.8452` |
| `false_negative_rate` | `0.1072` |
| `false_positive_rate` | `0.8795` |

解释：

- CLIP 在 clean 上是当前本地最强 baseline
- 它说明数据集和评测链路本身没有坏
- 但 CLIP 不是本项目唯一目标，因为项目更关心 VLM / LLaVA 风格模型

### 6.2 LLaVA-OneVision 0.5B（本地结果）

结果文件：

- `projects/work/baseline_llava/llava_0p5b/eval/clean_metrics.json`

clean 指标：

| 指标 | 值 |
| --- | ---: |
| `n` | `100` |
| `risk_accuracy` | `0.5` |
| `risk_f1` | `0.0` |
| `false_negative_rate` | `1.0` |

解释：

- 原始 prompt 下几乎全 `no`
- 换 prompt 后又容易塌到接近全 `yes`
- 说明 0.5B 极度 prompt-sensitive，不适合作为主 victim baseline

### 6.3 LLaVA 1.5 7B（本地结果）

结果文件：

- `projects/work/baseline_llava/llava_7b/eval/clean_metrics.json`

clean 指标：

| 指标 | 值 |
| --- | ---: |
| `n` | `100` |
| `risk_accuracy` | `0.5` |
| `risk_f1` | `0.0` |
| `false_negative_rate` | `1.0` |

解释：

- 旧 LLaVA 1.5 7B 在这个任务上基本接近“常量输出”
- prompt 改进后也没有明显救回来
- 可以认为它在当前任务设定下不是合格的 victim baseline

### 6.4 OneVision 7B clean100（本地结果）

结果文件：

- `projects/work/baseline_llava/llava_onevision_7b/clean_only_100_preds.jsonl`
- `projects/codes/safety_attack_dada_bundle/logs/llava_onevision_7b_clean100_4bit.log`

clean100 指标：

| 指标 | 值 |
| --- | ---: |
| `n` | `100` |
| `accuracy` | `0.51` |
| `precision` | `0.5063` |
| `recall` | `0.8` |
| `f1` | `0.6202` |
| `false_negative_rate` | `0.2` |
| `false_positive_rate` | `0.78` |

解释：

- 这是当前最关键的一步
- 它说明 OneVision 7B 已经不再是“全 yes / 全 no”塌缩
- 它终于具备了“本来会报一部分风险，因此值得测试攻击是否让它失效”的条件
- 但 FPR 仍然偏高，所以它是 **可用的 preliminary victim baseline**，不是最终最优版本

### 6.5 OneVision 7B attack subset 80（本地结果）

结果文件：

- `projects/work/baseline_llava/llava_onevision_7b/attack_subset_80_metrics.json`
- `projects/work/baseline_llava/llava_onevision_7b/attack_subset_80_group_metrics.csv`

设置：

- 从 clean100 中筛出 `40` 个 clean-positive candidate base frames
- 每个 `(overlay, severity)` 取 `10` 条
- 总计 `80` attacked rows

关键结果：

| 设置 | ASR |
| --- | ---: |
| overall | `0.1` |
| `logo_patch`, severity `2` | `0.0` |
| `logo_patch`, severity `4` | `0.2` |
| `semi_transparent_bar`, severity `2` | `0.0` |
| `semi_transparent_bar`, severity `4` | `0.1` |
| `text_watermark`, severity `2` | `0.0` |
| `text_watermark`, severity `4` | `0.3` |
| `timestamp_box`, severity `2` | `0.0` |
| `timestamp_box`, severity `4` | `0.2` |

解释：

- `severity 2` 几乎没有价值
- `severity 4` 才开始出现真正可测的攻击效果

### 6.6 OneVision 7B severity4 all 160（本地结果）

结果文件：

- `projects/work/baseline_llava/llava_onevision_7b/attack_severity4_all_160_metrics.json`
- `projects/work/baseline_llava/llava_onevision_7b/attack_severity4_all_160_group_metrics.csv`

设置：

- 只保留 `severity = 4`
- 使用全部 `40` 个 clean-positive candidate base frames
- `4 overlays × 40 frames = 160 attacked rows`

关键结果：

| 设置 | ASR |
| --- | ---: |
| overall | `0.0625` |
| `logo_patch` | `0.075` |
| `semi_transparent_bar` | `0.025` |
| `text_watermark` | `0.075` |
| `timestamp_box` | `0.075` |

解释：

- 攻击在大样本下仍然有效，但总体并不强
- 这说明当前瓶颈开始转到 **attack placement / attack strength**
- 也说明不能只靠“更大样本”解决问题，后续必须改攻击设计

### 6.7 OneVision 7B full clean 6935（本地结果，当前已暂停）

原计划：

- 对 full clean `6935` 全量跑 `OneVision 7B 4-bit`
- 使用当前主 prompt：`binary_imminent_risk_prompt_v3_json.txt`

当前状态：

- 已启动并稳定运行
- 后由当前 agent 手动暂停
- 保留全部已完成结果，不删除、不覆盖

暂停状态文件：

- `projects/work/baseline_llava/llava_onevision_7b/clean_only_6935_pause_state.json`

当前停点：

| 项目 | 值 |
| --- | --- |
| 状态 | `paused` |
| 暂停时间 | `2026-03-24T00:17:40` |
| 已完成条数 | `258 / 6935` |
| 最后日志行 | `START 259/6935 ... unspecified_4_3_0036__clean` |

相关文件：

- manifest：`projects/work/baseline_llava/llava_onevision_7b/clean_only_6935.csv`
- partial preds：`projects/work/baseline_llava/llava_onevision_7b/clean_only_6935_preds.jsonl`
- latest run info：`projects/work/baseline_llava/llava_onevision_7b/clean_only_6935_latest_run.json`
- pause state：`projects/work/baseline_llava/llava_onevision_7b/clean_only_6935_pause_state.json`
- log：`projects/codes/safety_attack_dada_bundle/logs/llava_onevision_7b_clean6935_20260323_190916.log`

说明：

- 这条 full clean 任务只是“暂停”，不是“废弃”
- 以后如果需要恢复，可以继续基于当前 `preds.jsonl` + `--skip-existing` 续跑

---

## 7. 队友提供的外部结果与结论

这一节不是本地原始结果文件，而是当前对话中用户提供的截图内容。  
后续如果队友给出原始 `predictions / eval csv / metrics json`，应以原始文件替换本节的截图摘要。

### 7.1 DeepSeek-VL-1.3B（来自队友截图）

截图明确显示的 clean 基线指标：

| 指标 | 值 |
| --- | ---: |
| Accuracy | `21.5%` |
| Recall | `2.5%` |
| Precision | `80.5%` |
| TP | `136` |
| TN | `1353` |
| FP | `33` |
| FN | `5413` |
| Parse rate | `3%` |

截图摘要给出的结论：

- `DeepSeek-VL-1.3B` 太小，事故检测能力极弱
- 几乎不回答 `yes`
- JSON 输出遵循能力很差，导致 parse rate 极低
- 在模型有限的检测能力范围内，overlay attack 仍然有效
- frame ASR 最高可达 `80%`
- strongest attack 排名：
  - `timestamp_box`
  - `logo_patch`
  - `semi_transparent_bar`
  - `text_watermark`
- `severity 4` 明显强于 `severity 2`
- placement 效应不明显
- 因 clean baseline 太弱，delay 指标无法得出有效结论

如何理解这组结果：

- 它很适合作为“小模型 baseline”
- 它不适合作为“强 victim model”
- 它可以支撑的论点是：
  - 小模型本身 clean 能力不足
  - 即使只有有限 clean 能力，也会被简单 overlay 明显绕过

### 7.2 Qwen 结果（当前信息不完整）

当前本地没有找到队友 Qwen 的原始输出文件。  
目前只有两类可引用的信息：

1. README / bundle 推荐过的参考模型：`Qwen/Qwen2.5-VL-3B-Instruct`
2. 现有计划文档里有一句观察：
   - `Qwen2.5-VL-3B` 在现有 work plan 中表现出明显 `all-yes` 倾向，更适合作为对照，而不适合作为唯一 attack target

因此当前对 Qwen 的写法必须保守：

- 可以写“队友已有 Qwen 方向实验”
- 可以写“现阶段观察到其存在明显偏置倾向”
- 但如果没有原始 metrics 文件，不建议把 Qwen 的具体数字写死到最终论文主表中

### 7.3 用户额外提供的一张未标注 confusion matrix 截图

用户还提供了一张 confusion matrix 风格截图，内容为：

- `TP = 0`
- `FN = 5549`
- `FP = 0`
- `TN = 1386`

这张图没有在截图中明确标注模型名。  
因为它与 DeepSeek 那组 `TP=136 / FP=33 / TN=1353 / FN=5413` 不一致，所以目前 **不应直接把它强归因到 DeepSeek 或 Qwen**。  
如果后续能确认来源，再补回主表。

### 7.4 与队友 DeepSeek 口径对齐的统一数据表

为了后续和队友统一口径，本地已经额外生成了两份对齐文件：

- `projects/work/baseline_llava/model_alignment_with_teammates.csv`
- `projects/work/baseline_llava/model_alignment_with_teammates.md`

这两份文件统一使用了下面这套字段：

- `Parse rate`
- `Accuracy`
- `Precision`
- `Recall`
- `F1`
- `TP / TN / FP / FN`
- `Confusion Matrix`

从 2026-03-24 开始，本地评测脚本 `scripts/07_eval_safety_attack.py` 已经被补齐为固定导出这套口径。  
也就是说，后续任何 OneVision / LLaVA full-clean 或 attack 评测，原则上都应至少产出：

- `clean_metrics.json`
  - 包含：`parse_success_rate`、`risk_accuracy`、`risk_precision`、`risk_recall`、`risk_f1`、`tp`、`tn`、`fp`、`fn`
- `clean_confusion_matrix.csv`
  - clean 条件下的 2x2 confusion matrix
- `summary_condition_metrics.csv`
  - 各条件下的 `accuracy / precision / recall / F1 / TP / TN / FP / FN`
- `summary_condition_confusion_matrix.csv`
  - 各条件对应的 confusion matrix 平铺表

当前可直接引用的对齐表如下：

| 模型 | 数据来源 | 评测范围 | Parse rate | Accuracy | Precision | Recall | TP | TN | FP | FN | 是否可直接对齐 DeepSeek |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DeepSeek-VL-1.3B | 队友截图 | full clean `6935` | `0.03` | `0.2150` | `0.8050` | `0.0250` | `136` | `1353` | `33` | `5413` | 是（基准行） |
| CLIP ViT-B-32 | 本地结果 | full clean `6935` | `1.0` | `0.7384` | `0.8025` | `0.8928` | `4954` | `167` | `1219` | `595` | 是 |
| LLaVA-OneVision-0.5B | 本地结果 | subset clean `100` | `1.0` | `0.5000` | `0.0000` | `0.0000` | `0` | `50` | `0` | `50` | 否 |
| LLaVA-1.5-7B | 本地结果 | subset clean `100` | `1.0` | `0.5000` | `0.0000` | `0.0000` | `0` | `50` | `0` | `50` | 否 |
| LLaVA-OneVision-Qwen2-7B | 本地结果 | balanced clean `100` | `1.0` | `0.5100` | `0.5063` | `0.8000` | `40` | `11` | `39` | `10` | 否 |
| LLaVA-OneVision-Qwen2-7B | 本地 partial 结果 | full clean 前 `258` 条 | `0.0372` | `0.6744` | `0.8211` | `0.7573` | `156` | `18` | `34` | `50` | 仅部分可参考 |

如何解读这张表：

- 目前真正能和队友 `DeepSeek full clean 6935` 直接对齐比较的，本地只有：
  - `CLIP full clean 6935`
- `OneVision 7B` 当前虽然已经开始跑 full clean，但只完成了前 `258` 条，因此还不能作为严格的 full-clean 对齐结果
- `0.5B / 1.5 7B / OneVision clean100` 都属于阶段性小规模验证，适合解释模型行为，不适合和队友 full-clean 主表直接并排作为最终结论

### 7.5 新 prompt 的 OneVision full clean 主线（进行中）

当前正在跑的新主线是：

- 模型：`llava-hf/llava-onevision-qwen2-7b-ov-hf`
- prompt：`projects/codes/safety_attack_dada_bundle/safety_attack_dada/prompts/binary_imminent_risk_prompt.txt`
- 数据：`projects/work/baseline_llava/llava_onevision_7b/clean_only_6935.csv`
- 输出：`projects/work/baseline_llava/llava_onevision_7b/clean_only_6935_prompt_r1_20260324_preds.jsonl`
- 状态文件：`projects/work/baseline_llava/llava_onevision_7b/clean_only_6935_prompt_r1_20260324_latest_run.json`
- 日志：`projects/codes/safety_attack_dada_bundle/logs/llava_onevision_7b_clean6935_prompt_r1_20260324_20260324_035017.log`

这一轮的目标不是先看 attack，而是先重新确认：

1. 新 prompt 下的 parse rate 是否稳定  
2. 新 prompt 下的 clean full `6935` 能力是否明显优于旧 prompt 主线  
3. 最终能否产出一套和队友完全同口径的 `Accuracy / Precision / Recall / F1 / TP / TN / FP / FN / Confusion Matrix`

截至本次会话的 smoke 自检（只基于已完成的前 `4` 条预测）：

- `parse_success_rate = 0.0005768`
- `n_eval = 4`
- `accuracy = 0.75`
- `precision = 1.0`
- `recall = 0.6667`
- `f1 = 0.8`
- `TP = 2`
- `TN = 1`
- `FP = 0`
- `FN = 1`

注意：

- 这只是“前 4 条”的 smoke 结果，只能说明新 prompt 这轮没有立刻塌到全 `yes` 或全 `no`
- 这个数值 **绝对不能** 当成正式 clean `6935` 结果引用
- 正式对齐队友时，必须等 full clean 运行完成后，再读取这一轮的完整 `clean_metrics.json + clean_confusion_matrix.csv`

补充更新（2026-03-24 中期快照）：

截至当前会话中期检查，这一轮新 prompt 主线已经完成到 `447 / 6935`，并重新做了一次 partial clean eval。

当前 partial clean `447` 的结果为：

- `coverage = 447 / 6935 = 6.45%`
- `parse_success_rate_over_manifest = 0.0645`
- `parse_success_rate_over_completed = 1.0`
- `accuracy = 0.6130`
- `precision = 0.7973`
- `recall = 0.6818`
- `f1 = 0.7351`
- `TP = 240`
- `TN = 34`
- `FP = 61`
- `FN = 112`

中期解读：

- 这说明新 prompt 下的 OneVision 7B **已经明显不是“全 no / 全 yes 塌缩”**
- 它在当前已完成样本中，确实抓到了大量正样本，说明它具备成为 victim baseline 的基础能力
- 但它目前仍然存在明显误报问题：
  - 负样本里 `FP = 61`，`TN = 34`
  - 也就是当前 `false_positive_rate` 仍然偏高
- 因此，这一轮中期状态更接近：
  - “已经是一个可用的 VLM victim candidate”
  - 但“还没有达到像 CLIP 那样稳定的 clean baseline”

与旧 prompt 暂停时的 partial `258` 相比：

- 旧 prompt partial `258`：`accuracy = 0.6744`，`precision = 0.8211`，`recall = 0.7573`，`f1 = 0.7879`
- 新 prompt 在“前 258 条”上的公平对比结果：`accuracy = 0.6512`，`precision = 0.8187`，`recall = 0.7233`，`f1 = 0.7680`

这说明：

- 新 prompt 目前 **没有明显优于** 旧 prompt 的 clean 检测能力
- 但新 prompt 这轮到目前为止运行稳定、解析稳定，且没有重新塌到旧的小模型问题
- 因此它值得继续跑满；但如果最终 full clean 结果仍然弱于旧 prompt，就要认真考虑继续改 prompt，而不是直接拿它进入大规模 attack 主实验

补充更新（2026-03-24 晚些时候）：

- 这轮 `prompt_r1_20260324` 后续并没有“真实地快速跑完”
- 实际情况是：
  - 在第 `488` 条样本处首次出现 `CUDA out of memory`
  - 之后脚本没有自动停下，而是把后续剩余样本全部以 `status=error` 的形式秒级写入结果文件
- 因此结果文件虽然出现了 `6935 / 6935` 行，但其中只有：
  - `ok = 487`
  - `error = 6448`
  - 且这 `6448` 条 error 全部是 OOM 连锁失败

这意味着：

- 这一轮 **不能** 被当成“full clean 6935 已完成”
- 当前真正可用的有效 clean 结果，只到前 `487` 条左右
- 后续如果要恢复这轮评测，应先清理这次 OOM 之后的 error 尾部，再从第 `488` 条附近继续

工程修复说明：

- `scripts/06_run_vlm_attack_hf.py` 已在本地补丁中加入“遇到 fatal CUDA/OOM 后立刻停止”逻辑
- 修复后，未来再遇到类似 OOM，不会再把后面几千条样本误写成“已完成 error”，而会直接停表，便于安全续跑

恢复策略（当前已执行）：

- 原始 OOM 全量文件已保留备份：
  - `projects/work/baseline_llava/llava_onevision_7b/clean_only_6935_prompt_r1_20260324_preds_oom_fullcopy.jsonl`
- 可用前缀 `487` 条已单独抽出：
  - `projects/work/baseline_llava/llava_onevision_7b/clean_only_6935_prompt_r1_20260324_prefix_ok487.jsonl`
- 从第 `488` 条开始的 resume manifest 已生成：
  - `projects/work/baseline_llava/llava_onevision_7b/clean_only_6935_prompt_r1_20260324_resume_from_488.csv`
- 新的续跑输出文件：
  - `projects/work/baseline_llava/llava_onevision_7b/clean_only_6935_prompt_r1_20260324_resume_from_488_preds.jsonl`
- 新的续跑状态文件：
  - `projects/work/baseline_llava/llava_onevision_7b/clean_only_6935_prompt_r1_20260324_resume_from_488_latest_run.json`
- 新的续跑日志：
  - `projects/codes/safety_attack_dada_bundle/logs/llava_onevision_7b_clean6935_prompt_r1_20260324_resume_from_488_*.log`

后续合并方式：

1. 保留 `prefix_ok487.jsonl` 作为 clean full 结果的前缀  
2. 等 `resume_from_488_preds.jsonl` 跑完后，将其拼接在后面  
3. 不再使用那份带有 `6448` 条 OOM error 尾部的伪 full 文件做正式评测

---

## 8. 当前结论

### 8.1 已经确定的结论

1. 本地 benchmark 工程链路已经跑通  
2. `CLIP` clean 上表现最稳定  
3. 旧的 LLaVA 0.5B / 1.5 7B 在当前任务设定下不适合作为主 victim baseline  
4. `LLaVA-OneVision Qwen2 7B` 是当前本地最有希望的 LLaVA-family victim baseline  
5. overlay attack 已经在 OneVision 上产生可测退化，但当前随机攻击整体仍然偏弱  

### 8.2 当前最重要的判断

当前最大问题已经不是“能不能把模型跑起来”，而是下面两个：

1. prompt 还需要重做  
2. attack 还需要更强设计

这和当前本地结果、队友 DeepSeek 截图结论，是一致的。

换句话说：

- `victim model` 侧：已经从“完全不可用”进展到“有一个可工作的 preliminary baseline”
- `attack` 侧：已经从“有没有作用”进展到“作用存在，但还不够强、不够有说服力”

---

## 9. 为什么现在要改 prompt

当前决定暂停 full clean `6935`，转去讨论和重做 prompt，是合理的。原因如下：

1. 旧 LLaVA 系列已经证明在当前 prompt 下效果不好  
2. OneVision 7B 虽然可用，但 clean FPR 偏高，说明 prompt / 输出格式还有优化空间  
3. 如果 prompt 改了，full clean `6935` 继续跑旧 prompt 的价值会下降  
4. 因此保留当前 `258` 条 partial results，先停，再改 prompt，是更合理的实验管理方式

---

## 10. 后续建议给下一个 agent

### 10.1 第一优先级

先做 **prompt redesign**，再恢复大规模运行。

建议直接从以下文件开始：

- `projects/codes/safety_attack_dada_bundle/safety_attack_dada/prompts/binary_imminent_risk_prompt.txt`
- `projects/codes/safety_attack_dada_bundle/safety_attack_dada/prompts/binary_imminent_risk_prompt_v2.txt`
- `projects/codes/safety_attack_dada_bundle/safety_attack_dada/prompts/binary_imminent_risk_prompt_v3_json.txt`

### 10.2 prompt redesign 建议

下一步建议做一个小规模 prompt ablation：

| 维度 | 建议 |
| --- | --- |
| 输出格式 | 保留 JSON，但尽量更短、更刚性 |
| reason 字段 | 可以考虑先去掉，减少生成负担 |
| 风险定义 | 明确“未来 1 秒内事故是否开始”，避免太模糊 |
| 保守偏置 | 明确不要因为场景正常就默认 `no`，也不要因为复杂就默认 `yes` |
| score 使用 | 明确 `risk_score` 与 `risk` 的一致性要求 |

### 10.3 attack 侧建议

在 prompt 确定后，再继续：

1. 恢复 full clean `6935`
2. 从 full clean 中筛出真正的 clean-positive candidates
3. 主力攻击优先跑：
   - `severity 4`
   - `random` 作为最低限度主线
4. 如果随机攻击仍然弱，优先做：
   - `critical placement`
   - 更贴近 ROI 的 targeted overlay
   - 更强遮挡

### 10.4 当前最不建议做的事

当前不建议：

- 继续扩大 `severity 2` 攻击
- 在旧 prompt 上把 full clean `6935` 一口气全跑完
- 在没有队友原始文件的情况下，把 Qwen 具体数字写进最终定稿主表

---

## 11. 当前关键文件清单

### 11.1 总状态与实验记录

- `projects/work/baseline_llava/STATUS.md`
- `projects/work/baseline_llava/EXPERIMENT_LOG.md`
- `projects/work/baseline_llava/PROJECT_SUMMARY_CN.md`

### 11.2 OneVision 主线

- `projects/work/baseline_llava/llava_onevision_7b/clean_only_100_preds.jsonl`
- `projects/work/baseline_llava/llava_onevision_7b/attack_subset_80_preds.jsonl`
- `projects/work/baseline_llava/llava_onevision_7b/attack_severity4_all_160_preds.jsonl`
- `projects/work/baseline_llava/llava_onevision_7b/attack_severity4_all_160_metrics.json`
- `projects/work/baseline_llava/llava_onevision_7b/clean_only_6935_preds.jsonl`
- `projects/work/baseline_llava/llava_onevision_7b/clean_only_6935_latest_run.json`
- `projects/work/baseline_llava/llava_onevision_7b/clean_only_6935_pause_state.json`
- `projects/work/baseline_llava/llava_onevision_7b/clean_only_6935_prompt_r1_20260324_preds.jsonl`
- `projects/work/baseline_llava/llava_onevision_7b/clean_only_6935_prompt_r1_20260324_latest_run.json`
- `projects/work/baseline_llava/llava_onevision_7b/eval_smoke_prompt_r1_partial/clean_metrics.json`
- `projects/work/baseline_llava/llava_onevision_7b/eval_smoke_prompt_r1_partial/clean_confusion_matrix.csv`

### 11.3 日志

- `projects/codes/safety_attack_dada_bundle/logs/llava_onevision_7b_clean100_4bit.log`
- `projects/codes/safety_attack_dada_bundle/logs/llava_onevision_7b_attack80_4bit.log`
- `projects/codes/safety_attack_dada_bundle/logs/llava_onevision_7b_attack_sev4_160_4bit.log`
- `projects/codes/safety_attack_dada_bundle/logs/llava_onevision_7b_clean6935_20260323_190916.log`
- `projects/codes/safety_attack_dada_bundle/logs/llava_onevision_7b_clean6935_prompt_r1_20260324_20260324_035017.log`

---

## 12. 一句话总结

到目前为止，本地已经把整个 benchmark 管线跑通，并找到 `OneVision 7B` 作为当前最可用的 LLaVA-family victim baseline；  
但现在最值得投入精力的，不再是继续盲跑旧 prompt，而是 **先重做 prompt，再用更强 attack 设计去验证鲁棒性退化**。

---

## 13. 2026-03-25 起的正式目标口径

接下来建议不再以当前本地最小 pilot 配置作为主线，而是改为**与队友 GitHub 仓库对齐的正式 benchmark 口径**。

参考目标配置如下：

| 项目 | 值 |
| --- | --- |
| 数据集 | `DADA-2000` clean `6935` 帧 |
| Overlay 类型 | `text_watermark`, `timestamp_box`, `logo_patch`, `semi_transparent_bar` |
| Placement 模式 | `random`, `critical`, `background` |
| Severity | `2`, `4` |
| 随机种子 | `42` |
| Alert threshold | `risk_score >= 3` 或 `risk = "yes"` |
| Delay threshold | `0.5` 秒 |
| Prompt | `projects/codes/safety_attack_dada_bundle/safety_attack_dada/prompts/binary_imminent_risk_prompt.txt` |

对应总样本量应为：

```text
6935 clean
+ 6935 × 4 overlays × 3 placements × 2 severities
= 173375 rows
```

这与队友仓库 README 中的正式 benchmark 口径一致，也与队友现在在跑的 `173,375` 行版本一致。

### 13.1 为什么 clean 必须单独评测

答案是：**必须单独跑 clean，而且它不是可选步骤，而是 attack benchmark 的前置条件。**

原因有三个：

1. 你们的 attack 成功不是泛泛看 attacked accuracy，而是看：
   - clean 时本来会不会报 `yes`
   - attacked 后是否从 `yes -> no`
2. 当前评测脚本 `07_eval_safety_attack.py` 的核心 attack candidate 定义，本来就是：
   - `label_risk = 1`
   - 且 `clean_pred_label = 1`
3. `warning delay` 指标也依赖 clean 条件下的首次报警时间；没有 clean，就没法定义 delay。

因此，后续正式主线应固定为：

1. 先跑 `clean 6935`
2. 确认 victim baseline 是否成立
3. 再跑 `173375` 的 full attack benchmark
4. 最后统一出 `frame ASR / delay ASR / confusion matrix / accuracy / precision / recall / F1`

---

## 14. 当前 OneVision 主线的真实状态

### 14.1 当前 prompt

当前主线 prompt 仍然是：

- `projects/codes/safety_attack_dada_bundle/safety_attack_dada/prompts/binary_imminent_risk_prompt.txt`

### 14.2 当前 clean 运行的有效进度

截至本次总结更新，OneVision 新 prompt clean 主线的有效进度应理解为：

- 第一段有效前缀：`487` 条
- 第二段 resume 有效结果：`207` 条
- 第二段 resume 中还有 `1` 条 OOM 错误

所以当前真正可用于 clean 分析的有效预测大约是：

- `487 + 207 = 694` 条有效样本

而不是把所有 JSONL 行数直接当成“full clean 已完成”。

### 14.3 OOM 状态说明

已经观察到两次 OOM：

1. 第一次：
   - 旧的 `prompt_r1_20260324` full clean 主线
   - 在第 `488` 条首次 OOM
   - 当时旧脚本未及时停表，导致后续写入了大量 error 尾部
2. 第二次：
   - `resume_from_488` 主线
   - 在新脚本保护下，于 `131/6371` 时再次 OOM
   - 这次脚本已经能在第一次 fatal CUDA error 后立即停止

第二次 resume 的当前暂停状态：

- pause state：
  - `projects/work/baseline_llava/llava_onevision_7b/clean_only_6935_prompt_r1_20260324_resume_from_488_pause_state.json`
- 当前 resume 输出：
  - `projects/work/baseline_llava/llava_onevision_7b/clean_only_6935_prompt_r1_20260324_resume_from_488_preds.jsonl`
- 对应日志：
  - `projects/codes/safety_attack_dada_bundle/logs/llava_onevision_7b_clean6935_prompt_r1_20260324_resume_from_488_20260324_210554.log`

结论上，当前 OneVision clean 主线是：

- **模型能力上仍值得继续**
- **工程上已支持安全停表和续跑**
- **但本机 12GB 显存下长跑 full clean 仍然容易 OOM**

这也是为什么下一阶段很适合迁到 AutoDL 的 `RTX PRO 6000` 上继续。

---

## 15. 迁移到 GitHub / AutoDL 的建议方案

### 15.1 当前本地情况

当前工作区还不是一个可直接 `git push` 的整洁仓库：

- 当前目录下没有初始化好的 Git repo
- 本机也没有可直接使用的 `gh` CLI
- 当前也没有现成配置好的 GitHub remote
- 现有实验文件分散在：
  - `projects/codes/safety_attack_dada_bundle/safety_attack_dada/`
  - `projects/codes/safety_attack_dada_bundle/logs/`
  - `projects/work/baseline_llava/`

所以不建议直接把当前整个工作区原样上传。

### 15.2 推荐的上传策略

推荐做法不是“上传当前全部本地实验垃圾桶”，而是：

1. 保留当前本地工作区原样不删
2. 在 `projects/codes` 下整理出一个**面向 GitHub / AutoDL 的干净 repo 版本**
3. GitHub 只放：
   - 代码
   - prompts
   - 配置
   - 说明文档
   - 轻量级结果摘要
4. 大体量本地产物继续只保存在本机或云端存储，不进 Git

### 15.3 推荐的 GitHub repo 结构

建议对齐你队友仓库风格，整理成下面这种结构：

```text
repo_root/
  README.md
  requirements.txt
  prompts/
  scripts/
  configs/
  results/
    local_summary/
  docs/
```

其中：

- `prompts/`
  - 放当前正式 prompt 与历史 prompt
- `scripts/`
  - 放清洗后的核心脚本
- `configs/`
  - 放 benchmark 参数、模型配置
- `results/local_summary/`
  - 只放轻量 summary，例如 `clean_metrics.json`、对齐表、attack group metrics
- `docs/`
  - 放 `PROJECT_SUMMARY_CN.md`、迁移说明、实验说明

### 15.4 哪些内容建议保留在本地，不上传 GitHub

这些内容建议**保留，但不要推到 GitHub**：

- `projects/work/baseline_llava/hf_cache/`
- `projects/work/baseline_llava/pilot_attack/images/`
- 大体积 `.jsonl` 原始预测文件
- 所有长日志 `logs/*.log`
- OOM 备份文件
- 本地绝对路径绑定很强的 full manifest

### 15.5 哪些内容建议整理后上传 GitHub

这些内容适合上传：

- `projects/codes/safety_attack_dada_bundle/safety_attack_dada/scripts/`
- `projects/codes/safety_attack_dada_bundle/safety_attack_dada/prompts/`
- `projects/codes/safety_attack_dada_bundle/safety_attack_dada/configs/`
- `projects/codes/safety_attack_dada_bundle/safety_attack_dada/requirements.txt`
- `projects/codes/safety_attack_dada_bundle/safety_attack_dada/README_CN.md`
- `projects/work/baseline_llava/PROJECT_SUMMARY_CN.md`
- `projects/work/baseline_llava/model_alignment_with_teammates.csv`
- `projects/work/baseline_llava/model_alignment_with_teammates.md`
- 轻量级 summary json / csv

### 15.6 AutoDL 上的建议工作流

推荐迁移顺序：

1. 在本机先整理出干净 repo
2. push 到 GitHub
3. 在 AutoDL 上 `git clone`
4. 在 AutoDL 上重建环境
5. 重新挂载或上传数据集
6. 按正式口径重建 benchmark：
   - clean `6935`
   - full attack `173375`
7. 在 AutoDL 上优先跑：
   - OneVision clean full `6935`
   - 如果 clean 通过，再跑 full attack

### 15.7 当前 GitHub 迁移状态

截至当前更新，已经完成一个**干净仓库版**的整理与首次推送：

- 本地干净仓库目录：
  - `projects/codes/VLM-Attack-on-Traffic-Safety`
- GitHub 仓库：
  - `git@github.com:QiPan-Ronnie/VLM-Attack-on-Traffic-Safety.git`
  - `https://github.com/QiPan-Ronnie/VLM-Attack-on-Traffic-Safety`
- 首次提交：
  - `aa90ee0c07f684fc60e4814e4c98be73db2863ec`

这个 GitHub 仓库当前已经包含：

- 核心脚本
- prompts
- configs
- PowerShell 工具
- 中文交接文档
- 轻量级结果摘要

当前仍然只保留在本地、未上传到 GitHub 的内容：

- 大体积 raw prediction `.jsonl`
- `hf_cache`
- 长日志
- 生成图片
- 本地 OOM 备份和暂停状态文件

---

## 16. 下一位 agent 的建议执行顺序

### 16.1 第一件事

先把“正式 repo 版本”整理出来，而不是继续在当前分散目录里加实验。

### 16.2 第二件事

把 benchmark 参数完全切换到正式口径：

- `4 overlays`
- `3 placements`
- `2 severities`
- `seed = 42`
- `clean = 6935`
- `full attack total = 173375`

### 16.3 第三件事

clean 评测优先，顺序固定为：

1. OneVision clean `6935`
2. clean 指标与 confusion matrix 对齐队友口径
3. 只有 clean baseline 合格时，才继续全量 attack

### 16.4 第四件事

若本机继续 OOM，不要再硬顶：

- 直接停
- 保留 partial output
- 迁移到 AutoDL 的更大显存卡继续

### 16.5 当前最重要的交接文件

新的 agent 至少先看这几份：

- `projects/work/baseline_llava/PROJECT_SUMMARY_CN.md`
- `projects/work/baseline_llava/model_alignment_with_teammates.md`
- `projects/work/baseline_llava/model_alignment_with_teammates.csv`
- `projects/work/baseline_llava/STATUS.md`
- `projects/work/baseline_llava/EXPERIMENT_LOG.md`

如果要恢复 OneVision clean 主线，再看：

- `projects/work/baseline_llava/llava_onevision_7b/clean_only_6935_prompt_r1_20260324_prefix_ok487.jsonl`
- `projects/work/baseline_llava/llava_onevision_7b/clean_only_6935_prompt_r1_20260324_resume_from_488_preds.jsonl`
- `projects/work/baseline_llava/llava_onevision_7b/clean_only_6935_prompt_r1_20260324_resume_from_488_pause_state.json`
