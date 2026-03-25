# 与队友 DeepSeek 结果对齐表

这张表优先服务于 `full_clean_6935` 口径下与队友结果的对齐比较。  
其中 `subset` 和 `partial` 结果仍然保留，但只能作为阶段性参考，不能直接替代 full-clean 主表。

| 模型 | 数据来源 | 评测范围 | Parse rate | Accuracy | Precision | Recall | F1 | TP | TN | FP | FN | 是否可直接对齐 DeepSeek |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DeepSeek-VL-1.3B | 队友截图 | full clean `6935` | 0.0300 | 0.2150 | 0.8050 | 0.0250 | 0.0485 | 136 | 1353 | 33 | 5413 | 是（基准行） |
| CLIP ViT-B-32 | 本地结果 | full clean `6935` | 1.0000 | 0.7384 | 0.8025 | 0.8928 | 0.8452 | 4954 | 167 | 1219 | 595 | 是 |
| LLaVA-OneVision-0.5B | 本地结果 | subset clean `100` | 1.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0 | 50 | 0 | 50 | 否 |
| LLaVA-1.5-7B | 本地结果 | subset clean `100` | 1.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0 | 50 | 0 | 50 | 否 |
| LLaVA-OneVision-Qwen2-7B | 本地结果 | balanced clean `100` | 1.0000 | 0.5100 | 0.5063 | 0.8000 | 0.6202 | 40 | 11 | 39 | 10 | 否 |
| LLaVA-OneVision-Qwen2-7B | 本地 partial 结果 | full clean 前 `258` 条 | 0.0372 | 0.6744 | 0.8211 | 0.7573 | 0.7879 | 156 | 18 | 34 | 50 | 仅部分可参考 |

说明：

- `DeepSeek-VL-1.3B / full_clean_6935`：来自队友截图，当前本地还没有对应原始结果文件。
- `CLIP ViT-B-32 / full_clean_6935`：当前本地唯一已经和队友 full-clean 规模完全对齐的稳定基线。
- `LLaVA-OneVision-0.5B / subset_clean_100`：仅用于说明 0.5B 在当前任务设定下几乎塌到全 `no`。
- `LLaVA-1.5-7B / subset_clean_100`：仅用于说明旧 LLaVA 1.5 7B 在当前任务设定下也几乎塌到全 `no`。
- `LLaVA-OneVision-Qwen2-7B / balanced_clean_100`：当前最有希望的 LLaVA-family victim candidate，但还只是 clean100 阶段结果。
- `LLaVA-OneVision-Qwen2-7B / partial_full_clean_258_of_6935`：旧 prompt 主线暂停前的 partial full-clean 结果，只能看趋势，不能当最终结果。

后续要求：

- 新 prompt 的 OneVision full-clean 跑完后，必须补出同口径的：
  - `Parse rate`
  - `Accuracy`
  - `Precision`
  - `Recall`
  - `F1`
  - `TP / TN / FP / FN`
  - `Confusion Matrix`
