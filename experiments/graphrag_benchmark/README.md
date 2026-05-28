# GraphRAG 对比实验（独立目录，不影响线上）

本目录用于做三组对照实验：

1. `llm_only`：只给用户问题
2. `rag_only`：用户问题 + 文档检索片段
3. `graph_rag`：用户问题 + 文档片段 + 图谱三元组

## 目录结构

- `run_experiment.py`：执行三组推理并导出结果
- `questions.csv`：测试问题（可扩展到 30-50 条）
- `manual_scores_template.csv`：人工评分模板（自动生成/覆盖）
- `aggregate_manual_scores.py`：汇总人工评分并输出指标表
- `outputs/`：实验输出目录

## 运行步骤

推荐直接用 PowerShell 一键脚本：

```powershell
powershell -ExecutionPolicy Bypass -File experiments/graphrag_benchmark/run_benchmark.ps1
```

说明：
- 会优先读取当前终端已有的 `SILICONFLOW_API_KEY` / `NEO4J_PASSWORD`
- 若未设置，会在运行时交互提示输入
- 默认顺序执行：`run_experiment.py` -> `aggregate_manual_scores.py`

如需指定 Python（例如 `jsjsj`）：

```powershell
powershell -ExecutionPolicy Bypass -File experiments/graphrag_benchmark/run_benchmark.ps1 -PythonExe "D:\Anaconda3\envs\jsjsj\python.exe"
```

1. 在项目根目录执行：

```powershell
python experiments/graphrag_benchmark/run_experiment.py
```

2. 填写人工评分文件：

- 文件：`experiments/graphrag_benchmark/manual_scores_template.csv`
- 由 2-3 名同学分别打分（建议复制多份：`manual_scores_userA.csv`、`manual_scores_userB.csv`）

3. 汇总评分（以单份为例）：

```powershell
python experiments/graphrag_benchmark/aggregate_manual_scores.py --input experiments/graphrag_benchmark/manual_scores_template.csv
```

4. 半自动预打分 + 总表与短分析（可选）：

```powershell
python experiments/graphrag_benchmark/prefill_scores.py --input experiments/graphrag_benchmark/manual_scores_template.csv
python experiments/graphrag_benchmark/analysis_short_report.py --input experiments/graphrag_benchmark/manual_scores_template.csv
```

## 人工评分字段说明

- `is_correct`：回答准确率（0/1）
- `evidence_hit`：证据命中率（0/1）
- `hallucination`：是否幻觉（0/1，1 表示有幻觉）
- `root_path_hit`：根因路径命中率（0/1）

脚本还会自动统计平均响应时间（毫秒）。

## 注意

- 该实验脚本只读取现有 `app.py` 的能力，不修改线上接口逻辑。
- 如果未配置 `SILICONFLOW_API_KEY`，三组中需要 LLM 的部分会失败并在结果中标记错误。
