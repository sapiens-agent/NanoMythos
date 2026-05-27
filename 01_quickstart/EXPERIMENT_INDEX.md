# NanoMythos T=1 实验全索引

**图例：** ★ = 建议对外沟通团队引用 · 📦 = 本 bundle 已含脚本或报告

| 阶段 | 语料 | 对比 | 要点 | 脚本 📦 | 报告 📦 | Checkpoint（/data/zetyun） |
|------|------|------|------|---------|---------|------------------------------|
| **521** ★ | FineWeb-Edu 10K | base vs T=1 | 5k, seeds 2025/2027/2048 | `521_fw_cross_seed_5k.py` | `521_fineweb10k_cross_seed.md` | `phase2_521_fw_cs_*` |
| **520** | FineWeb 10K | base vs T=1 | 5k s2025 | `eval_external/520_fineweb_train.sh` | 521 引用 | `phase2_520_fineweb_fw_*_5k` |
| **518 Wiki** ★ | Wiki full/100/**50** | base vs T=1 | 3k | `wiki/run_phase2_518_wiki.sh` | `reports_wiki/*` | `phase2_518_wiki*` |
| **520 Wiki** | Wiki 50 | +T2/T4, 10k | 3k multi-seed | `wiki/run_phase2_520_wiki50_deep.sh` | `reports_wiki/*` | `phase2_520_wiki50_*` |
| **521 Wiki** | Wiki 50 | 1k/2k/5k sweep | **2k 最佳** | `wiki/run_phase2_521_earlystop.sh` | `wiki50_earlystop` | `phase2_521_wiki50_*` |
| **522** | Wiki 50 | base vs T=1 | 2k, 2027/2048 | `wiki/run_phase2_522.sh` | `522 wiki50 2k` | `phase2_522_wiki50_*` |
| **519 pilot** | FineWeb pilot parquet | T=1/2/4 depth | staged PPL | `pilot519/519_run_pilot.sh` | 519reports | `phase2_519_*` |
| **519 20ng** | 20 Newsgroups | base vs T=1 | 3k | `pilot519/519_run_pilot_20ng.sh` | `514 20ng` | `phase2_519_20ng_*` |
| **514/518 JD** | 京东中文短评 | base vs T=1 | 3k, 2 seeds | `jd_20ng/run_phase2_518*.sh` | `reports_514/jd*` | `phase2_518_t1on*`, `519_jd_*` |
| **515/518** | 截断 50 字 / packed | 长度因果 | 机制 | `phase2_misc/515_*` | `reports_515`, `518 shorttext` | `phase2_515_trunc50_*` |
| **516** | small LM parquet | base/t1/t4 | 500 step | `phase2_misc/516_*` | `reports_misc/516` | `phase2_516_t1on_*` |
| **518/520 LM** | phase2_lm_real | wall-match T=1 | 1k–1.8k | `fineweb/run_phase2_520_train.sh` | `518 real_lm` | `phase2_520_lmreal_*` |
| **521 下午** | FineWeb val 分桶 | T=1 vs T=2 | eval | `521_eval_length_binned.py` | `521_afternoon_findings` | eval/521 JSON |
| **521** | GPT-2 对照 | 外部 | 5k | — | `521_gpt2_small_standard` | `phase2_520_fineweb_gpt2std_*` |
| **521** | DDP smoke | T=1 only | 200 step | `train_pretrain` | `521_ddp_smoke_audit` | `phase2_521_ddp_smoke_*` |

## 数据路径速查

| 文件 | 用途 |
|------|------|
| `fineweb_edu_10k.parquet` | 521 / 520 FineWeb |
| `nanowhale_518_wiki_text_short50.parquet` | Wiki50 ★ |
| `nanowhale_519_fineweb_pilot.parquet` | 519 pilot |
| `nanowhale_phase2_lm_real_train.parquet` | 518/520 real LM |
| `wiki_demo.txt` → 518 parquet 构建 | 见 `reports_wiki/518_nanowhale_wiki_text_data_build*` |

## 未收录（主仓库）

526 Coconut-light · GPT-2 训练 `520_gpt2_*.py` · 50K 主训（已取消）

完整收录列表：`docs/BUNDLE_CHECKLIST.md`
