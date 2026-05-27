# 依赖与路径

1. **代码根目录：** 克隆完整 `nanowhale` 仓库；本 bundle 中 `core/*.py` 为快照，建议以仓库根为准同步更新。
2. **Tokenizer：** `tokenizer/`（PreTrainedTokenizerFast），所有 `train_pretrain` / `521_fw_cross_seed_5k` 依赖。
3. **数据（/data/zetyun，可按环境改脚本常量）：**
   - FineWeb：`fineweb_edu_10k.parquet`
   - Wiki：`nanowhale_518_wiki_text_short50.parquet` 等
   - JD / 20ng / LM real：见 `EXPERIMENT_INDEX.md`
4. **Python 环境：** `nanowhale_runtime`（PyTorch + CUDA + `trl`, `datasets`, `transformers`）。
