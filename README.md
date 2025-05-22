# A股年报“管理层讨论与分析”智能提取工具

## 项目简介
本项目旨在从A股上市公司年报PDF中，自动、批量、智能地提取“管理层讨论与分析”（MD&A）章节文本，支持多核并行、OCR识别、文本清洗和AI辅助修复，极大提升年报数据分析的效率。

## 主要功能
- **多核并行处理**：充分利用CPU多核能力，批量处理速度快。
- **智能章节识别**：通过正则和关键词灵活定位MD&A章节。
- **OCR支持**：自动识别低质量PDF页面文本。
- **文本清洗**：一键去除页码、页眉、特殊符号等杂质。
- **AI修复失败**：调用大语言模型自动修复常规方法无法提取的文件。
- **详细日志与调试**：日志、失败列表、调试报告一应俱全。

## 目录结构说明
```
PDF_to_TXT/
├── areport/                # 存放原始PDF文件（按年份分类）
├── mdna_txts/              # 提取的MD&A文本输出目录
├── mdna_txts_cleaned/      # 清洗后的文本输出目录
├── fail/                   # 提取失败的PDF及调试信息
├── logs/                   # 日志文件
├── config.py               # 配置文件（路径、参数、正则等）
├── extract.py              # 主提取脚本
├── fix_failed_extractions.py # AI修复失败文件脚本
├── clean.py                # 文本清洗脚本
├── debug_extractor.py      # 提取失败调试工具
├── suppress_warnings.py    # 警告抑制工具
├── requirements.txt        # 依赖包列表
└── README.md               # 使用说明
```

## 快速上手
### 1. 安装依赖
```bash
pip install -r requirements.txt
```
如需OCR功能，请确保已安装Tesseract（macOS可用`brew install tesseract`）。

### 2. 配置参数
编辑`config.py`，设置PDF目录、输出目录、API密钥等：
```python
PDF_DIR = "areport/2012_2493份"   # PDF文件目录
OUT_DIR = "mdna_txts/2012"        # 输出目录
LLM_API_KEY = "你的API密钥"        # 如需AI修复功能
```

### 3. 批量提取MD&A章节
```bash
python extract.py
```
输出文本将保存在`mdna_txts/`下。

### 4. 修复提取失败的文件（可选）
如需AI辅助修复：
```bash
python fix_failed_extractions.py
```
修复结果保存在`mdna_txts_llm_fixed/`。

### 5. 文本清洗（可选）
对提取文本进一步去除杂质：
```bash
python clean.py
```
清洗后文本保存在`mdna_txts_cleaned/`。

### 6. 调试与自定义
- 若遇到大批量识别失败，可用`debug_extractor.py`分析失败原因，调整`config.py`中的正则表达式。
- 日志和失败列表可在`logs/`和`fail/`目录下查看。

## 常见问题
- **OCR报错**：请确保Tesseract已正确安装并配置环境变量。
- **内存不足**：可在`config.py`中减少CPU核心数。
- **识别失败**：可自定义`config.py`中的正则表达式和关键词。
- **AI修复需API密钥**：如需AI修复功能，请申请并填写有效API密钥。

## 贡献与反馈
欢迎提交issue、PR或年报样本，帮助完善提取规则和功能。

---
如有疑问或建议，欢迎联系作者。
