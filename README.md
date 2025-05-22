# A股年报"管理层讨论与分析"智能提取工具

## 项目简介
本项目旨在从A股上市公司年报PDF中，自动、批量、智能地提取"管理层讨论与分析"（MD&A）章节文本，支持多核并行、OCR识别、文本清洗和API辅助修复，极大提升年报数据分析的效率。

## 主要功能
- **多核并行处理**：充分利用CPU多核能力，批量处理速度快。
- **智能章节识别**：通过正则和关键词灵活定位MD&A章节。
- **OCR支持**：自动识别低质量PDF页面文本。
- **文本清洗**：一键去除页码、页眉、特殊符号等杂质。
- **API修复失败**：调用大语言模型自动修复常规方法无法提取的文件。
- **异步API处理**：使用异步IO优化API请求效率。
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
├── fix_failed_extractions.py # API修复脚本
├── clean.py                # 文本清洗脚本
├── debug_extractor.py      # 提取失败调试工具
├── suppress_warnings.py    # 警告抑制工具
└── requirements.txt        # 依赖包列表
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
PDF_DIR = "areport/2013_2537份"   # PDF文件目录
OUT_DIR = "mdna_txts/2013"        # 输出目录
LLM_API_KEY = "你的API密钥"        # API密钥
LLM_BASE_URL = "https://api.siliconflow.cn/v1"  # API基础URL
LLM_MODEL = "deepseek-ai/DeepSeek-V3"  # 使用的模型
```

### 3. 分步执行处理流程

按照以下步骤顺序执行：

```bash
# 步骤1：使用正则表达式提取MD&A章节
python extract.py
# 此步骤会在OUT_DIR目录下生成提取结果，并在FAIL_DIR中生成失败文件列表

# 步骤2：使用API修复失败的文件
python fix_failed_extractions.py  # 同步模式
# 或使用异步模式（更高效）
python fix_failed_extractions.py --async-mode --workers 5
# 修复结果保存在OUT_DIR + "_llm_fixed"目录

# 步骤3：清理文本，移除页码、页眉等干扰信息
python clean.py
# 清理后的文本保存在OUT_DIR + "_cleaned"目录
```

### 4. 高级选项
`fix_failed_extractions.py`支持多种运行模式：

```bash
# 查看帮助信息
python fix_failed_extractions.py --help

# 使用异步模式并设置并行工作数（加快处理速度）
python fix_failed_extractions.py --async-mode --workers 8

# 设置API请求频率限制（避免触发限流）
python fix_failed_extractions.py --async-mode --rate-limit 30
```

## 处理流程详解

### 初始提取 (extract.py)
- 通过正则表达式和关键词匹配定位MD&A章节
- 多进程并行处理，提高效率
- 生成成功提取文件和失败列表

### 失败文件修复 (fix_failed_extractions.py)
- 读取失败文件列表
- 调用大模型API定位内容位置
- 根据返回的位置信息提取内容
- 支持同步和异步两种处理模式

### 文本清洗 (clean.py)
- 移除页码、页眉页脚
- 清理特殊字符和多余空行
- 处理不正确的断行

## 异步API处理特点
- **并发控制**：限制最大并行请求数
- **频率限制**：自动控制API请求频率
- **错误重试**：指数退避重试失败请求
- **进度显示**：实时显示处理进度

## 常见问题
- **OCR报错**：请确保Tesseract已正确安装并配置环境变量。
- **内存不足**：可在`config.py`中减少CPU核心数。
- **API密钥错误**：确保在config.py中配置了正确的API密钥。
- **识别失败**：可自定义`config.py`中的正则表达式和关键词。

## 调试与排错
- 如遇提取失败，可使用`debug_extractor.py`分析失败原因：
  ```bash
  python debug_extractor.py 某公司年报.pdf
  ```
- 详细日志保存在`logs`目录下
- API调用相关问题可查看`logs/llm_fix.log`

## 贡献与反馈
欢迎提交issue、PR或年报样本，帮助完善提取规则和功能。

---
如有疑问或建议，欢迎联系作者。
