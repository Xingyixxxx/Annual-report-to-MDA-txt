import os
import re
import logging
import multiprocessing
import warnings

# ———— 文件路径配置 ————
PDF_DIR = "areport/2013_2537份"  # PDF文件目录
OUT_DIR = "mdna_txts/2013"  # 输出目录
LOG_DIR = "logs"  # 日志目录
FAIL_DIR = "fail/2013"  # 失败文件目录
LOG_PATH = os.path.join(LOG_DIR, "extract_mdna.log")  # 日志文件
FAIL_LIST_TXT = os.path.join(FAIL_DIR, "fail.txt")  # 失败列表
DEBUG_REPORTS_DIR = os.path.join(FAIL_DIR, "debug_reports")  # 调试报告保存目录

# ———— 日志配置 ————
LOG_LEVEL = logging.ERROR  # 设置为ERROR级别，减少警告信息 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_TO_CONSOLE = True  # 是否将日志输出到控制台

# 全局警告抑制设置
SUPPRESS_WARNINGS = True  # 是否抑制所有警告信息

# ———— 多进程配置 ————
# 默认使用所有可用CPU核心数, 可自行调整
CPU_COUNT = multiprocessing.cpu_count()

# ———— LLM API配置 ————
# API密钥，使用前请替换为你的实际密钥
LLM_API_KEY = "你的API密钥"
# API基础URL
LLM_BASE_URL = "https://api.siliconflow.cn/v1"
# 使用的模型名称
LLM_MODEL = "Pro/deepseek-ai/DeepSeek-V3"

# ———— OCR配置 ————
USE_OCR = True  # 是否使用OCR处理低质量文本
OCR_THRESHOLD = 50  # 文本少于此字符数时尝试OCR处理
OCR_LANG = "chi_sim"  # OCR语言设置

# ———— MDA章节识别配置 ————
# 章节开始正则表达式
START_PATTERNS = [
    # 严格匹配模式 - 只使用这一个模式以确保提取精度
    r"^第[一二三四五六七八九十0-9]{1,3}(?:章|节)\s*(?:管理层讨论与分析|经营情况讨论与分析|董事会报告|董事局报告)\s*$",
    
    # 新增: 匹配"四、董事会报告"格式 (中文数字或阿拉伯数字+标点+关键词)
    r"^\s*[一二三四五六七八九十0-9]+[、，：:]\s*(?:管理层讨论与分析|经营情况讨论与分析|董事会报告|董事局报告)\s*$"
]

# 章节结束正则表达式
END_PATTERNS = [
    r"^第[四五六七八九十0-9]{1,3}(?:章|节)",
    r"^\s*(?:第\s*)?(?:[一二三四五六七八九十0-9]+)\s*[章节]?\s*[：:]*\s*(?:重要事项|公司治理|财务报告|企业管治报告|监事会报告)",
    r"^\s*(?:重要事项|公司治理|财务报告|企业管治报告|监事会报告)\s*$",
]

# 将多个模式组合为一个正则表达式对象
START_REGEX = re.compile('|'.join(START_PATTERNS))
END_REGEX = re.compile('|'.join(END_PATTERNS))

# 内容关键词集 - 这些词通常在管理层讨论中出现，可用于内容确认
MDA_KEYWORDS = [
    "总体经营情况", "主营业务分析", "财务状况", "资产负债", "现金流量", 
    "投资状况", "经营成果", "行业格局", "发展战略", "研发投入",
    "营业收入", "营业利润", "净利润", "毛利率", "业务回顾",
    "市场分析", "风险因素", "未来展望", "可持续发展"
]

# 每个文件最多尝试的页数
MAX_PAGES_TO_TRY = 30  # 避免遍历整个文件，通常MDA在前30页内会有明显标记

# ———— LLM API配置 ————
LLM_API_KEY = "sk-wqtpbqudgyfxjgaalqnizbmkowpirgnnorzpxfprjpsodbrf"  # 填入你的API密钥
LLM_BASE_URL = "https://api.siliconflow.cn/v1"
LLM_MODEL = "deepseek-ai/DeepSeek-V3"
LLM_MAX_TOKENS = 2048
LLM_TEMPERATURE = 0.2  # 较低的温度让结果更加确定性

# 是否使用LLM修复失败文件
USE_LLM_FIX = True
