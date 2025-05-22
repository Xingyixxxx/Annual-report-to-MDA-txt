#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDF提取失败文件修复工具 - 顺序处理版本

功能:
1. 读取失败文件列表
2. 使用API定位内容位置
3. 根据位置信息从本地PDF文件提取内容
4. 逐个处理PDF，不使用并发
5. 精确计算token使用量，控制每分钟请求频率

日期: 2025-05-22
"""

import os
import re
import json
import time
import logging
import argparse
import sys
import pdfplumber
from tqdm import tqdm
from openai import OpenAI
import config
from extract import extract_page_text, find_mdna_section_with_position
from collections import deque
import transformers  # 添加transformers依赖

# ===== Token计数器及速率限制设置 =====
# 全局定义相关常量和历史记录
TOKENS_PER_MINUTE_LIMIT = 5000000  # DeepSeek-V3每分钟token限制
REQUESTS_PER_MINUTE_LIMIT = 30000  # DeepSeek-V3每分钟请求数量限制
RATE_LIMIT_THRESHOLD = 0.85  # 触发等待的API使用率阈值
# token_usage_history 需要能容纳至少一分钟内最大请求数
token_usage_history = deque(maxlen=REQUESTS_PER_MINUTE_LIMIT + 5000) # 例如 30000 + 5000 = 35000

# 全局tokenizer变量
tokenizer = None
TOKEN_COUNTER_LOADED = False

# 注意: 以下日志记录依赖于 logging 模块的基本配置或已由 setup_logging() 配置的 logger 实例。
# 如果 setup_logging() 在此之后调用，这些早期日志可能行为不同。

try:
    tokenizer_dir = os.path.join(os.path.dirname(__file__), "deepseek_v3_tokenizer")
    _tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    tokenizer = _tokenizer 
    TOKEN_COUNTER_LOADED = True
    logging.info("Token计数器初始化成功 (DeepSeek-V3).")
except Exception as e:
    logging.warning(f"警告: Token计数器 (DeepSeek-V3) 初始化失败 - {e}")
    logging.warning("将使用估计方式计算token。请求速率限制仍将按配置工作。")
    TOKEN_COUNTER_LOADED = False

def count_tokens_real(text_to_count):
    """使用实际加载的tokenizer计算token，带异常处理"""
    if not text_to_count:
        return 0
    if not tokenizer: 
        logging.warning("尝试使用实际tokenizer计数，但tokenizer未加载。返回估计值。")
        return len(text_to_count) // 4
    try:
        tokens = tokenizer.encode(text_to_count)
        return len(tokens)
    except Exception as e:
        logging.warning(f"使用DeepSeek-V3 tokenizer计算Token时出错: {e}. 对此文本回退到估计值。")
        return len(text_to_count) // 4

def count_tokens(text_to_count):
    """根据TOKEN_COUNTER_LOADED状态选择真实计数或估算"""
    if not text_to_count:
        return 0
    if TOKEN_COUNTER_LOADED and tokenizer: # Ensure tokenizer is not None
        return count_tokens_real(text_to_count)
    else:
        return len(text_to_count) // 4

def get_current_token_usage():
    """获取当前一分钟内的总token使用量"""
    now = time.time()
    recent_usage = [tokens for timestamp, tokens in token_usage_history if now - timestamp < 60]
    return sum(recent_usage)

def get_current_request_count():
    """获取当前一分钟内的请求次数"""
    now = time.time()
    return sum(1 for timestamp, _ in token_usage_history if now - timestamp < 60)

def record_token_usage(input_text, output_text=None):
    """记录API调用使用的token数量，并更新历史记录"""
    input_tokens = count_tokens(input_text)
    output_tokens = count_tokens(output_text) if output_text else 0
    total_tokens = input_tokens + output_tokens
    token_usage_history.append((time.time(), total_tokens))
    return total_tokens

def wait_for_token_limit():
    """
    检查API使用是否接近限制，如果接近则等待适当时间。
    循环检查，直到用量低于阈值。
    """
    while True:
        current_tokens = get_current_token_usage()
        current_requests = get_current_request_count()

        token_usage_percent = (current_tokens / TOKENS_PER_MINUTE_LIMIT) * 100 if TOKENS_PER_MINUTE_LIMIT > 0 else 0
        request_usage_percent = (current_requests / REQUESTS_PER_MINUTE_LIMIT) * 100 if REQUESTS_PER_MINUTE_LIMIT > 0 else 0
        
        # logging.debug(f"当前API使用 - Tokens: {current_tokens:,}/{TOKENS_PER_MINUTE_LIMIT:,} ({token_usage_percent:.2f}%), "
        #             f"请求: {current_requests:,}/{REQUESTS_PER_MINUTE_LIMIT:,} ({request_usage_percent:.2f}%)")

        token_limit_triggered = current_tokens > TOKENS_PER_MINUTE_LIMIT * RATE_LIMIT_THRESHOLD
        request_limit_triggered = current_requests > REQUESTS_PER_MINUTE_LIMIT * RATE_LIMIT_THRESHOLD

        if not token_limit_triggered and not request_limit_triggered:
            break 

        now = time.time()
        oldest_ts_in_window = float('inf')
        found_active_request_in_window = False

        for ts, _ in token_usage_history:
            if now - ts < 60: 
                oldest_ts_in_window = ts
                found_active_request_in_window = True
                break  

        wait_duration = 1.0 
        if found_active_request_in_window:
            time_to_oldest_expiry = (oldest_ts_in_window + 60) - now
            wait_duration = max(0, time_to_oldest_expiry) + 1.0 
        else:
            if token_limit_triggered or request_limit_triggered:
                 logging.warning("API超限但未在历史记录中找到活跃请求以计算精确等待时间，默认等待1秒。")

        reasons = []
        if token_limit_triggered:
            reasons.append(f"Tokens {token_usage_percent:.2f}%")
        if request_limit_triggered:
            reasons.append(f"Requests {request_usage_percent:.2f}%")
        
        if wait_duration > 0 and (token_limit_triggered or request_limit_triggered):
            logging.info(f"API使用率超阈值 ({', '.join(reasons)}). 等待 {wait_duration:.2f} 秒...")
            time.sleep(wait_duration)
        elif not (token_limit_triggered or request_limit_triggered): 
            continue 
        else: 
            time.sleep(0.1)

    return current_tokens, current_requests

# ===== 日志设置 =====
def setup_logging():
    """设置日志系统"""
    log_dir = os.path.dirname(config.LOG_PATH)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    handlers = [logging.FileHandler(os.path.join(log_dir, "llm_fix_sequential.log"))]
    if config.LOG_TO_CONSOLE:
        handlers.append(logging.StreamHandler())
        
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ===== LLM API 客户端 =====
def create_llm_client():
    """创建大模型API客户端"""
    try:
        if not config.LLM_API_KEY or config.LLM_API_KEY == "你的API密钥":
            logger.error("API密钥未设置，请在config.py中配置LLM_API_KEY")
            print("\n错误: API密钥未设置！")
            print("请在config.py中配置有效的LLM_API_KEY后再运行此脚本")
            return None
        return OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    except Exception as e:
        logger.error(f"创建LLM客户端失败: {e}")
        print(f"\n错误: 创建LLM客户端失败: {e}")
        print("请检查API密钥和base_url是否正确")
        return None

# ===== PDF 提取功能 =====
def extract_full_text(pdf_path, max_pages=30):
    """从PDF中提取前N页文本用于位置识别"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            # 仅处理前N页以控制API调用中的文本长度
            max_pages = min(max_pages, len(pdf.pages))
            for i in range(max_pages):
                page_text = extract_page_text(pdf.pages[i])
                if page_text:
                    text += f"===== 第{i+1}页 =====\n{page_text}\n\n"
            return text
    except Exception as e:
        logger.error(f"PDF提取文本失败: {e}")
        return None

def find_mdna_position_with_llm(client, pdf_text, file_name):
    """使用大模型定位MDA部分位置信息，同步方式"""
    if not client or not pdf_text:
        return None
    
    # 限制文本长度，避免超出API限制
    if len(pdf_text) > 15000:
        pdf_text = pdf_text[:15000] + "...(文本已截断)"
    
    try:
        # 构建提示，只获取位置信息，简化要求以提高成功率
        prompt = f"""
请帮我在以下A股公司年报中定位"管理层讨论与分析"(MDA)部分的精确位置。
这部分标题可能是："管理层讨论与分析"、"经营情况讨论与分析"、"董事会报告"等。

请严格按照以下格式返回位置信息:
{{
  "start_page": 数字(第几页开始，1-based),
  "end_page": 数字(第几页结束，1-based),
  "start_keyword": "找到的标题完整文本",
  "confidence": 0.0到1.0之间的数值，表示对结果的置信度
}}

只返回上述格式的位置信息JSON，不要添加其他内容。

文件名: {file_name}

年报文本:
{pdf_text}
"""
        # 检查和等待token限制
        wait_for_token_limit() # 调用更新后的函数，不再接收返回值
        # logger.info(f"当前API使用情况 - Tokens: {current_tokens:,}/{TOKENS_PER_MINUTE_LIMIT:,}, 请求数: {current_requests}/{REQUESTS_PER_MINUTE_LIMIT}") # 此行移除
        
        # 预计算输入token
        input_tokens = count_tokens(prompt)
        logger.debug(f"API请求输入token数: {input_tokens} (文件: {file_name})")
        
        # 调用API
        logger.info(f"正在使用LLM定位文件MDA位置: {file_name}")
        start_time = time.time()
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个专业的文档定位助手。你只返回JSON格式的位置信息，不返回任何其他文本。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"}  # 请求JSON响应
        )
        elapsed = time.time() - start_time
        
        # 获取响应文本
        result = response.choices[0].message.content
        logger.debug(f"LLM原始响应: {result}")
        
        # 记录token使用情况
        if hasattr(response, 'usage') and response.usage:
            total_tokens = response.usage.total_tokens
            logger.info(f"API响应 - 总计使用{total_tokens}个token，响应时间{elapsed:.2f}秒")
        else:
            # 如果API没有返回token使用情况，则使用我们的计数器
            total_tokens = record_token_usage(prompt, result)
            logger.info(f"API响应 - 估计使用{total_tokens}个token，响应时间{elapsed:.2f}秒")
        
        # 尝试多种方式解析位置信息
        position_info = None
        
        # 方法1: 直接尝试解析整个响应为JSON
        try:
            position_info = json.loads(result.strip())
            logger.debug("成功通过直接JSON解析获取位置信息")
        except Exception as e:
            logger.debug(f"JSON解析失败: {e}")
        
        # 方法2: 如果解析失败，使用默认值
        if not position_info or not isinstance(position_info, dict):
            logger.warning(f"无法解析位置信息，使用默认值")
            position_info = {
                "start_page": 10,
                "end_page": 30,
                "start_keyword": None,
                "confidence": 0.3
            }
        
        # 验证和调整值
        if not position_info.get('start_page') or not position_info.get('end_page'):
            position_info['start_page'] = position_info.get('start_page', 10)
            position_info['end_page'] = position_info.get('end_page', 30)
        
        return position_info
            
    except Exception as e:
        logger.error(f"LLM API调用失败: {e}")
        return None

def process_file(pdf_path, client, output_dir):
    """顺序处理单个PDF文件"""
    file_name = os.path.basename(pdf_path)
    txt_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + ".txt")
    debug_path = os.path.join(config.DEBUG_REPORTS_DIR, os.path.splitext(file_name)[0] + "_llm_debug.txt")
    
    # 确保调试目录存在
    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
    
    try:
        # 检查输出文件是否已存在
        if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
            logger.info(f"文件已存在，跳过处理: {file_name}")
            return True
            
        # 提取PDF文本用于位置识别
        pdf_text = extract_full_text(pdf_path, max_pages=30)
        if not pdf_text:
            logger.error(f"无法从PDF提取文本: {file_name}")
            return False
        
        # 使用LLM获取MDA部分位置
        position_info = find_mdna_position_with_llm(client, pdf_text, file_name)
        if not position_info:
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(f"无法获取MDA位置信息: {file_name}")
            return False
            
        # 记录位置信息到调试文件
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(f"MDA位置信息:\n")
            f.write(f"开始页: {position_info.get('start_page', 'unknown')}\n")
            f.write(f"结束页: {position_info.get('end_page', 'unknown')}\n")
            f.write(f"起始关键词: {position_info.get('start_keyword', 'unknown')}\n")
            f.write(f"置信度: {position_info.get('confidence', 0.0)}\n")
        
        # 基于1的页码转换为基于0的索引
        start_page_idx = max(0, int(position_info.get('start_page', 10)) - 1)
        end_page_idx = max(0, int(position_info.get('end_page', 30)) - 1)
        
        # 使用位置信息提取MDA内容
        mda_text = find_mdna_section_with_position(
            pdf_path, 
            start_page_idx,
            end_page_idx,
            position_info.get('start_keyword')
        )
        
        # 检查提取结果
        if not mda_text or len(mda_text.strip()) < 100:
            logger.warning(f"根据位置信息无法提取有效内容: {file_name}")
            with open(debug_path, "a", encoding="utf-8") as f:
                f.write("\n提取失败或内容太少，尝试备用方法...\n")
            
            # 备用方法：尝试提取更大范围
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    # 扩大提取范围
                    start = max(0, start_page_idx - 5)
                    end = min(len(pdf.pages) - 1, end_page_idx + 10)
                    
                    # 提取文本
                    fallback_text = []
                    for i in range(start, end + 1):
                        page_text = extract_page_text(pdf.pages[i])
                        fallback_text.append(page_text)
                    
                    mda_text = "\n\n".join(fallback_text)
                    
                    with open(debug_path, "a", encoding="utf-8") as f:
                        f.write("使用备用方法提取内容\n")
            except Exception as e:
                logger.error(f"备用方法提取失败: {e}")
                return False
        
        if not mda_text or len(mda_text.strip()) < 200:
            logger.error(f"所有方法都无法提取足够内容: {file_name}")
            return False
        
        # 保存结果
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(mda_text)
        
        # 更新调试信息
        with open(debug_path, "a", encoding="utf-8") as f:
            f.write(f"\n提取成功! 内容长度: {len(mda_text)} 字符\n")
        
        logger.info(f"成功修复并提取: {file_name}")
        return True
        
    except Exception as e:
        logger.error(f"处理失败: {file_name}, 错误: {e}")
        # 记录错误到调试文件
        try:
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(f"处理错误: {str(e)}")
        except:
            pass
        return False

def process_files_sequential(failed_files, output_dir):
    """顺序处理失败文件列表"""
    # 创建LLM客户端
    client = create_llm_client()
    if not client:
        logger.error("无法创建LLM客户端，请检查API配置")
        return 0, len(failed_files)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 过滤已存在的文件
    filtered_files = []
    skipped_count = 0
    for file_name in failed_files:
        pdf_path = os.path.join(config.PDF_DIR, file_name)
        txt_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + ".txt")
        
        # 检查PDF文件是否存在
        if not os.path.exists(pdf_path):
            logger.warning(f"PDF文件不存在: {file_name}")
            continue
            
        # 检查是否已存在处理结果
        if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
            logger.info(f"文件已存在，跳过处理: {file_name}")
            skipped_count += 1
            continue
            
        filtered_files.append(file_name)
    
    if skipped_count > 0:
        logger.info(f"跳过 {skipped_count} 个已处理的文件")
    
    if not filtered_files:
        logger.info("没有需要处理的文件")
        return skipped_count, 0
    
    # 显示需要处理的文件数量
    logger.info(f"需要处理 {len(filtered_files)} 个文件")
    print(f"需要处理 {len(filtered_files)} 个文件")
    
    # 逐个处理文件
    success_count = 0
    fail_count = 0
    
    with tqdm(total=len(filtered_files), desc="处理PDF") as pbar:
        for file_name in filtered_files:
            pdf_path = os.path.join(config.PDF_DIR, file_name)
            
            # 处理单个文件
            success = process_file(pdf_path, client, output_dir)
            
            if success:
                success_count += 1
            else:
                fail_count += 1
                
            # 更新进度条
            pbar.update(1)
            
            # 在请求之间添加间隔，避免API限制
            time.sleep(1.5)
    
    # 返回处理结果，包括跳过的文件
    return success_count + skipped_count, fail_count

def main():
    """主函数：处理所有失败的提取文件"""
    parser = argparse.ArgumentParser(description="PDF提取失败文件修复工具 - 顺序处理版本")
    parser.add_argument("--fail-list", type=str, help="自定义失败文件列表路径")
    parser.add_argument("--dry-run", action="store_true", help="仅检查配置，不实际执行")
    args = parser.parse_args()

    # 验证API配置
    if config.LLM_API_KEY == "你的API密钥" and not args.dry_run:
        print("\n错误: API密钥未配置!")
        print("请在config.py文件中设置有效的LLM_API_KEY后再运行此脚本\n")
        return 1
    
    print("=== 开始使用LLM顺序修复失败的提取任务 ===")
    logger.info("=== 开始使用LLM顺序修复失败的提取任务 ===")
    
    # 创建输出目录
    fixed_output_dir = os.path.join(config.OUT_DIR + "_llm_fixed")
    if not os.path.exists(fixed_output_dir):
        os.makedirs(fixed_output_dir)
        logger.info(f"创建输出目录: {fixed_output_dir}")
    
    # 检查失败文件列表
    fail_list_path = args.fail_list if args.fail_list else config.FAIL_LIST_TXT
    if not os.path.exists(fail_list_path):
        logger.error(f"失败文件列表不存在: {fail_list_path}")
        return 1
    
    # 读取失败文件列表
    with open(fail_list_path, "r", encoding="utf-8") as f:
        failed_files = [line.strip() for line in f.readlines() if line.strip()]
    
    if not failed_files:
        logger.info("没有失败的文件需要处理")
        return 0
    
    # 检查有多少文件已经处理过
    existing_count = 0
    for file_name in failed_files:
        txt_name = os.path.splitext(file_name)[0] + ".txt"
        txt_path = os.path.join(fixed_output_dir, txt_name)
        if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
            existing_count += 1
    
    logger.info(f"找到 {len(failed_files)} 个失败的文件需要修复，其中 {existing_count} 个已存在处理结果")
    print(f"找到 {len(failed_files)} 个失败的文件需要修复，其中 {existing_count} 个已存在处理结果")
    
    if args.dry_run:
        logger.info("干运行模式，不执行实际处理")
        return 0
    
    try:
        # 开始处理
        start_time = time.time()
        
        # 顺序处理
        success_count, fail_count = process_files_sequential(failed_files, fixed_output_dir)
        
        elapsed_time = time.time() - start_time
        
        # 输出处理结果
        logger.info(f"修复处理完成! 成功: {success_count}, 失败: {fail_count}, 耗时: {elapsed_time:.2f}秒")
        print(f"\n修复处理完成!")
        print(f"成功: {success_count}, 失败: {fail_count}")
        print(f"耗时: {elapsed_time:.2f}秒")
        print(f"处理速度: {success_count/elapsed_time*60:.1f} 文件/分钟")
        
        # 更新失败列表（仍然失败的文件）
        still_failed = []
        for file_name in failed_files:
            txt_name = os.path.splitext(file_name)[0] + ".txt"
            txt_path = os.path.join(fixed_output_dir, txt_name)
            if not os.path.exists(txt_path) or os.path.getsize(txt_path) == 0:
                still_failed.append(file_name)
        
        if still_failed:
            still_failed_path = os.path.join(fixed_output_dir, "still_failed.txt")
            with open(still_failed_path, "w", encoding="utf-8") as f:
                for file_name in still_failed:
                    f.write(f"{file_name}\n")
            logger.info(f"仍有 {len(still_failed)} 个文件未能修复，已保存到 {still_failed_path}")
            print(f"仍有 {len(still_failed)} 个文件未能修复，已保存到 {still_failed_path}")
        else:
            logger.info("所有文件都已成功修复！")
            print("所有文件都已成功修复！")
    
    except KeyboardInterrupt:
        logger.warning("用户中断处理")
        print("\n用户中断处理。已处理的结果已保存。")
        return 130  # SIGINT的标准返回码
    except Exception as e:
        logger.error(f"处理过程出错: {str(e)}")
        print(f"\n处理过程出错: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
