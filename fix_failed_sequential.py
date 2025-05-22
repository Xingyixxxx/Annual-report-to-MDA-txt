#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDF提取失败文件修复工具 - 顺序处理版本

功能:
1. 读取失败文件列表
2. 使用API定位内容位置
3. 根据位置信息从本地PDF文件提取内容
4. 逐个处理PDF，不使用并发
5. API请求失败时进行重试

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
from openai import OpenAI # 确保 OpenAI 被导入
import config
from extract import extract_page_text, find_mdna_section_with_position

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
    logger_instance = logging.getLogger(__name__)
    logger_instance.info("日志系统初始化完成。") # 简单日志记录
    return logger_instance

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
            max_pages_to_extract = min(max_pages, len(pdf.pages)) 
            for i in range(max_pages_to_extract):
                page_text = extract_page_text(pdf.pages[i])
                if page_text:
                    text += f"===== 第{i+1}页 =====\n{page_text}\n\n"
            return text
    except Exception as e:
        logger.error(f"PDF提取文本失败 ({pdf_path}): {e}")
        return None

def find_mdna_position_with_llm(client, pdf_text, file_name):
    """使用大模型定位MDA部分位置信息，同步方式，带429错误重试机制"""
    if not client or not pdf_text:
        logger.warning(f"客户端或PDF文本为空，无法定位MDA: {file_name}")
        return None
    
    # 限制文本长度，避免超出API限制
    max_text_len = 15000 
    if len(pdf_text) > max_text_len:
        pdf_text = pdf_text[:max_text_len] + "...(文本已截断)"
    
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
    
    max_retries = 3 # 最大重试次数
    retry_delay = 30  # 秒

    for attempt in range(max_retries):
        try:
            logger.debug(f"API请求 (文件: {file_name}), 尝试次数: {attempt + 1}/{max_retries}")
            
            # 调用API
            logger.info(f"正在使用LLM定位文件MDA位置: {file_name} (尝试 {attempt + 1}/{max_retries})")
            start_time = time.time()
            response = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个专业的文档定位助手。你只返回JSON格式的位置信息，不返回任何其他文本。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500, 
                response_format={"type": "json_object"}
            )
            elapsed = time.time() - start_time
            
            result = response.choices[0].message.content
            logger.debug(f"LLM原始响应 (文件: {file_name}): {result}")
            
            # API响应日志
            if hasattr(response, 'usage') and response.usage and response.usage.total_tokens is not None:
                total_tokens_api = response.usage.total_tokens
                logger.info(f"API响应 (文件: {file_name}) - 总计使用 {total_tokens_api} 个token (来自API)，响应时间 {elapsed:.2f} 秒")
            else:
                logger.info(f"API响应 (文件: {file_name}) - 响应时间 {elapsed:.2f} 秒. API未提供token使用情况。")

            # 尝试解析位置信息
            position_info = None
            try:
                position_info = json.loads(result.strip())
                logger.debug(f"成功通过直接JSON解析获取位置信息 (文件: {file_name})")
            except json.JSONDecodeError as e:
                logger.warning(f"JSON解析失败 (文件: {file_name}): {e}. 响应内容: {result}")
            
            # 验证和调整值
            if not position_info or not isinstance(position_info, dict) or \
               not position_info.get('start_page') or not position_info.get('end_page'):
                logger.warning(f"无法从LLM响应解析有效位置信息或关键字段缺失，将使用默认值 (文件: {file_name})。响应: {result}")
                position_info = {
                    "start_page": 10, # 默认值
                    "end_page": 30,   # 默认值
                    "start_keyword": "未知 (LLM解析失败)",
                    "confidence": 0.3 #较低置信度
                }
            else:
                position_info.setdefault('start_keyword', "未知 (LLM未提供)")
                position_info.setdefault('confidence', 0.5) 

            # 类型转换和校验
            try:
                position_info['start_page'] = int(position_info.get('start_page', 10))
                position_info['end_page'] = int(position_info.get('end_page', 30))
                position_info['confidence'] = float(position_info.get('confidence', 0.3))
            except (ValueError, TypeError) as e:
                logger.warning(f"LLM返回的位置信息字段类型无效，使用默认值 (文件: {file_name}): {e}. 原始值: {position_info}")
                position_info['start_page'] = 10
                position_info['end_page'] = 30
                position_info['confidence'] = 0.3
            
            return position_info
            
        except OpenAI.RateLimitError as e: 
            logger.warning(f"API速率限制错误 (429) (文件: {file_name}): {e}. 尝试 {attempt + 1}/{max_retries}. 将在 {retry_delay} 秒后重试...")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logger.error(f"达到最大重试次数 ({max_retries}) 后仍遇速率限制，放弃处理文件: {file_name}")
                return None 
        except OpenAI.APIError as e: 
            logger.error(f"LLM API调用时发生错误 (文件: {file_name}): {e} (类型: {type(e)}). 尝试 {attempt + 1}/{max_retries}.")
            # 对于其他API错误，可以考虑不同的重试策略或延迟
            if attempt < max_retries - 1:
                # logger.info(f"将在 {retry_delay / 2:.1f} 秒后重试 (非速率限制错误)...")
                # time.sleep(retry_delay / 2) 
                # 也可以选择不重试或使用与429相同的延迟
                logger.info(f"将在 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                logger.error(f"达到最大重试次数 ({max_retries}) 后仍遇API错误，放弃处理文件: {file_name}")
                return None
        except Exception as e: 
            logger.error(f"处理LLM请求时发生未知错误 (文件: {file_name}): {e} (类型: {type(e)}).")
            # 对于未知错误，可能不立即重试，或者只重试一次
            if attempt < 1 : # 例如，只为未知错误重试一次
                 logger.info(f"将为未知错误尝试重试一次，在 {retry_delay / 2:.1f} 秒后...")
                 time.sleep(retry_delay / 2)
            else:
                logger.error(f"未知错误发生，已尝试重试，放弃处理文件: {file_name}")
                return None
    
    logger.error(f"所有重试均失败，无法为文件获取LLM位置信息: {file_name}")
    return None

def process_file(pdf_path, client, output_dir):
    """顺序处理单个PDF文件"""
    file_name = os.path.basename(pdf_path)
    txt_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + ".txt")
    debug_path = os.path.join(config.DEBUG_REPORTS_DIR, os.path.splitext(file_name)[0] + "_llm_debug.txt")
    
    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
    
    try:
        if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
            logger.info(f"文件已存在，跳过处理: {file_name}")
            return True
            
        pdf_text = extract_full_text(pdf_path, max_pages=30) # max_pages可以根据需要调整
        if not pdf_text:
            logger.error(f"无法从PDF提取文本: {file_name}")
            # 写入调试信息表明提取失败
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(f"无法从PDF提取文本用于LLM定位: {file_name}\n")
            return False
        
        position_info = find_mdna_position_with_llm(client, pdf_text, file_name)
        if not position_info:
            logger.error(f"无法获取MDA位置信息: {file_name}")
            with open(debug_path, "w", encoding="utf-8") as f: # Overwrite or append based on desired behavior
                f.write(f"LLM未能提供MDA位置信息: {file_name}\n")
            return False
            
        with open(debug_path, "w", encoding="utf-8") as f: # Start fresh debug file for this attempt
            f.write(f"MDA位置信息 (来自LLM):\n")
            f.write(f"  开始页: {position_info.get('start_page', '未知')}\n")
            f.write(f"  结束页: {position_info.get('end_page', '未知')}\n")
            f.write(f"  起始关键词: {position_info.get('start_keyword', '未知')}\n")
            f.write(f"  置信度: {position_info.get('confidence', 0.0):.2f}\n")
        
        start_page_idx = max(0, position_info.get('start_page', 10) - 1) # Default to 10 (page 9 index)
        end_page_idx = max(0, position_info.get('end_page', 30) - 1)   # Default to 30 (page 29 index)
        
        if start_page_idx > end_page_idx:
            logger.warning(f"LLM返回的开始页 ({start_page_idx+1}) 大于结束页 ({end_page_idx+1}) for {file_name}. 尝试交换或使用默认范围.")
            if position_info.get('confidence', 0.0) < 0.5 : # Example threshold
                 logger.info(f"置信度低或页码无效，将尝试更广的默认范围 for {file_name}")
                 start_page_idx = 5 # example broad start
                 end_page_idx = 50 # example broad end
            else: # Swap if confidence is reasonable
                 start_page_idx, end_page_idx = end_page_idx, start_page_idx


        mda_text = find_mdna_section_with_position(
            pdf_path, 
            start_page_idx,
            end_page_idx,
            position_info.get('start_keyword')
        )
        
        min_mda_length = 100 # 最小有效内容长度
        if not mda_text or len(mda_text.strip()) < min_mda_length:
            logger.warning(f"根据LLM位置信息无法提取有效内容 (长度 < {min_mda_length}): {file_name}. 尝试备用方法.")
            with open(debug_path, "a", encoding="utf-8") as f:
                f.write("\n提取失败或内容太少，尝试备用方法 (扩大范围)...\n") # Corrected newline character
            
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    fallback_start = max(0, start_page_idx - 5) # 扩大范围
                    fallback_end = min(len(pdf.pages) - 1, end_page_idx + 10) # 扩大范围
                    
                    if fallback_start > fallback_end: # Ensure valid range
                        fallback_start = 0
                        fallback_end = min(len(pdf.pages) -1, 50) # Default broad range if still invalid

                    logger.info(f"备用方法：从页 {fallback_start+1} 到 {fallback_end+1} 提取 for {file_name}")
                    fallback_text_list = []
                    for i in range(fallback_start, fallback_end + 1):
                        page_content = extract_page_text(pdf.pages[i])
                        if page_content:
                             fallback_text_list.append(page_content)
                    
                    mda_text = "\n\n".join(fallback_text_list)
                    
                    with open(debug_path, "a", encoding="utf-8") as f:
                        f.write(f"使用备用方法提取内容 (页 {fallback_start+1}-{fallback_end+1}). 内容长度: {len(mda_text)}\n")
            except Exception as e:
                logger.error(f"备用方法提取失败 ({file_name}): {e}")
                with open(debug_path, "a", encoding="utf-8") as f:
                    f.write(f"备用方法提取失败: {e}\n")
                # Do not return False yet, check mda_text from primary attempt or if fallback produced something
        
        if not mda_text or len(mda_text.strip()) < min_mda_length: # Check again after potential fallback
            logger.error(f"所有方法都无法提取足够内容 (长度 < {min_mda_length}): {file_name}")
            with open(debug_path, "a", encoding="utf-8") as f:
                f.write(f"所有方法均提取内容不足或失败.\n")
            return False
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(mda_text)
        
        with open(debug_path, "a", encoding="utf-8") as f:
            f.write(f"\n提取成功! 内容长度: {len(mda_text)} 字符. 保存到: {txt_path}\n")
        
        logger.info(f"成功修复并提取: {file_name}")
        return True
        
    except Exception as e:
        logger.error(f"处理文件时发生意外错误: {file_name}, 错误: {e}", exc_info=True)
        try:
            with open(debug_path, "a", encoding="utf-8") as f: # Append to existing debug if error happens late
                f.write(f"\n处理文件时发生意外错误: {str(e)}\n")
        except Exception: # Catch specific exception if possible, or general Exception
            pass # Ignore if cannot write to debug file
        return False

def process_files_sequential(failed_files, output_dir):
    """顺序处理失败文件列表"""
    client = create_llm_client()
    if not client:
        logger.error("无法创建LLM客户端，请检查API配置。中止处理。")
        return 0, len(failed_files) # No successes, all remaining are fails
    
    os.makedirs(output_dir, exist_ok=True)
    
    files_to_process = []
    skipped_count = 0
    initial_fail_count = len(failed_files)

    for file_name in failed_files:
        pdf_path = os.path.join(config.PDF_DIR, file_name)
        txt_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + ".txt")
        
        if not os.path.exists(pdf_path):
            logger.warning(f"PDF文件不存在，跳过: {pdf_path}")
            skipped_count +=1 # Counts as skipped, will reduce fail_count later
            continue
            
        if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
            logger.info(f"输出文件已存在且非空，跳过: {txt_path}")
            skipped_count += 1
            continue
            
        files_to_process.append(file_name)
    
    if skipped_count > 0:
        logger.info(f"跳过 {skipped_count} 个文件 (源PDF不存在或输出已存在).")
    
    if not files_to_process:
        logger.info("没有需要处理的文件。")
        # All files were skipped, so success is skipped_count, fail is 0 from processing perspective
        return skipped_count, 0 
    
    logger.info(f"需要处理 {len(files_to_process)} 个文件 (总共 {initial_fail_count} 个失败文件，已跳过 {skipped_count}).")
    print(f"需要处理 {len(files_to_process)} 个文件。")
    
    current_success_count = 0
    current_fail_count = 0
    
    with tqdm(total=len(files_to_process), desc="处理PDF") as pbar:
        for file_name in files_to_process:
            pdf_path = os.path.join(config.PDF_DIR, file_name)
            
            success = process_file(pdf_path, client, output_dir)
            
            if success:
                current_success_count += 1
            else:
                current_fail_count += 1
                
            pbar.update(1)
            
            # Consider if this sleep is still needed. 
            # If API calls are frequent even with 429 retries, a small delay can be a good general measure.
            # If 429s are handled well, this might be less critical.
            time.sleep(config.POST_REQUEST_DELAY_S) # Using a config value for delay
    
    # Total successes = newly processed + skipped (already successful or non-existent source)
    # Total fails = newly failed
    return current_success_count + skipped_count, current_fail_count

def main():
    """主函数：处理所有失败的提取文件"""
    parser = argparse.ArgumentParser(description="PDF提取失败文件修复工具 - 顺序处理版本")
    parser.add_argument("--fail-list", type=str, help="自定义失败文件列表路径")
    parser.add_argument("--dry-run", action="store_true", help="仅检查配置，不实际执行")
    args = parser.parse_args()

    if config.LLM_API_KEY == "你的API密钥" and not args.dry_run:
        # Logger might not be fully set up here if there's an early exit
        print("\n错误: API密钥未配置!")
        print("请在config.py文件中设置有效的LLM_API_KEY后再运行此脚本\n")
        logger.error("API密钥未配置! 请在config.py中设置。") # Attempt to log as well
        return 1
    
    logger.info("=== 开始使用LLM顺序修复失败的提取任务 ===")
    print("=== 开始使用LLM顺序修复失败的提取任务 ===")
    
    fixed_output_dir = os.path.join(config.OUT_DIR + "_llm_fixed")
    if not os.path.exists(fixed_output_dir):
        os.makedirs(fixed_output_dir)
        logger.info(f"创建输出目录: {fixed_output_dir}")
    
    # Ensure debug directory from config exists
    if not os.path.exists(config.DEBUG_REPORTS_DIR):
        os.makedirs(config.DEBUG_REPORTS_DIR)
        logger.info(f"创建调试报告目录: {config.DEBUG_REPORTS_DIR}")

    fail_list_path = args.fail_list if args.fail_list else config.FAIL_LIST_TXT
    if not os.path.exists(fail_list_path):
        logger.error(f"失败文件列表不存在: {fail_list_path}")
        return 1
    
    try:
        with open(fail_list_path, "r", encoding="utf-8") as f:
            failed_files_from_list = [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        logger.error(f"读取失败文件列表失败 ({fail_list_path}): {e}")
        return 1
        
    if not failed_files_from_list:
        logger.info("失败文件列表为空，没有文件需要处理。")
        return 0
    
    # This initial count of "existing" might be redundant if process_files_sequential handles skipping
    # logger.info(f"从列表找到 {len(failed_files_from_list)} 个失败的文件记录。")
    # print(f"从列表找到 {len(failed_files_from_list)} 个失败的文件记录。")
    
    if args.dry_run:
        logger.info("干运行模式，不执行实际处理。将检查文件列表和配置。")
        print("干运行模式，不执行实际处理。")
        # Optionally, list files that would be processed
        return 0
    
    try:
        start_time = time.time()
        
        total_processed_successfully, total_failed_processing = process_files_sequential(failed_files_from_list, fixed_output_dir)
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"修复处理完成! 成功处理/跳过: {total_processed_successfully}, 新失败: {total_failed_processing}, 耗时: {elapsed_time:.2f}秒")
        print(f"\n修复处理完成!")
        print(f"成功处理/跳过: {total_processed_successfully}")
        print(f"处理中失败: {total_failed_processing}")
        print(f"耗时: {elapsed_time:.2f}秒")
        
        if total_processed_successfully > 0 and elapsed_time > 0 : # Avoid division by zero
             # This rate should ideally reflect only newly processed files, not skipped ones.
             # The current `total_processed_successfully` includes skipped files.
             # For a more accurate rate, one might need to track new successes separately.
             # For now, this is a general rate.
            files_actually_processed = total_processed_successfully - sum(1 for f in failed_files_from_list if os.path.exists(os.path.join(fixed_output_dir, os.path.splitext(f)[0] + ".txt")) and os.path.getsize(os.path.join(fixed_output_dir, os.path.splitext(f)[0] + ".txt")) > 0 and f not in getattr(process_files_sequential, "processed_in_this_run", [])) # Approximation
            # A better way would be for process_files_sequential to return new_success_count
            # Assuming total_processed_successfully - skipped_count (if available) or just total_processed_successfully for now.
            logger.info(f"处理速度 (基于成功和跳过): {total_processed_successfully/elapsed_time*60:.1f} 文件/分钟 (可能包含已跳过文件)")
            print(f"处理速度 (基于成功和跳过): {total_processed_successfully/elapsed_time*60:.1f} 文件/分钟")

        # Update the list of files that are still considered "failed"
        still_failed_after_run = []
        for file_name in failed_files_from_list: # Iterate original list
            txt_name = os.path.splitext(file_name)[0] + ".txt"
            txt_path = os.path.join(fixed_output_dir, txt_name)
            pdf_path = os.path.join(config.PDF_DIR, file_name)

            if not os.path.exists(pdf_path): # Source PDF missing, counts as "still failed" in a sense
                still_failed_after_run.append(f"{file_name} (源PDF文件不存在)")
            elif not os.path.exists(txt_path) or os.path.getsize(txt_path) == 0:
                still_failed_after_run.append(file_name)
        
        if still_failed_after_run:
            still_failed_path = os.path.join(fixed_output_dir, "still_failed_after_run.txt")
            with open(still_failed_path, "w", encoding="utf-8") as f:
                for file_name_status in still_failed_after_run:
                    f.write(f"{file_name_status}\n")
            logger.info(f"仍有 {len(still_failed_after_run)} 个文件未能成功生成输出，列表已保存到 {still_failed_path}")
            print(f"仍有 {len(still_failed_after_run)} 个文件未能成功生成输出，列表已保存到 {still_failed_path}")
        else:
            logger.info("所有原始失败列表中的文件现在都有了对应的输出（或源PDF不存在）。")
            print("所有原始失败列表中的文件现在都有了对应的输出（或源PDF不存在）。")
    
    except KeyboardInterrupt:
        logger.warning("用户中断处理")
        print("\n用户中断处理。已处理的结果已保存。")
        return 130 
    except Exception as e:
        logger.error(f"主处理过程发生严重错误: {str(e)}", exc_info=True)
        print(f"\n处理过程发生严重错误: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    # Add a placeholder in config.py if it doesn't exist:
    # POST_REQUEST_DELAY_S = 1.0 (or whatever default you prefer)
    if not hasattr(config, 'POST_REQUEST_DELAY_S'):
        print("提醒: config.py 中缺少 POST_REQUEST_DELAY_S 设置，默认为1.0秒。建议添加此配置。")
        config.POST_REQUEST_DELAY_S = 1.0 
    if not hasattr(config, 'DEBUG_REPORTS_DIR'):
        print("提醒: config.py 中缺少 DEBUG_REPORTS_DIR 设置，默认为 'debug_llm_reports'。建议添加此配置。")
        config.DEBUG_REPORTS_DIR = "debug_llm_reports"


    sys.exit(main())
