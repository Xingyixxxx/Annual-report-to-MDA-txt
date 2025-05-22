#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDF提取失败文件修复工具 - 整合本地处理和API辅助功能

功能:
1. 读取失败文件列表
2. 尝试使用API定位内容位置
3. 根据位置信息从本地PDF文件提取内容
4. 包含请求频率控制、错误重试和异步处理

日期: 2025-05-22
"""

import os
import re
import json
import time
import logging
import argparse
import asyncio
import aiohttp
import concurrent.futures
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import config
import pdfplumber
from extract import find_mdna_section_with_position, extract_page_text

# ===== 数据模型 =====
@dataclass
class PDFPosition:
    """PDF中内容的位置数据"""
    start_page: int
    end_page: int
    start_keyword: Optional[str] = None
    confidence: float = 0.0
    
    def to_dict(self):
        return asdict(self)

@dataclass
class APIResult:
    """API请求结果"""
    success: bool
    file_name: str
    position: Optional[PDFPosition] = None
    error: Optional[str] = None

# ===== 日志设置 =====
def setup_logging():
    """设置日志系统"""
    log_dir = os.path.dirname(config.LOG_PATH)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    handlers = [logging.FileHandler(os.path.join(log_dir, "llm_fix.log"))]
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

# ===== 同步方式处理失败文件 =====
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

        # 调用API
        logger.info(f"正在使用LLM定位文件MDA位置: {file_name}")
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
        
        # 获取响应文本
        result = response.choices[0].message.content
        logger.debug(f"LLM原始响应: {result}")
        
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

def process_failed_file(pdf_path, client, output_dir):
    """处理单个失败的文件"""
    file_name = os.path.basename(pdf_path)
    txt_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + ".txt")
    debug_path = os.path.join(config.DEBUG_REPORTS_DIR, os.path.splitext(file_name)[0] + "_llm_debug.txt")
    
    # 确保调试目录存在
    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
    
    try:
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

# ===== 异步API处理部分 =====
class AsyncAPIProcessor:
    """异步API处理类"""
    
    def __init__(self, 
                max_workers: int = 5, 
                rate_limit_per_minute: int = 30000,  # 更新为30,000 (DeepSeek-V3)
                tokens_per_minute: int = 5000000):   # 更新为5,000,000 (DeepSeek-V3)
        """
        初始化异步处理器
        
        Args:
            max_workers: 最大并行工作数
            rate_limit_per_minute: 每分钟最大请求次数(RPM)，DeepSeek-V3默认为30,000
            tokens_per_minute: 每分钟最大Token数(TPM)，DeepSeek-V3默认为5,000,000
        """
        self.max_workers = max_workers
        self.rate_limit_per_minute = rate_limit_per_minute
        self.tokens_per_minute = tokens_per_minute
        
        # 用于频率限制的信号量
        self.semaphore = asyncio.Semaphore(max_workers)
        
        # 请求计数和时间戳
        self.request_timestamps = []
        
        # Token使用量和时间戳
        self.token_usage_timestamps = []
    
    async def _wait_for_rate_limit(self, estimated_tokens=500):
        """
        等待以遵守频率限制
        
        Args:
            estimated_tokens: 本次请求估计使用的token数量
        """
        now = time.time()
        # 移除过期的时间戳（1分钟前的）
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < 60]
        self.token_usage_timestamps = [(ts, tokens) for ts, tokens in self.token_usage_timestamps if now - ts < 60]
        
        # 计算当前分钟内的总token使用量
        current_token_usage = sum(tokens for _, tokens in self.token_usage_timestamps)
        
        wait_time = 0
        
        # 检查是否达到请求速率限制
        if len(self.request_timestamps) >= self.rate_limit_per_minute:
            # 计算需要等待的时间
            oldest_timestamp = min(self.request_timestamps)
            rpm_wait_time = 60 - (now - oldest_timestamp) + 0.1  # 额外0.1秒作为缓冲
            wait_time = max(wait_time, rpm_wait_time)
            logger.debug(f"达到请求频率限制(RPM)，需要等待 {rpm_wait_time:.2f} 秒")
        
        # 检查是否达到Token使用量限制
        if current_token_usage + estimated_tokens >= self.tokens_per_minute:
            # 简单处理：如果当前窗口内的Token已接近限制，等待一些旧token过期
            tpm_wait_time = 5.0  # 默认等待5秒，让一些token过期
            wait_time = max(wait_time, tpm_wait_time)
            logger.debug(f"接近Token频率限制(TPM)，需要等待 {tpm_wait_time:.2f} 秒")
        
        # 如果需要等待，进行等待
        if wait_time > 0:
            logger.debug(f"频率限制，等待 {wait_time:.2f} 秒")
            await asyncio.sleep(wait_time)
        
        # 添加当前请求时间戳和token估计
        self.request_timestamps.append(time.time())
        self.token_usage_timestamps.append((time.time(), estimated_tokens))
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _make_api_request(self, session, pdf_text, file_name):
        """发送API请求并处理重试逻辑"""
        # 计算估计的token数量（粗略计算，中文约1.5个字符一个token）
        estimated_input_tokens = len(pdf_text) // 2  # 输入文本的估计token数
        estimated_output_tokens = 500  # 输出结果的估计token数
        estimated_total_tokens = min(estimated_input_tokens + estimated_output_tokens, 10000)  # 限制最大值
        
        # 等待频率限制
        await self._wait_for_rate_limit(estimated_total_tokens)
        
        # 构建提示
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

        system_prompt = "你是一个专业的文档定位助手。你只返回JSON格式的位置信息，不返回任何其他文本。"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
            
        headers = {
            "Authorization": f"Bearer {config.LLM_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": config.LLM_MODEL,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 800,
            "stream": False,
            "response_format": {"type": "json_object"}
        }
        
        try:
            async with self.semaphore:
                async with session.post(
                    f"{config.LLM_BASE_URL}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"API错误: {response.status} - {error_text}")
                        raise Exception(f"API请求失败: {response.status} - {error_text}")
                    
                    result = await response.json()
                    return result
        except Exception as e:
            logger.error(f"API请求异常: {str(e)}")
            raise
    
    async def process_pdf(self, session, pdf_path: str) -> APIResult:
        """处理单个PDF文件"""
        file_name = os.path.basename(pdf_path)
        
        try:
            # 提取PDF文本
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pdf_text = await loop.run_in_executor(pool, extract_full_text, pdf_path, 30)
            
            if not pdf_text:
                return APIResult(
                    success=False,
                    file_name=file_name,
                    error="无法提取PDF文本"
                )
            
            # 发送API请求
            response = await self._make_api_request(session, pdf_text, file_name)
            
            # 解析结果
            if "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0]["message"]["content"]
                try:
                    position_data = json.loads(content)
                    position = PDFPosition(
                        start_page=int(position_data.get("start_page", 0)),
                        end_page=int(position_data.get("end_page", 0)),
                        start_keyword=position_data.get("start_keyword"),
                        confidence=float(position_data.get("confidence", 0.7))
                    )
                    
                    if position.start_page <= 0 or position.end_page <= 0:
                        logger.warning(f"无效的页码: {file_name}")
                        position = PDFPosition(
                            start_page=10,  # 默认值
                            end_page=30,    # 默认值
                            confidence=0.3
                        )
                    
                    return APIResult(success=True, file_name=file_name, position=position)
                except json.JSONDecodeError:
                    logger.warning(f"JSON解析失败: {file_name} - {content}")
                    return APIResult(success=False, file_name=file_name, error="JSON解析失败")
            
            return APIResult(success=False, file_name=file_name, error="API响应中无有效内容")
            
        except Exception as e:
            logger.error(f"处理PDF失败 {file_name}: {str(e)}")
            return APIResult(success=False, file_name=file_name, error=str(e))
    
    async def extract_content(self, result: APIResult, pdf_path: str, output_dir: str) -> bool:
        """根据API结果提取内容并保存"""
        if not result.success or not result.position:
            return False
        
        file_name = result.file_name
        txt_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + ".txt")
        
        try:
            # 基于1的页码转换为基于0的索引
            start_page = max(0, result.position.start_page - 1)
            end_page = max(0, result.position.end_page - 1)
            
            # 在线程池中执行阻塞操作
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                content = await loop.run_in_executor(
                    pool,
                    find_mdna_section_with_position,
                    pdf_path,
                    start_page,
                    end_page,
                    result.position.start_keyword
                )
            
            if not content or len(content.strip()) < 200:
                logger.warning(f"提取内容太少: {file_name}")
                return False
            
            # 保存内容
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            logger.info(f"成功提取: {file_name}")
            return True
        
        except Exception as e:
            logger.error(f"提取内容失败: {file_name} - {str(e)}")
            return False
    
    async def process_batch(self, pdf_paths: List[str], output_dir: str):
        """批量处理PDF文件"""
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理统计
        total = len(pdf_paths)
        success_count = 0
        fail_count = 0
        
        connector = aiohttp.TCPConnector(limit=self.max_workers)
        async with aiohttp.ClientSession(connector=connector) as session:
            # 步骤1: 获取位置信息
            position_results = []
            async for result in tqdm_asyncio.as_completed(
                [self.process_pdf(session, pdf_path) for pdf_path in pdf_paths],
                total=total,
                desc="定位内容位置"
            ):
                position_results.append(result)
            
            # 步骤2: 提取内容
            extract_tasks = []
            for result in position_results:
                if result.success and result.position:
                    pdf_path = os.path.join(os.path.dirname(output_dir), result.file_name)
                    extract_tasks.append(self.extract_content(result, pdf_path, output_dir))
            
            results = []
            async for result in tqdm_asyncio.as_completed(
                extract_tasks,
                total=len(extract_tasks),
                desc="提取内容"
            ):
                results.append(result)
                if result:
                    success_count += 1
                else:
                    fail_count += 1
        
        return success_count, fail_count

# ===== 主函数 =====
def process_failed_files_sync(failed_files, output_dir):
    """同步处理失败文件（原方法）"""
    # 创建LLM客户端
    client = create_llm_client()
    if not client:
        logger.error("无法创建LLM客户端，请检查API配置")
        return 0, len(failed_files)
    
    # 处理每个失败的文件
    success_count = 0
    fail_count = 0
    
    with tqdm(total=len(failed_files), desc="修复进度") as pbar:
        for file_name in failed_files:
            pdf_path = os.path.join(config.PDF_DIR, file_name)
            
            if not os.path.exists(pdf_path):
                logger.warning(f"文件不存在: {pdf_path}")
                fail_count += 1
                pbar.update(1)
                continue
            
            success = process_failed_file(pdf_path, client, output_dir)
            if success:
                success_count += 1
            else:
                fail_count += 1
                
            pbar.update(1)
            # 添加短暂延迟避免API限制
            time.sleep(0.5)
    
    return success_count, fail_count

async def process_failed_files_async(failed_files, output_dir, workers=5, rate_limit=30000, tokens_per_min=5000000):
    """
    异步处理失败文件（新方法）
    
    Args:
        failed_files: 失败文件列表
        output_dir: 输出目录
        workers: 并行工作数
        rate_limit: 每分钟API请求限制(RPM)，DeepSeek-V3为30,000
        tokens_per_min: 每分钟Token数限制(TPM)，DeepSeek-V3为5,000,000
    """
    # 准备PDF文件路径
    pdf_paths = [os.path.join(config.PDF_DIR, file_name) for file_name in failed_files]
    
    # 过滤存在的文件
    pdf_paths = [path for path in pdf_paths if os.path.exists(path)]
    
    if not pdf_paths:
        logger.error("没有找到有效的PDF文件")
        return 0, len(failed_files)
    
    # 创建异步处理器
    processor = AsyncAPIProcessor(
        max_workers=workers,
        rate_limit_per_minute=rate_limit,
        tokens_per_minute=tokens_per_min
    )
    
    # 批量处理
    return await processor.process_batch(pdf_paths, output_dir)

def main():
    """主函数：处理所有失败的提取文件"""
    parser = argparse.ArgumentParser(description="PDF提取失败文件修复工具")
    parser.add_argument("--async-mode", action="store_true", help="使用异步处理方式", dest="async_mode")
    parser.add_argument("--workers", type=int, default=5, help="并行工作数")
    parser.add_argument("--rate-limit", type=int, default=30000, help="每分钟API请求限制(RPM)，DeepSeek-V3为30,000")
    parser.add_argument("--tokens-per-min", type=int, default=5000000, help="每分钟Token数限制(TPM)，DeepSeek-V3为5,000,000")
    parser.add_argument("--dry-run", action="store_true", help="仅检查配置，不实际执行")
    args = parser.parse_args()

    # 验证API配置
    if config.LLM_API_KEY == "你的API密钥" and not args.dry_run:
        print("\n错误: API密钥未配置!")
        print("请在config.py文件中设置有效的LLM_API_KEY后再运行此脚本\n")
        return 1
    
    print("=== 开始使用LLM修复失败的提取任务 ===")
    logger.info("=== 开始使用LLM修复失败的提取任务 ===")
    
    # 创建输出目录
    fixed_output_dir = os.path.join(config.OUT_DIR + "_llm_fixed")
    if not os.path.exists(fixed_output_dir):
        os.makedirs(fixed_output_dir)
    
    # 检查失败文件列表
    if not os.path.exists(config.FAIL_LIST_TXT):
        logger.error(f"失败文件列表不存在: {config.FAIL_LIST_TXT}")
        return 1
    
    # 读取失败文件列表
    with open(config.FAIL_LIST_TXT, "r", encoding="utf-8") as f:
        failed_files = [line.strip() for line in f.readlines() if line.strip()]
    
    if not failed_files:
        logger.info("没有失败的文件需要处理")
        return 0
    
    logger.info(f"找到 {len(failed_files)} 个失败的文件需要修复")
    
    # 根据处理模式选择方法
    start_time = time.time()
    if args.async_mode:
        logger.info(f"使用异步处理方式，并行数: {args.workers}，API限制: RPM={args.rate_limit}/分钟，TPM={args.tokens_per_min}/分钟")
        # 异步处理
        loop = asyncio.get_event_loop()
        success_count, fail_count = loop.run_until_complete(
            process_failed_files_async(failed_files, fixed_output_dir, args.workers, args.rate_limit, args.tokens_per_min)
        )
    else:
        logger.info("使用同步处理方式")
        # 同步处理
        success_count, fail_count = process_failed_files_sync(failed_files, fixed_output_dir)
    
    elapsed_time = time.time() - start_time
    
    # 输出处理结果
    logger.info(f"修复处理完成! 成功: {success_count}, 失败: {fail_count}, 耗时: {elapsed_time:.2f}秒")
    
    # 更新失败列表（仍然失败的文件）
    still_failed = []
    for file_name in failed_files:
        txt_name = os.path.splitext(file_name)[0] + ".txt"
        if not os.path.exists(os.path.join(fixed_output_dir, txt_name)):
            still_failed.append(file_name)
    
    if still_failed:
        with open(os.path.join(fixed_output_dir, "still_failed.txt"), "w", encoding="utf-8") as f:
            for file_name in still_failed:
                f.write(f"{file_name}\n")
        logger.info(f"仍有 {len(still_failed)} 个文件未能修复，已保存到 {os.path.join(fixed_output_dir, 'still_failed.txt')}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
