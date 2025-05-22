import os
import re
import logging
import pdfplumber
import time
import json
import sys
import io
from tqdm import tqdm
from openai import OpenAI
import config
from extract import find_mdna_section_with_position, extract_page_text

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

def create_llm_client():
    """创建大模型API客户端"""
    try:
        return OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    except Exception as e:
        logging.error(f"创建LLM客户端失败: {e}")
        return None

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
        logging.error(f"PDF提取文本失败: {e}")
        return None

def find_mdna_position_with_llm(client, pdf_text, file_name):
    """使用大模型定位MDA部分位置信息，不提取内容"""
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
  "start_keyword": "找到的标题完整文本"
}}

只返回上述格式的位置信息JSON，不要添加其他内容。

文件名: {file_name}

年报文本:
{pdf_text}
"""

        # 调用API
        logging.info(f"正在使用LLM定位文件MDA位置: {file_name}")
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个专业的文档定位助手。你只返回JSON格式的位置信息，不返回任何其他文本。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # 降低温度以获得更精确的回答
            max_tokens=500    # 位置信息很简短，减少token消耗
        )
        
        # 获取响应文本
        result = response.choices[0].message.content
        logging.debug(f"LLM原始响应: {result}")
        
        # 尝试多种方式解析位置信息
        position_info = None
        
        # 方法1: 直接尝试解析整个响应为JSON
        try:
            position_info = json.loads(result.strip())
            logging.debug("成功通过直接JSON解析获取位置信息")
        except:
            logging.debug("直接JSON解析失败，尝试正则提取")
        
        # 方法2: 使用正则表达式提取可能的JSON部分
        if not position_info:
            try:
                # 尝试多种可能的JSON模式
                json_patterns = [
                    r'({[\s\S]*})',  # 任何花括号包围的内容
                    r'{\s*"start_page"\s*:\s*(\d+)\s*,\s*"end_page"\s*:\s*(\d+)[\s\S]*}',  # 提取关键字段
                ]
                
                for pattern in json_patterns:
                    match = re.search(pattern, result)
                    if match:
                        json_str = match.group(1)
                        try:
                            position_info = json.loads(json_str)
                            logging.debug(f"通过正则表达式'{pattern}'成功提取JSON")
                            break
                        except:
                            continue
            except Exception as e:
                logging.debug(f"正则提取JSON失败: {e}")
        
        # 方法3: 手动解析关键信息
        if not position_info:
            try:
                # 尝试直接提取数字和关键词
                start_match = re.search(r'"start_page"\s*:\s*(\d+)', result)
                end_match = re.search(r'"end_page"\s*:\s*(\d+)', result)
                keyword_match = re.search(r'"start_keyword"\s*:\s*"([^"]+)"', result)
                
                if start_match and end_match:
                    position_info = {
                        "start_page": int(start_match.group(1)),
                        "end_page": int(end_match.group(1)),
                        "start_keyword": keyword_match.group(1) if keyword_match else None
                    }
                    logging.debug("通过直接正则匹配关键字段构建位置信息")
            except Exception as e:
                logging.debug(f"手动解析关键信息失败: {e}")
        
        # 方法4: 最后的回退策略 - 在文本中查找页码和关键词
        if not position_info:
            try:
                # 尝试从文本中找到提及的页码
                page_matches = re.findall(r'第\s*(\d+)\s*页', result)
                page_numbers = [int(p) for p in page_matches]
                keywords = ["管理层讨论与分析", "经营情况讨论与分析", "董事会报告"]
                
                start_page = min(page_numbers) if page_numbers else 10  # 默认从第10页开始
                end_page = max(page_numbers) + 10 if page_numbers else start_page + 20  # 默认范围
                
                # 尝试找到提及的关键词
                start_keyword = None
                for keyword in keywords:
                    if keyword in result:
                        start_keyword = keyword
                        break
                
                position_info = {
                    "start_page": start_page,
                    "end_page": end_page,
                    "start_keyword": start_keyword
                }
                logging.debug("通过回退策略构建位置信息")
            except Exception as e:
                logging.debug(f"回退策略失败: {e}")
        
        # 最终检查和调整
        if position_info:
            # 验证必要的字段
            if 'start_page' in position_info and 'end_page' in position_info:
                # 调整为0-based索引 (pdfplumber使用0-based)
                position_info['start_page'] = max(0, position_info['start_page'] - 1)
                position_info['end_page'] = max(0, position_info['end_page'] - 1)
                
                # 确保end_page > start_page
                if position_info['end_page'] <= position_info['start_page']:
                    position_info['end_page'] = position_info['start_page'] + 15  # 默认跨度
                
                return position_info
        
        # 如果所有方法都失败，返回一个默认值
        logging.warning(f"无法解析位置信息，使用默认值: {result}")
        return {
            "start_page": 9,  # 默认从第10页开始(0-based)
            "end_page": 29,   # 默认到第30页结束
            "start_keyword": None
        }
            
    except Exception as e:
        logging.error(f"LLM API调用失败: {e}")
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
            f.write(f"开始页: {position_info.get('start_page', 'unknown')+1}\n")
            f.write(f"结束页: {position_info.get('end_page', 'unknown')+1}\n")
            f.write(f"起始关键词: {position_info.get('start_keyword', 'unknown')}\n")
        
        # 使用位置信息提取MDA内容
        mda_text = find_mdna_section_with_position(
            pdf_path, 
            position_info.get('start_page', 0),
            position_info.get('end_page', 20),
            position_info.get('start_keyword', None)
        )
        
        # 检查提取结果
        if not mda_text:
            logging.warning(f"根据位置信息无法提取内容: {file_name}")
            with open(debug_path, "a", encoding="utf-8") as f:
                f.write("\n提取失败，尝试回退到全文段落...\n")
            
            # 回退策略：尝试提取更大范围
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    pages_to_try = range(max(0, position_info.get('start_page', 0) - 5), 
                                        min(len(pdf.pages), position_info.get('end_page', 20) + 10))
                    fallback_text = ""
                    for i in pages_to_try:
                        page_text = extract_page_text(pdf.pages[i])
                        if any(keyword in page_text for keyword in config.MDA_KEYWORDS):
                            fallback_text += page_text + "\n\n"
                    
                    if len(fallback_text.strip()) > 200:  # 确保提取了足够的文本
                        mda_text = fallback_text
                        with open(debug_path, "a", encoding="utf-8") as f:
                            f.write("回退提取成功\n")
            except Exception as e:
                logging.error(f"回退提取失败: {e}")
        
        if not mda_text or len(mda_text.strip()) < 100:  # 检查提取结果是否有效
            logging.warning(f"根据位置提取内容失败或内容太少: {file_name}")
            with open(debug_path, "a", encoding="utf-8") as f:
                f.write("\n提取失败或内容太少\n")
            return False
        
        # 保存结果
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(mda_text)
        
        # 更新调试信息
        with open(debug_path, "a", encoding="utf-8") as f:
            f.write(f"\n提取成功! 内容长度: {len(mda_text)} 字符\n")
        
        logging.info(f"成功修复并提取: {file_name}")
        return True
        
    except Exception as e:
        logging.error(f"处理失败: {file_name}, 错误: {e}")
        # 记录错误到调试文件
        try:
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(f"处理错误: {str(e)}")
        except:
            pass
        return False

def main():
    """主函数：处理所有失败的提取文件"""
    logger = setup_logging()
    logger.info("=== 开始使用LLM修复失败的提取任务 ===")
    
    # 创建输出目录
    fixed_output_dir = os.path.join(config.OUT_DIR + "_llm_fixed")
    if not os.path.exists(fixed_output_dir):
        os.makedirs(fixed_output_dir)
    
    # 检查失败文件列表
    if not os.path.exists(config.FAIL_LIST_TXT):
        logger.error(f"失败文件列表不存在: {config.FAIL_LIST_TXT}")
        return
    
    # 读取失败文件列表
    with open(config.FAIL_LIST_TXT, "r", encoding="utf-8") as f:
        failed_files = [line.strip() for line in f.readlines() if line.strip()]
    
    if not failed_files:
        logger.info("没有失败的文件需要处理")
        return
    
    logger.info(f"找到 {len(failed_files)} 个失败的文件需要修复")
    
    # 创建LLM客户端
    client = create_llm_client()
    if not client:
        logger.error("无法创建LLM客户端，请检查API配置")
        return
    
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
            
            success = process_failed_file(pdf_path, client, fixed_output_dir)
            if success:
                success_count += 1
            else:
                fail_count += 1
                
            pbar.update(1)
            # 添加短暂延迟避免API限制
            time.sleep(0.5)
    
    # 输出处理结果
    logger.info(f"修复处理完成! 成功: {success_count}, 失败: {fail_count}")
    
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

if __name__ == "__main__":
    main()
