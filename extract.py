import os
import re
import logging
import pdfplumber
import pytesseract
from PIL import Image
import concurrent.futures
import time
from tqdm import tqdm
import config
import io
import sys
from debug_extractor import check_title_format

# 添加警告抑制
if config.SUPPRESS_WARNINGS:
    from suppress_warnings import suppress_all_warnings
    suppress_all_warnings()

def setup_logging():
    """设置日志系统"""
    if not os.path.exists(os.path.dirname(config.LOG_PATH)):
        os.makedirs(os.path.dirname(config.LOG_PATH))
        
    handlers = [logging.FileHandler(config.LOG_PATH)]
    if config.LOG_TO_CONSOLE:
        handlers.append(logging.StreamHandler())
        
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)

def extract_text_with_ocr(image):
    """使用OCR从图像提取文本"""
    try:
        text = pytesseract.image_to_string(image, lang=config.OCR_LANG)
        return text
    except Exception as e:
        logging.debug(f"OCR处理失败: {e}")  # 降级为DEBUG级别
        return ""

def extract_page_text(page):
    """提取页面文本，必要时使用OCR"""
    text = page.extract_text() or ""
    
    # 如果文本量太少且启用了OCR，尝试使用OCR
    if len(text) < config.OCR_THRESHOLD and config.USE_OCR:
        try:
            img = page.to_image()
            pil_img = img.original
            ocr_text = extract_text_with_ocr(pil_img)
            if len(ocr_text) > len(text):
                return ocr_text
        except Exception as e:
            logging.debug(f"页面OCR转换失败: {e}")  # 降级为DEBUG级别
    
    return text

def find_mdna_section(pdf_path):
    """
    在PDF中定位并提取管理层讨论与分析(MDA)部分
    使用严格的模式匹配以提高提取精度
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) == 0:
                raise ValueError("PDF没有页面")
                
            start_page = -1
            end_page = -1
            
            # 仅搜索前MAX_PAGES_TO_TRY页以提高效率
            max_pages = min(len(pdf.pages), config.MAX_PAGES_TO_TRY)
            
            # 第一遍：使用严格的模式查找章节开始
            for i in range(max_pages):
                text = extract_page_text(pdf.pages[i])
                if not text:
                    continue
                    
                lines = text.split('\n')
                for line_num, line in enumerate(lines):
                    if config.START_REGEX.search(line):
                        start_page = i
                        logging.info(f"在第{i+1}页找到MDA章节开始: '{line}'")
                        break
                
                if start_page >= 0:
                    break
            
            # 由于使用严格匹配，如果未找到开始位置，则直接返回失败
            if start_page == -1:
                raise ValueError("未找到MDA章节开始位置")
            
            # 从MDA章节开始后查找章节结束
            for i in range(start_page + 1, len(pdf.pages)):
                text = extract_page_text(pdf.pages[i])
                if not text:
                    continue
                    
                lines = text.split('\n')
                for line in lines:
                    if config.END_REGEX.search(line) and i > start_page + 1:
                        end_page = i - 1  # 上一页结束
                        logging.info(f"在第{i+1}页找到MDA章节结束: '{line}'")
                        break
                
                if end_page >= 0:
                    break
            
            # 如果没找到结束标志，默认取后20页或文件末尾
            if end_page == -1:
                end_page = min(start_page + 20, len(pdf.pages) - 1)
                logging.info(f"未找到明确的MDA结束标志，默认使用{end_page+1}页作为结束")
            
            # 提取MDA章节文本
            mdna_text = ""
            for i in range(start_page, end_page + 1):
                page_text = extract_page_text(pdf.pages[i])
                mdna_text += page_text + "\n\n"
            
            return mdna_text
    
    except Exception as e:
        logging.error(f"处理 {os.path.basename(pdf_path)} 时出错: {e}")
        return None

def capture_debug_output(pdf_path, pages=10):
    """捕获debug_extractor的输出并返回文本结果"""
    # 保存原始stdout
    old_stdout = sys.stdout
    # 创建一个io对象来捕获输出
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    
    try:
        # 调用debug_extractor的功能
        check_title_format(pdf_path, pages)
        # 获取捕获的输出
        output = new_stdout.getvalue()
        return output
    finally:
        # 恢复stdout
        sys.stdout = old_stdout

def find_mdna_section_with_position(pdf_path, start_page, end_page, start_keyword=None):
    """
    根据精确位置提取MDA部分文本
    :param pdf_path: PDF文件路径
    :param start_page: 起始页码(基于0的索引)
    :param end_page: 结束页码(基于0的索引)
    :param start_keyword: 可选的起始关键词，用于在页面内精确定位
    :return: 提取的文本
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) == 0:
                raise ValueError("PDF没有页面")
            
            # 确保页码在合法范围内
            start_page = max(0, min(start_page, len(pdf.pages)-1))
            end_page = max(start_page, min(end_page, len(pdf.pages)-1))
            
            # 提取MDA章节文本
            mdna_text = ""
            for i in range(start_page, end_page + 1):
                page_text = extract_page_text(pdf.pages[i])
                
                # 如果是第一页且有关键词，尝试定位到关键词后开始
                if i == start_page and start_keyword and start_keyword in page_text:
                    lines = page_text.split('\n')
                    found = False
                    filtered_lines = []
                    for line in lines:
                        if not found and start_keyword in line:
                            found = True
                            filtered_lines.append(line)
                        elif found:
                            filtered_lines.append(line)
                    
                    if filtered_lines:
                        page_text = '\n'.join(filtered_lines)
                
                mdna_text += page_text + "\n\n"
            
            return mdna_text
    
    except Exception as e:
        logging.error(f"根据位置提取文本失败 {os.path.basename(pdf_path)}: {e}")
        return None

def process_pdf(pdf_file):
    """处理单个PDF文件，供多线程使用"""
    pdf_path = os.path.join(config.PDF_DIR, pdf_file)
    txt_path = os.path.join(config.OUT_DIR, os.path.splitext(pdf_file)[0] + ".txt")
    debug_path = os.path.join(config.DEBUG_REPORTS_DIR, os.path.splitext(pdf_file)[0] + "_debug.txt")
    
    # 确保调试报告目录存在
    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
    
    try:
        if os.path.exists(txt_path):
            return {"file": pdf_file, "status": "已存在，跳过", "success": True}
        
        mdna_text = find_mdna_section(pdf_path)
        
        if mdna_text:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(mdna_text)
            return {"file": pdf_file, "status": "成功", "success": True}
        else:
            # 提取失败时，运行调试分析
            logging.warning(f"{pdf_file} 提取失败，正在运行调试分析...")
            debug_output = capture_debug_output(pdf_path)
            
            # 将调试输出保存到文件
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(debug_output)
            
            logging.info(f"调试分析已保存到 {debug_path}")
            return {"file": pdf_file, "status": "失败：未能提取MDA内容，已生成调试报告", "success": False}
            
    except Exception as e:
        logging.error(f"处理 {pdf_file} 失败: {e}")
        
        # 在异常情况下也尝试运行调试
        try:
            debug_output = capture_debug_output(pdf_path)
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(f"处理错误: {str(e)}\n\n")
                f.write(debug_output)
            logging.info(f"错误调试分析已保存到 {debug_path}")
        except Exception as debug_error:
            logging.error(f"生成调试报告失败: {debug_error}")
            
        return {"file": pdf_file, "status": f"失败：{str(e)}", "success": False}

def main():
    """主函数：处理所有PDF文件"""
    logger = setup_logging()
    
    # 创建输出目录
    if not os.path.exists(config.OUT_DIR):
        os.makedirs(config.OUT_DIR)
    
    # 获取PDF文件列表
    pdf_files = [f for f in os.listdir(config.PDF_DIR) if f.endswith('.pdf')]
    total_files = len(pdf_files)
    logger.info(f"找到 {total_files} 个PDF文件待处理")
    
    # 失败文件列表
    fail_files = []
    
    start_time = time.time()
    
    # 使用进度条
    with tqdm(total=total_files, desc="处理PDF") as pbar:
        # 使用进程池处理PDF文件
        with concurrent.futures.ProcessPoolExecutor(max_workers=config.CPU_COUNT) as executor:
            futures = [executor.submit(process_pdf, pdf_file) for pdf_file in pdf_files]
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                pbar.update(1)
                
                if not result["success"]:
                    fail_files.append(result["file"])
                    logger.error(f"{result['file']}: {result['status']}")
                else:
                    logger.info(f"{result['file']}: {result['status']}")
    
    # 输出处理结果
    elapsed_time = time.time() - start_time
    logger.info(f"处理完成! 总耗时: {elapsed_time:.2f}秒")
    logger.info(f"成功: {total_files - len(fail_files)}, 失败: {len(fail_files)}")
    
    # 保存失败文件列表
    if fail_files:
        with open(config.FAIL_LIST_TXT, "w", encoding="utf-8") as f:
            for file in fail_files:
                f.write(f"{file}\n")
        logger.info(f"失败文件列表已保存至 {config.FAIL_LIST_TXT}")

if __name__ == "__main__":
    main()