"""
PDF提取诊断工具 - 用于分析为什么某些文件匹配失败
"""

import os
import re
import pdfplumber
import argparse
from colorama import Fore, Style, init 
import config

# 初始化色彩支持
init()

def extract_page_text(page):
    """从页面提取文本"""
    return page.extract_text() or ""

def check_title_format(pdf_path, show_pages=10):
    """检查PDF中的标题格式，找出为什么没有匹配成功"""
    print(f"\n{Fore.BLUE}===== 分析文件: {os.path.basename(pdf_path)} ====={Style.RESET_ALL}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            max_pages = min(show_pages, len(pdf.pages))
            
            # 显示前N页的每一行，检查可能的标题
            print(f"\n{Fore.YELLOW}--- 前{max_pages}页的潜在标题行 ---{Style.RESET_ALL}")
            
            potential_titles = []
            matched = False
            
            # 输出当前使用的匹配模式
            print(f"\n{Fore.CYAN}当前使用的匹配模式:{Style.RESET_ALL}")
            for i, pattern in enumerate(config.START_PATTERNS):
                print(f"{i+1}. {pattern}")
            print()
            
            for i in range(max_pages):
                text = extract_page_text(pdf.pages[i])
                lines = text.split('\n')
                
                print(f"\n{Fore.CYAN}第{i+1}页:{Style.RESET_ALL}")
                
                for line in lines:
                    # 检查是否包含关键词
                    if ('管理' in line and '讨论' in line) or ('董事' in line and '报告' in line) or ('经营' in line and '分析' in line):
                        if config.START_REGEX.search(line):
                            print(f"{Fore.GREEN}匹配成功: {line}{Style.RESET_ALL}")
                            matched = True
                        else:
                            print(f"{Fore.RED}关键词存在但不匹配模式: {line}{Style.RESET_ALL}")
                            potential_titles.append((i+1, line))
            
            if matched:
                print(f"\n{Fore.GREEN}分析结果: 找到匹配标题，但实际提取可能失败于其他原因{Style.RESET_ALL}")
            else:
                print(f"\n{Fore.YELLOW}分析结果: 未找到严格匹配的标题格式{Style.RESET_ALL}")
                
                if potential_titles:
                    print("\n可能的标题候选:")
                    for page_num, title in potential_titles:
                        print(f"- 第{page_num}页: {title}")
                    print(f"\n{Fore.YELLOW}匹配失败原因:{Style.RESET_ALL}")
                    print("1. 当前正则表达式要求严格匹配: ^第[一二三四五六七八九十0-9]{{1,3}}(?:章|节)\\s*(?:管理层讨论与分析|经营情况讨论与分析|董事会报告|董事局报告)\\s*$")
                    print("2. 您可以修改config.py中的START_PATTERNS，增加更多匹配模式")
                else:
                    print(f"\n{Fore.RED}未发现任何相关标题，可能需要手动检查文件{Style.RESET_ALL}")
    
    except Exception as e:
        print(f"\n{Fore.RED}分析过程中出错: {str(e)}{Style.RESET_ALL}")

def main():
    parser = argparse.ArgumentParser(description="PDF提取诊断工具")
    parser.add_argument("filename", help="要分析的PDF文件名")
    parser.add_argument("-p", "--pages", type=int, default=10, help="要检查的页数 (默认: 10)")
    args = parser.parse_args()
    
    # 构建文件路径
    pdf_path = args.filename
    if not os.path.isabs(pdf_path):
        pdf_path = os.path.join(config.PDF_DIR, pdf_path)
    
    if not os.path.exists(pdf_path):
        print(f"错误: 文件不存在 - {pdf_path}")
        return
    
    # 分析文件
    check_title_format(pdf_path, args.pages)
    
    # 提供建议
    print(f"\n{Fore.BLUE}===== 改进建议 ====={Style.RESET_ALL}")
    print("1. 考虑在config.py中添加更灵活的匹配模式")
    print("2. 对于批量问题，可以先使用LLM修复功能进行处理")
    print("3. 匹配规则示例:")
    print("   r\"^.*?第[一二三四五六七八九十0-9]+.*?(?:管理层|经营情况|董事会).*?(?:讨论|分析|报告).*$\"")

if __name__ == "__main__":
    main()
