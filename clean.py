import os
import re
import logging
from tqdm import tqdm
import config

def setup_logging():
    """设置日志系统"""
    handlers = [logging.FileHandler(os.path.join(config.OUT_DIR, "clean.log"))]
    if config.LOG_TO_CONSOLE:
        handlers.append(logging.StreamHandler())
        
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)

def clean_text(text):
    """清理文本中的常见问题"""
    # 移除页码
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # 移除多余空白行
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # 移除页眉页脚
    text = re.sub(r'^\s*(.*年度报告|年年度报告全文|.*公司年度报告)\s*$', '', text, flags=re.MULTILINE)
    
    # 移除特殊字符
    text = re.sub(r'[□◇▲●■★△▼]', '', text)
    
    # 简单断行处理 - 尝试处理不正确的断行
    lines = text.split('\n')
    for i in range(len(lines)-1):
        if lines[i] and lines[i+1] and not lines[i].endswith(('。', '；', '：', '，', '！', '?', '、', '"', '）')):
            # 检查是否应该合并这两行
            if not lines[i+1].startswith(('一、', '二、', '三、', '四、', '五、', '1、', '2、', '3、')):
                lines[i] = lines[i] + lines[i+1]
                lines[i+1] = ''
    
    return '\n'.join(line for line in lines if line.strip())

def main():
    """清理所有提取的文本文件"""
    logger = setup_logging()
    
    # 创建清洁输出目录
    clean_dir = os.path.join(config.OUT_DIR + "_cleaned")
    if not os.path.exists(clean_dir):
        os.makedirs(clean_dir)
    
    # 获取所有文本文件
    txt_files = [f for f in os.listdir(config.OUT_DIR) if f.endswith('.txt')]
    logger.info(f"找到 {len(txt_files)} 个文本文件待清理")
    
    # 使用进度条处理文件
    with tqdm(total=len(txt_files), desc="清理文本") as pbar:
        for txt_file in txt_files:
            try:
                input_path = os.path.join(config.OUT_DIR, txt_file)
                output_path = os.path.join(clean_dir, txt_file)
                
                # 读取原文本
                with open(input_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # 清理文本
                cleaned_text = clean_text(text)
                
                # 保存清理后的文本
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                
                logger.info(f"已清理: {txt_file}")
            except Exception as e:
                logger.error(f"清理 {txt_file} 失败: {e}")
            
            pbar.update(1)
    
    logger.info(f"清理完成! 结果保存在 {clean_dir}")

if __name__ == "__main__":
    main()