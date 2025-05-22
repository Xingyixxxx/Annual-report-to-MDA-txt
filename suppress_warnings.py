"""
警告抑制工具 - 用于在程序启动时屏蔽各种库的警告信息
"""

import warnings
import logging
import os

def suppress_all_warnings():
    """抑制所有警告信息"""
    # 抑制Python内置警告
    warnings.filterwarnings('ignore')
    
    # 抑制PDF相关库的警告
    logging.getLogger('pdfminer').setLevel(logging.ERROR)
    logging.getLogger('PIL').setLevel(logging.ERROR)
    logging.getLogger('pdf').setLevel(logging.ERROR)
    logging.getLogger('fontTools').setLevel(logging.ERROR)
    logging.getLogger('pdfplumber').setLevel(logging.ERROR)
    
    # 抑制OCR相关警告
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 用于TensorFlow后端
    
    # 抑制其他常见库的警告
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('requests').setLevel(logging.ERROR)
    
    # 设置终端输出缓冲，减少实时输出
    os.environ['PYTHONUNBUFFERED'] = '0'
    
    # 禁用图片处理警告
    os.environ['PYTHONWARNINGS'] = 'ignore'
    
    print("已启用警告抑制模式，只会显示错误信息")
