# WhisperX Web 应用配置文件

import os

class Config:
    # 基本配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    DEBUG = True
    
    # 文件上传配置
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024 * 1024  # 16GB
    UPLOAD_FOLDER = 'uploads'
    OUTPUT_FOLDER = 'outputs'
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm', 'm4v'}
    
    # WhisperX 配置
    DEFAULT_MODEL = 'large-v3'
    DEFAULT_FORMAT = 'srt'
    BATCH_SIZE = 16
    
    # 可用模型列表
    AVAILABLE_MODELS = [
        "tiny", "base", "small", "medium", "large", "large-v2", "large-v3"
    ]
    
    # 输出格式
    OUTPUT_FORMATS = {
        'srt': 'SubRip 字幕格式',
        'vtt': 'WebVTT 字幕格式', 
        'json': 'JSON 完整数据',
        'txt': '纯文本格式'
    }
    
    # 服务器配置
    HOST = '0.0.0.0'
    PORT = 5000
    
    # 性能配置
    USE_CUDA = True  # 是否使用CUDA
    COMPUTE_TYPE_GPU = 'float16'  # GPU计算类型
    COMPUTE_TYPE_CPU = 'int8'     # CPU计算类型 