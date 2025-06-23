import os
import json
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template, Response, stream_with_context
from flask_cors import CORS
from werkzeug.utils import secure_filename
import whisperx
import torch
import tempfile
import shutil
from pathlib import Path
import ollama
import re
import logging

# 日志配置
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 配置
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024  # 16GB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm', 'm4v'}

# 创建必要的文件夹
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_ollama_models():
    """获取本地可用的Ollama模型列表"""
    try:
        models = ollama.list()
        return [model['name'] for model in models['models']]
    except Exception as e:
        print(f"无法连接到 Ollama 服务: {e}")
        return []

def clean_ollama_codeblock(text):
    """去除Ollama返回内容中的```srt和```包裹"""
    # 去除开头的```srt或```，以及结尾的```
    text = re.sub(r'^```srt\s*', '', text.strip(), flags=re.IGNORECASE)
    text = re.sub(r'^```\s*', '', text.strip(), flags=re.IGNORECASE)
    text = re.sub(r'```\s*$', '', text.strip())
    return text.strip()

def parse_srt_segments(srt_content):
    """解析SRT为段落列表，每段为(dict: number, time, text)"""
    segments = []
    blocks = re.split(r'\n{2,}', srt_content.strip())
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            number = lines[0].strip()
            time = lines[1].strip()
            text = '\n'.join(lines[2:]).strip()
            segments.append({'number': number, 'time': time, 'text': text})
    return segments

def build_srt_from_segments(segments):
    """将分段列表拼接为SRT字符串"""
    srt_lines = []
    for seg in segments:
        srt_lines.append(str(seg['number']))
        srt_lines.append(seg['time'])
        srt_lines.append(seg['text'])
        srt_lines.append('')
    return '\n'.join(srt_lines).strip()

def ollama_translate_text(text, target_lang, model_name):
    """同步调用Ollama翻译一段文本"""
    prompt = f"请将下列内容翻译为{target_lang}，只输出翻译后的文本，不要输出任何解释。\n内容：{text}"
    response = ollama.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response['message']['content'].strip()

def translate_srt_content(srt_content, target_lang, model_name):
    """分段翻译SRT字幕内容，保留编号和时间戳"""
    try:
        logger.info(f"开始分段翻译SRT，目标语言: {target_lang}，模型: {model_name}")
        segments = parse_srt_segments(srt_content)
        translated_segments = []
        for seg in segments:
            logger.info(f"翻译字幕段 {seg['number']} 时间: {seg['time']}")
            translated_text = ollama_translate_text(seg['text'], target_lang, model_name)
            if isinstance(translated_text, str):
                translated_text = translated_text.strip()
            seg_new = {
                'number': seg['number'],
                'time': seg['time'],
                'text': translated_text
            }
            translated_segments.append(seg_new)
        logger.info(f"SRT分段翻译完成，共{len(translated_segments)}段")
        return build_srt_from_segments(translated_segments)
    except Exception as e:
        logger.error(f"分段翻译SRT时出错: {e}")
        return None

def get_available_models():
    """获取可用的WhisperX模型列表"""
    return [
        "tiny", "base", "small", "medium", "large", "large-v2", "large-v3"
    ]

def transcribe_video(video_path, model_name="large-v3", language=None, output_dir=None):
    """使用WhisperX转录音频"""
    try:
        # 设置设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        
        print(f"使用设备: {device}, 计算类型: {compute_type}")
        print(f"加载模型: {model_name}")
        
        # 加载模型
        model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language)
        
        # 转录音频
        print("开始转录...")
        start_time = datetime.now()
        result = model.transcribe(video_path, batch_size=16)
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # 加载音频
        audio = whisperx.load_audio(video_path)
        
        # 对齐时间戳
        print("对齐时间戳...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        
        return result
        
    except Exception as e:
        print(f"转录过程中出错: {str(e)}")
        raise e

def save_subtitles(result, output_path, format_type="srt"):
    """保存字幕文件"""
    try:
        if format_type == "srt":
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(result["segments"], 1):
                    start_time = format_timestamp(segment["start"])
                    end_time = format_timestamp(segment["end"])
                    text = segment["text"].strip()
                    
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")
        
        elif format_type == "vtt":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("WEBVTT\n\n")
                for segment in result["segments"]:
                    start_time = format_timestamp_vtt(segment["start"])
                    end_time = format_timestamp_vtt(segment["end"])
                    text = segment["text"].strip()
                    
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")
        
        elif format_type == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        elif format_type == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                for segment in result["segments"]:
                    f.write(f"{segment['text'].strip()}\n")
        
        return True
        
    except Exception as e:
        print(f"保存字幕文件时出错: {str(e)}")
        return False

def format_timestamp(seconds):
    """格式化时间戳为SRT格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

def format_timestamp_vtt(seconds):
    """格式化时间戳为VTT格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"

# 替换secure_filename，保留空格，只去除危险字符
def safe_subtitle_filename(filename):
    # 只去除/和\等危险字符，保留空格
    filename = re.sub(r'[\\/]+', '', filename)
    return filename

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    """获取可用的模型列表"""
    return jsonify({
        'models': get_available_models(),
        'default': 'large-v3'
    })

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    """单个视频转录接口"""
    try:
        # 检查文件
        if 'video' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件格式'}), 400
        
        # 获取参数
        model_name = request.form.get('model', 'large-v3')
        language = request.form.get('language')
        if not language:
            language = None
        output_format = request.form.get('format', 'srt')
        custom_output_dir = request.form.get('output_dir', None)
        
        # 翻译相关参数
        translate = request.form.get('translate') == 'true'
        target_lang = request.form.get('target_lang', 'zh')
        ollama_model = request.form.get('ollama_model')

        # 生成唯一文件名
        filename = safe_subtitle_filename(file.filename)
        unique_id = str(uuid.uuid4())
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")
        
        # 保存上传的文件
        file.save(video_path)
        
        # 设置输出目录
        if custom_output_dir:
            output_dir = custom_output_dir
        else:
            output_dir = app.config['OUTPUT_FOLDER']
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 转录
        start_time = datetime.now()
        result = transcribe_video(video_path, model_name, language)
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # 生成输出文件名
        base_name, ext = os.path.splitext(filename)
        output_filename = f"{base_name}.{output_format}"
        output_path = os.path.join(output_dir, output_filename)
        
        # 保存字幕文件
        success = save_subtitles(result, output_path, output_format)
        
        if not success:
            return jsonify({'error': '保存字幕文件失败'}), 500
        
        # 如果需要，进行翻译
        translated_output_filename = None
        if success and translate and ollama_model and output_format in ['srt', 'vtt']:
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    original_srt = f.read()
                
                translated_srt = translate_srt_content(original_srt, target_lang, ollama_model)

                if translated_srt:
                    translated_output_filename = f"{base_name}.{target_lang}.{output_format}"
                    translated_output_path = os.path.join(output_dir, translated_output_filename)
                    with open(translated_output_path, 'w', encoding='utf-8') as f:
                        f.write(translated_srt)
            except Exception as e:
                print(f"翻译文件处理失败: {e}")

        # 清理临时文件
        os.remove(video_path)
        
        response_data = {
            'success': True,
            'message': '转录完成',
            'output_file': output_filename,
            'output_path': output_path,
            'language': result.get('language', 'unknown'),
            'segments_count': len(result.get('segments', [])),
            'duration': result.get('segments', [{}])[-1].get('end', 0) if result.get('segments') else 0,
            'processing_time': round(processing_time, 2)
        }
        if translated_output_filename:
            response_data['translated_output_file'] = translated_output_filename

        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'处理失败: {str(e)}'}), 500

@app.route('/api/batch-transcribe', methods=['POST'])
def batch_transcribe():
    """批量视频转录接口 - 使用流式响应"""
    try:
        if 'videos' not in request.files:
            # 这是一个设置错误，所以我们不能流式传输一个错误
            return jsonify({'error': '没有上传文件'}), 400
        
        files = request.files.getlist('videos')
        if not files or files[0].filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        # 获取参数 - 这些参数对所有文件都相同
        model_name = request.form.get('model', 'large-v3')
        language = request.form.get('language')
        if not language:
            language = None
        output_format = request.form.get('format', 'srt')
        custom_output_dir = request.form.get('output_dir', None)

        # 翻译相关参数
        translate = request.form.get('translate') == 'true'
        target_lang = request.form.get('target_lang', 'zh')
        ollama_model = request.form.get('ollama_model')

        def generate_results():
            """为每个文件生成结果的生成器函数"""
            # 设置输出目录
            if custom_output_dir:
                output_dir = custom_output_dir
            else:
                output_dir = app.config['OUTPUT_FOLDER']
            
            os.makedirs(output_dir, exist_ok=True)
            
            for file in files:
                if file.filename == '' or not allowed_file(file.filename):
                    continue
                
                result_data = {}
                try:
                    # 生成唯一文件名
                    filename = safe_subtitle_filename(file.filename)
                    unique_id = str(uuid.uuid4())
                    video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")
                    
                    # 保存上传的文件
                    file.save(video_path)
                    
                    # 转录
                    start_process_time = datetime.now()
                    result = transcribe_video(video_path, model_name, language)
                    end_process_time = datetime.now()
                    processing_time = (end_process_time - start_process_time).total_seconds()
                    
                    # 生成输出文件名
                    base_name, ext = os.path.splitext(filename)
                    output_filename = f"{base_name}.{output_format}"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # 保存字幕文件
                    success = save_subtitles(result, output_path, output_format)
                    
                    if not success:
                        raise Exception("保存原始字幕文件失败")
                    
                    result_data = {
                        'filename': filename,
                        'output_file': output_filename,
                        'status': 'success',
                        'language': result.get('language', 'unknown'),
                        'segments_count': len(result.get('segments', [])),
                        'duration': result.get('segments', [{}])[-1].get('end', 0) if result.get('segments') else 0,
                        'processing_time': round(processing_time, 2)
                    }

                    # 如果需要，进行翻译
                    if translate and ollama_model and output_format in ['srt', 'vtt']:
                        try:
                            with open(output_path, 'r', encoding='utf-8') as f:
                                original_srt = f.read()
                            
                            translated_srt = translate_srt_content(original_srt, target_lang, ollama_model)

                            if translated_srt:
                                translated_output_filename = f"{base_name}.{target_lang}.{output_format}"
                                translated_output_path = os.path.join(output_dir, translated_output_filename)
                                with open(translated_output_path, 'w', encoding='utf-8') as f:
                                    f.write(translated_srt)
                                result_data['translated_output_file'] = translated_output_filename
                        except Exception as e:
                            print(f"文件 '{filename}' 的翻译过程失败: {e}")
                            result_data['translation_error'] = str(e)
                        
                except Exception as e:
                    result_data = {
                        'filename': file.filename,
                        'status': 'failed',
                        'error': str(e)
                    }
                finally:
                    # 清理临时文件
                    if 'video_path' in locals() and os.path.exists(video_path):
                        os.remove(video_path)
                
                # 使用 newline-delimited JSON 格式流式传输每个结果
                yield json.dumps(result_data) + '\n'

        return Response(stream_with_context(generate_results()), mimetype='application/x-ndjson')
        
    except Exception as e:
        # 捕捉初始设置中的错误
        return jsonify({'error': f'批量处理失败: {str(e)}'}), 500

@app.route('/api/batch-translate-files', methods=['POST'])
def batch_translate_files():
    """批量翻译字幕文件（已存在的和新上传的）"""
    try:
        target_lang = request.form.get('target_lang', 'zh')
        ollama_model = request.form.get('ollama_model')
        output_dir_str = request.form.get('output_dir', '')
        logger.info(f"批量字幕翻译请求，目标语言: {target_lang}，模型: {ollama_model}，输出目录: {output_dir_str}")
        if not ollama_model:
            logger.warning('未指定Ollama模型')
            return jsonify({'error': '未指定Ollama模型'}), 400
        output_dir = output_dir_str if output_dir_str else app.config['OUTPUT_FOLDER']
        os.makedirs(output_dir, exist_ok=True)
        existing_files = request.form.getlist('existing_files[]')
        uploaded_files = request.files.getlist('subtitle_files')
        def is_valid_srt(content):
            lines = content.strip().split('\n')
            return (lines and lines[0].strip().isdigit() and '-->' in (lines[1] if len(lines)>1 else ''))
        def is_valid_vtt(content):
            lines = content.strip().split('\n')
            return (lines and lines[0].strip().upper() == 'WEBVTT' and '-->' in (lines[2] if len(lines)>2 else ''))
        def generate_translation_results():
            for filename in existing_files:
                try:
                    file_path = os.path.join(output_dir, filename)
                    base_name, ext = os.path.splitext(filename)
                    if not os.path.exists(file_path) or ext.lower() not in ['.srt', '.vtt']:
                        continue
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if ext.lower() == '.srt' and not is_valid_srt(content):
                        logger.warning(f"文件{filename} SRT格式错误")
                        yield json.dumps({'filename': filename, 'status': 'failed', 'error': 'SRT字幕格式错误'}) + '\n'
                        continue
                    if ext.lower() == '.vtt' and not is_valid_vtt(content):
                        logger.warning(f"文件{filename} VTT格式错误")
                        yield json.dumps({'filename': filename, 'status': 'failed', 'error': 'VTT字幕格式错误'}) + '\n'
                        continue
                    logger.info(f"开始翻译文件: {filename}")
                    translated_content = translate_srt_content(content, target_lang, ollama_model)
                    if translated_content:
                        translated_filename = f"{base_name}.{target_lang}{ext}"
                        translated_path = os.path.join(output_dir, translated_filename)
                        with open(translated_path, 'w', encoding='utf-8') as f:
                            f.write(translated_content)
                        logger.info(f"翻译完成: {translated_filename}")
                        yield json.dumps({'filename': filename, 'status': 'success', 'translated_file': translated_filename}) + '\n'
                    else:
                        raise Exception("翻译返回空内容")
                except Exception as e:
                    logger.error(f"翻译文件{filename}时出错: {e}")
                    yield json.dumps({'filename': filename, 'status': 'failed', 'error': str(e)}) + '\n'
            for file in uploaded_files:
                filename = safe_subtitle_filename(file.filename)
                try:
                    base_name, ext = os.path.splitext(filename)
                    if ext.lower() not in ['.srt', '.vtt']:
                        continue
                    content = file.read().decode('utf-8')
                    if ext.lower() == '.srt' and not is_valid_srt(content):
                        logger.warning(f"上传文件{filename} SRT格式错误")
                        yield json.dumps({'filename': filename, 'status': 'failed', 'error': 'SRT字幕格式错误'}) + '\n'
                        continue
                    if ext.lower() == '.vtt' and not is_valid_vtt(content):
                        logger.warning(f"上传文件{filename} VTT格式错误")
                        yield json.dumps({'filename': filename, 'status': 'failed', 'error': 'VTT字幕格式错误'}) + '\n'
                        continue
                    logger.info(f"开始翻译上传文件: {filename}")
                    translated_content = translate_srt_content(content, target_lang, ollama_model)
                    if translated_content:
                        translated_filename = f"{base_name}.{target_lang}{ext}"
                        translated_path = os.path.join(output_dir, translated_filename)
                        with open(translated_path, 'w', encoding='utf-8') as f:
                            f.write(translated_content)
                        logger.info(f"翻译完成: {translated_filename}")
                        yield json.dumps({'filename': filename, 'status': 'success', 'translated_file': translated_filename}) + '\n'
                    else:
                        raise Exception("翻译返回空内容")
                except Exception as e:
                    logger.error(f"翻译上传文件{filename}时出错: {e}")
                    yield json.dumps({'filename': filename, 'status': 'failed', 'error': str(e)}) + '\n'
        return Response(stream_with_context(generate_translation_results()), mimetype='application/x-ndjson')
    except Exception as e:
        logger.error(f"批量翻译处理失败: {e}")
        return jsonify({'error': f'批量翻译处理失败: {str(e)}'}), 500

@app.route('/api/ollama-models', methods=['GET'])
def ollama_models():
    """获取可用的Ollama模型"""
    models = get_ollama_models()
    return jsonify({'models': models})

@app.route('/api/download/<filename>')
def download_file(filename):
    """下载生成的字幕文件"""
    try:
        output_dir = request.args.get('output_dir')
        if not output_dir:
            output_dir = app.config['OUTPUT_FOLDER']

        file_path = os.path.join(output_dir, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': '文件不存在'}), 404
    except Exception as e:
        return jsonify({'error': f'下载失败: {str(e)}'}), 500

@app.route('/api/files', methods=['GET'])
def list_files():
    """列出输出文件夹中的文件"""
    try:
        output_dir = request.args.get('output_dir')
        if not output_dir:
            output_dir = app.config['OUTPUT_FOLDER']

        if not os.path.exists(output_dir):
            return jsonify({'files': []})
        
        files = []
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                files.append({
                    'name': filename,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': f'获取文件列表失败: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 