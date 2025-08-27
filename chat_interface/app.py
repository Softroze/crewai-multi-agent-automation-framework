#!/usr/bin/env python3
"""
واجهة دردشة تفاعلية مع نماذج Hugging Face
تدعم النص والصوت مع إمكانية التبديل بين النماذج
"""

import os
import json
import asyncio
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
import speech_recognition as sr
import pyttsx3
import threading
import uuid

# استيراد نماذج CrewAI
from crewai.llm import LLM
from crewai.models.model_manager import HuggingFaceModelManager

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# إعداد محرك النطق
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # سرعة النطق
tts_engine.setProperty('volume', 0.9)  # مستوى الصوت

# إعداد التعرف على الصوت
recognizer = sr.Recognizer()
microphone = sr.Microphone()

class ChatBot:
    def __init__(self):
        self.hf_manager = HuggingFaceModelManager()
        self.current_model = None
        self.conversation_history = []
        self.is_listening = False
        
    def initialize_model(self, model_type="general"):
        """تهيئة النموذج المحدد"""
        try:
            config = self.hf_manager.select_model(task_type=model_type)
            self.current_model = LLM(
                model=config["model"],
                api_key=config["api_key"],
                base_url=config["base_url"],
                temperature=config["temperature"]
            )
            return True, f"تم تحميل نموذج {model_type} بنجاح"
        except Exception as e:
            return False, f"خطأ في تحميل النموذج: {str(e)}"
    
    def get_response(self, message, context=None):
        """الحصول على رد من النموذج"""
        if not self.current_model:
            return "يرجى تحديد نموذج أولاً"
        
        try:
            # إضافة السياق إذا كان متوفراً
            if context:
                full_message = f"السياق: {context}\n\nالسؤال: {message}"
            else:
                full_message = message
            
            # إضافة تاريخ المحادثة للسياق
            if self.conversation_history:
                recent_history = self.conversation_history[-5:]  # آخر 5 رسائل
                history_text = "\n".join([f"المستخدم: {h['user']}\nالمساعد: {h['bot']}" 
                                        for h in recent_history])
                full_message = f"تاريخ المحادثة:\n{history_text}\n\nالرسالة الحالية: {message}"
            
            response = self.current_model.call(full_message)
            
            # حفظ في تاريخ المحادثة
            self.conversation_history.append({
                'user': message,
                'bot': response,
                'timestamp': datetime.now().isoformat()
            })
            
            return response
        except Exception as e:
            return f"خطأ في الحصول على الرد: {str(e)}"
    
    def get_available_models(self):
        """الحصول على النماذج المتاحة"""
        return {
            "general": "النموذج العام",
            "chat": "نموذج المحادثة",
            "code": "نموذج البرمجة",
            "code_python": "نموذج Python",
            "code_web": "نموذج تطوير الويب",
            "code_sql": "نموذج قواعد البيانات",
            "analysis": "نموذج التحليل",
            "arabic": "النموذج العربي"
        }

# إنشاء مثيل الشات بوت
chatbot = ChatBot()

@app.route('/')
def index():
    """الصفحة الرئيسية"""
    return render_template('index.html')

@app.route('/api/models')
def get_models():
    """الحصول على قائمة النماذج المتاحة"""
    return jsonify(chatbot.get_available_models())

@app.route('/api/initialize', methods=['POST'])
def initialize_model():
    """تهيئة النموذج المحدد"""
    data = request.get_json()
    model_type = data.get('model_type', 'general')
    
    success, message = chatbot.initialize_model(model_type)
    return jsonify({
        'success': success,
        'message': message,
        'model_type': model_type
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """معالجة رسائل الدردشة"""
    data = request.get_json()
    message = data.get('message', '')
    context = data.get('context', '')
    
    if not message.strip():
        return jsonify({'error': 'الرسالة فارغة'})
    
    response = chatbot.get_response(message, context)
    
    return jsonify({
        'response': response,
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('connect')
def handle_connect():
    """عند الاتصال بالسوكت"""
    print('عميل متصل')
    emit('status', {'message': 'متصل بالخادم'})

@socketio.on('disconnect')
def handle_disconnect():
    """عند قطع الاتصال"""
    print('عميل منقطع')

@socketio.on('start_listening')
def handle_start_listening():
    """بدء الاستماع للصوت"""
    def listen_for_speech():
        try:
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source)
            
            emit('listening_status', {'status': 'listening', 'message': 'جاري الاستماع...'})
            
            with microphone as source:
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
            
            emit('listening_status', {'status': 'processing', 'message': 'جاري معالجة الصوت...'})
            
            # التعرف على الكلام (عربي وإنجليزي)
            try:
                text = recognizer.recognize_google(audio, language='ar-SA')
            except:
                try:
                    text = recognizer.recognize_google(audio, language='en-US')
                except:
                    text = None
            
            if text:
                emit('speech_recognized', {'text': text})
                emit('listening_status', {'status': 'success', 'message': f'تم التعرف على: {text}'})
            else:
                emit('listening_status', {'status': 'error', 'message': 'لم يتم التعرف على الكلام'})
                
        except sr.WaitTimeoutError:
            emit('listening_status', {'status': 'timeout', 'message': 'انتهت مهلة الاستماع'})
        except Exception as e:
            emit('listening_status', {'status': 'error', 'message': f'خطأ: {str(e)}'})
    
    # تشغيل الاستماع في خيط منفصل
    thread = threading.Thread(target=listen_for_speech)
    thread.daemon = True
    thread.start()

@socketio.on('speak_text')
def handle_speak_text(data):
    """تحويل النص إلى كلام"""
    text = data.get('text', '')
    
    if not text.strip():
        emit('speech_status', {'status': 'error', 'message': 'النص فارغ'})
        return
    
    def speak():
        try:
            emit('speech_status', {'status': 'speaking', 'message': 'جاري النطق...'})
            tts_engine.say(text)
            tts_engine.runAndWait()
            emit('speech_status', {'status': 'completed', 'message': 'تم النطق بنجاح'})
        except Exception as e:
            emit('speech_status', {'status': 'error', 'message': f'خطأ في النطق: {str(e)}'})
    
    # تشغيل النطق في خيط منفصل
    thread = threading.Thread(target=speak)
    thread.daemon = True
    thread.start()

if __name__ == '__main__':
    # التحقق من وجود مفتاح API
    if not os.getenv('HUGGINGFACE_API_KEY'):
        print("❌ تحذير: HUGGINGFACE_API_KEY غير موجود!")
        print("يرجى إضافة مفتاح Hugging Face API في متغيرات البيئة")
    
    print("🚀 بدء تشغيل واجهة الدردشة...")
    print("📱 الواجهة متاحة على: http://localhost:5000")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
