#!/usr/bin/env python3
"""
تشغيل سريع لواجهة الدردشة الذكية
"""

import os
import sys
import subprocess

def check_requirements():
    """التحقق من المتطلبات الأساسية"""
    print("🔍 التحقق من المتطلبات...")
    
    # التحقق من مفتاح API
    if not os.getenv('HUGGINGFACE_API_KEY'):
        print("❌ مفتاح HUGGINGFACE_API_KEY غير موجود!")
        print("يرجى إضافة مفتاح Hugging Face API:")
        print("export HUGGINGFACE_API_KEY='your_api_key_here'")
        return False
    
    print("✅ مفتاح Hugging Face API موجود")
    return True

def install_dependencies():
    """تثبيت المتطلبات"""
    print("📦 تثبيت المتطلبات...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "flask", "flask-socketio", "python-socketio", "eventlet",
            "speechrecognition", "pyttsx3", "requests", "python-dotenv"
        ], check=True)
        print("✅ تم تثبيت المتطلبات بنجاح")
        return True
    except subprocess.CalledProcessError:
        print("❌ فشل في تثبيت المتطلبات")
        return False

def run_app():
    """تشغيل التطبيق"""
    print("🚀 بدء تشغيل واجهة الدردشة...")
    print("📱 الواجهة ستكون متاحة على: http://localhost:5000")
    print("⏹️ اضغط Ctrl+C للإيقاف")
    
    try:
        from app import socketio, app
        socketio.run(app, debug=False, host='0.0.0.0', port=5000)
    except ImportError:
        print("❌ خطأ في استيراد التطبيق. تأكد من وجود ملف app.py")
    except Exception as e:
        print(f"❌ خطأ في تشغيل التطبيق: {e}")

def main():
    """الدالة الرئيسية"""
    print("🤖 واجهة الدردشة الذكية - Hugging Face Models")
    print("=" * 50)
    
    # التحقق من المتطلبات
    if not check_requirements():
        return
    
    # تثبيت المتطلبات
    if not install_dependencies():
        return
    
    # تشغيل التطبيق
    run_app()

if __name__ == "__main__":
    main()
