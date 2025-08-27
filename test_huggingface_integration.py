#!/usr/bin/env python3
"""
اختبار تكامل نماذج Hugging Face مع CrewAI
"""

import os
import sys
from crewai.models.huggingface_models import (
    HF_MODELS, 
    get_hf_model_config, 
    validate_api_key,
    print_available_models,
    get_recommended_model
)
from crewai.models.model_manager import HuggingFaceModelManager

def test_api_key():
    """اختبار وجود مفتاح API"""
    print("🔑 اختبار مفتاح API...")
    
    if validate_api_key():
        print("✅ مفتاح HUGGINGFACE_API_KEY موجود")
        return True
    else:
        print("❌ مفتاح HUGGINGFACE_API_KEY غير موجود")
        print("يرجى إضافة مفتاح Hugging Face API في متغيرات البيئة")
        return False

def test_model_configurations():
    """اختبار تكوينات النماذج"""
    print("\n⚙️ اختبار تكوينات النماذج...")
    
    if not validate_api_key():
        print("❌ تخطي الاختبار - مفتاح API غير موجود")
        return False
    
    test_models = ["mistral-7b", "codellama-7b", "jais-13b"]
    success_count = 0
    
    for model_key in test_models:
        try:
            config = get_hf_model_config(model_key)
            print(f"✅ {model_key}: {config['model']}")
            success_count += 1
        except Exception as e:
            print(f"❌ {model_key}: {e}")
    
    print(f"📊 نجح {success_count}/{len(test_models)} نماذج")
    return success_count == len(test_models)

def test_model_manager():
    """اختبار مدير النماذج"""
    print("\n🎯 اختبار مدير النماذج...")
    
    if not validate_api_key():
        print("❌ تخطي الاختبار - مفتاح API غير موجود")
        return False
    
    try:
        hf_manager = HuggingFaceModelManager()
        
        # اختبار أنواع المهام المختلفة
        task_types = ["general", "code", "analysis", "arabic", "chat"]
        success_count = 0
        
        for task_type in task_types:
            try:
                config = hf_manager.select_model(task_type)
                print(f"✅ {task_type}: {config['model']}")
                success_count += 1
            except Exception as e:
                print(f"❌ {task_type}: {e}")
        
        print(f"📊 نجح {success_count}/{len(task_types)} أنواع مهام")
        return success_count == len(task_types)
        
    except Exception as e:
        print(f"❌ خطأ في إنشاء مدير النماذج: {e}")
        return False

def test_model_recommendations():
    """اختبار توصيات النماذج"""
    print("\n💡 اختبار توصيات النماذج...")
    
    test_cases = [
        ("chat", "llama-2-7b"),
        ("code", "codellama-7b"),
        ("arabic", "jais-13b"),
        ("عربي", "jais-13b"),
        ("default", "mistral-7b")
    ]
    
    success_count = 0
    
    for task_type, expected_model in test_cases:
        try:
            recommended = get_recommended_model(task_type)
            if recommended == expected_model:
                print(f"✅ {task_type} -> {recommended}")
                success_count += 1
            else:
                print(f"⚠️ {task_type} -> {recommended} (متوقع: {expected_model})")
        except Exception as e:
            print(f"❌ {task_type}: {e}")
    
    print(f"📊 نجح {success_count}/{len(test_cases)} توصيات")
    return success_count == len(test_cases)

def test_import_statements():
    """اختبار استيراد الوحدات"""
    print("\n📦 اختبار استيراد الوحدات...")
    
    try:
        from crewai.models import HF_MODELS, get_hf_model_config, HuggingFaceModelManager
        print("✅ استيراد الوحدات نجح")
        return True
    except ImportError as e:
        print(f"❌ خطأ في الاستيراد: {e}")
        return False

def display_available_models():
    """عرض النماذج المتاحة"""
    print("\n📋 النماذج المتاحة:")
    print("=" * 50)
    print_available_models()

def main():
    """تشغيل جميع الاختبارات"""
    print("🧪 بدء اختبار تكامل نماذج Hugging Face")
    print("=" * 60)
    
    tests = [
        ("اختبار استيراد الوحدات", test_import_statements),
        ("اختبار مفتاح API", test_api_key),
        ("اختبار توصيات النماذج", test_model_recommendations),
        ("اختبار تكوينات النماذج", test_model_configurations),
        ("اختبار مدير النماذج", test_model_manager),
    ]
    
    results = {}
    passed_tests = 0
    
    for test_name, test_function in tests:
        print(f"\n🔍 {test_name}...")
        print("-" * 40)
        
        try:
            result = test_function()
            results[test_name] = result
            if result:
                passed_tests += 1
        except Exception as e:
            print(f"❌ خطأ غير متوقع: {e}")
            results[test_name] = False
    
    # عرض النماذج المتاحة
    display_available_models()
    
    # النتائج النهائية
    print("\n" + "=" * 60)
    print("📊 ملخص نتائج الاختبار:")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ نجح" if result else "❌ فشل"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 النتيجة النهائية: {passed_tests}/{len(tests)} اختبارات نجحت")
    
    if passed_tests == len(tests):
        print("🎉 جميع الاختبارات نجحت! النظام جاهز للاستخدام.")
        return 0
    else:
        print("⚠️ بعض الاختبارات فشلت. يرجى مراجعة الأخطاء أعلاه.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
