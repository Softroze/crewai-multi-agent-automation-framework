#!/usr/bin/env python3
"""
أمثلة متخصصة لاستخدام أفضل نماذج توليد الكود البرمجي من Hugging Face
"""

import os
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
from crewai.models.model_manager import HuggingFaceModelManager

def create_python_developer():
    """إنشاء مطور Python متخصص"""
    print("🐍 إنشاء مطور Python متخصص...")
    
    try:
        hf_manager = HuggingFaceModelManager()
        config = hf_manager.select_model(task_type="code_python")
        
        python_llm = LLM(
            model=config["model"],
            api_key=config["api_key"],
            base_url=config["base_url"],
            temperature=config["temperature"]
        )
        
        python_dev = Agent(
            role='Python Developer Expert',
            goal='كتابة كود Python عالي الجودة ومحسن للأداء',
            backstory='أنت مطور Python خبير مع 10+ سنوات خبرة في تطوير التطبيقات والمكتبات.',
            verbose=True,
            llm=python_llm
        )
        
        python_task = Task(
            description='''
            اكتب كلاس Python لإدارة قاعدة بيانات SQLite مع الميزات التالية:
            1. إنشاء الاتصال وإغلاقه
            2. تنفيذ استعلامات SELECT, INSERT, UPDATE, DELETE
            3. معالجة الأخطاء
            4. استخدام context managers
            5. إضافة type hints
            6. كتابة docstrings مفصلة
            ''',
            agent=python_dev,
            expected_output='كلاس Python كامل مع التوثيق والأمثلة'
        )
        
        return Crew(
            agents=[python_dev],
            tasks=[python_task],
            verbose=2,
            process=Process.sequential
        )
        
    except Exception as e:
        print(f"❌ خطأ في إعداد مطور Python: {e}")
        return None

def create_web_developer():
    """إنشاء مطور ويب متخصص"""
    print("🌐 إنشاء مطور ويب متخصص...")
    
    try:
        hf_manager = HuggingFaceModelManager()
        config = hf_manager.select_model(task_type="code_web")
        
        web_llm = LLM(
            model=config["model"],
            api_key=config["api_key"],
            base_url=config["base_url"],
            temperature=config["temperature"]
        )
        
        web_dev = Agent(
            role='Full Stack Web Developer',
            goal='تطوير تطبيقات ويب حديثة وسريعة الاستجابة',
            backstory='أنت مطور ويب full-stack خبير في React, Node.js, وتقنيات الويب الحديثة.',
            verbose=True,
            llm=web_llm
        )
        
        web_task = Task(
            description='''
            اكتب مكونات React لتطبيق إدارة المهام (Todo App) مع:
            1. مكون رئيسي TodoApp
            2. مكون TodoList لعرض المهام
            3. مكون TodoItem لكل مهمة
            4. مكون AddTodo لإضافة مهام جديدة
            5. استخدام React Hooks (useState, useEffect)
            6. إضافة TypeScript types
            7. تصميم responsive مع CSS modules
            ''',
            agent=web_dev,
            expected_output='مكونات React كاملة مع TypeScript و CSS'
        )
        
        return Crew(
            agents=[web_dev],
            tasks=[web_task],
            verbose=2,
            process=Process.sequential
        )
        
    except Exception as e:
        print(f"❌ خطأ في إعداد مطور الويب: {e}")
        return None

def create_sql_developer():
    """إنشاء مطور قواعد البيانات متخصص"""
    print("🗄️ إنشاء مطور قواعد البيانات متخصص...")
    
    try:
        hf_manager = HuggingFaceModelManager()
        config = hf_manager.select_model(task_type="code_sql")
        
        sql_llm = LLM(
            model=config["model"],
            api_key=config["api_key"],
            base_url=config["base_url"],
            temperature=config["temperature"]
        )
        
        sql_dev = Agent(
            role='Database Developer Expert',
            goal='تصميم وكتابة استعلامات SQL محسنة وفعالة',
            backstory='أنت خبير قواعد البيانات مع خبرة عميقة في SQL وتحسين الأداء.',
            verbose=True,
            llm=sql_llm
        )
        
        sql_task = Task(
            description='''
            صمم قاعدة بيانات لنظام إدارة المكتبة واكتب الاستعلامات التالية:
            1. إنشاء الجداول (الكتب، المؤلفين، الأعضاء، الاستعارات)
            2. استعلام لعرض الكتب المتاحة مع معلومات المؤلف
            3. استعلام لعرض تاريخ استعارات عضو معين
            4. استعلام لإيجاد الكتب الأكثر استعارة
            5. استعلام لعرض الكتب المتأخرة في الإرجاع
            6. إضافة indexes لتحسين الأداء
            7. كتابة stored procedures للعمليات الشائعة
            ''',
            agent=sql_dev,
            expected_output='سكريبت SQL كامل مع الجداول والاستعلامات والفهارس'
        )
        
        return Crew(
            agents=[sql_dev],
            tasks=[sql_task],
            verbose=2,
            process=Process.sequential
        )
        
    except Exception as e:
        print(f"❌ خطأ في إعداد مطور SQL: {e}")
        return None

def create_algorithms_expert():
    """إنشاء خبير خوارزميات"""
    print("🧮 إنشاء خبير الخوارزميات...")
    
    try:
        hf_manager = HuggingFaceModelManager()
        config = hf_manager.select_model(task_type="code_algorithms")
        
        algo_llm = LLM(
            model=config["model"],
            api_key=config["api_key"],
            base_url=config["base_url"],
            temperature=config["temperature"]
        )
        
        algo_expert = Agent(
            role='Algorithms and Data Structures Expert',
            goal='حل المشاكل الخوارزمية المعقدة بكفاءة عالية',
            backstory='أنت خبير في الخوارزميات وهياكل البيانات مع خبرة في البرمجة التنافسية.',
            verbose=True,
            llm=algo_llm
        )
        
        algo_task = Task(
            description='''
            حل المشاكل الخوارزمية التالية بـ Python:
            1. تطبيق خوارزمية Dijkstra لإيجاد أقصر مسار
            2. تطبيق Binary Search Tree مع عمليات الإدراج والحذف والبحث
            3. حل مشكلة Longest Common Subsequence باستخدام Dynamic Programming
            4. تطبيق خوارزمية Quick Sort مع تحسينات
            5. حل مشكلة Two Sum وجميع المتغيرات
            
            لكل حل:
            - اكتب الكود مع التعليقات
            - حلل التعقيد الزمني والمكاني
            - أضف test cases
            - اقترح تحسينات ممكنة
            ''',
            agent=algo_expert,
            expected_output='حلول خوارزمية كاملة مع التحليل والاختبارات'
        )
        
        return Crew(
            agents=[algo_expert],
            tasks=[algo_task],
            verbose=2,
            process=Process.sequential
        )
        
    except Exception as e:
        print(f"❌ خطأ في إعداد خبير الخوارزميات: {e}")
        return None

def create_data_science_developer():
    """إنشاء مطور علوم البيانات"""
    print("📊 إنشاء مطور علوم البيانات...")
    
    try:
        hf_manager = HuggingFaceModelManager()
        config = hf_manager.select_model(task_type="code_data_science")
        
        ds_llm = LLM(
            model=config["model"],
            api_key=config["api_key"],
            base_url=config["base_url"],
            temperature=config["temperature"]
        )
        
        ds_dev = Agent(
            role='Data Science Developer',
            goal='تطوير حلول علوم البيانات والتعلم الآلي',
            backstory='أنت خبير في علوم البيانات والتعلم الآلي مع خبرة في Python وأدواته.',
            verbose=True,
            llm=ds_llm
        )
        
        ds_task = Task(
            description='''
            اكتب مشروع تحليل بيانات كامل لتوقع أسعار المنازل:
            1. تحميل وتنظيف البيانات باستخدام pandas
            2. تحليل استكشافي للبيانات (EDA) مع matplotlib/seaborn
            3. معالجة البيانات المفقودة والقيم الشاذة
            4. هندسة الميزات (Feature Engineering)
            5. تدريب نماذج مختلفة (Linear Regression, Random Forest, XGBoost)
            6. تقييم النماذج وضبط المعاملات
            7. حفظ النموذج الأفضل
            8. إنشاء pipeline للتنبؤ
            
            استخدم أفضل الممارسات في علوم البيانات.
            ''',
            agent=ds_dev,
            expected_output='مشروع علوم بيانات كامل مع الكود والتحليل'
        )
        
        return Crew(
            agents=[ds_dev],
            tasks=[ds_task],
            verbose=2,
            process=Process.sequential
        )
        
    except Exception as e:
        print(f"❌ خطأ في إعداد مطور علوم البيانات: {e}")
        return None

def display_coding_models():
    """عرض جميع نماذج البرمجة المتاحة"""
    print("\n🤖 نماذج البرمجة المتاحة:")
    print("=" * 60)
    
    try:
        hf_manager = HuggingFaceModelManager()
        coding_models = hf_manager.get_coding_models()
        
        for task_type, info in coding_models.items():
            print(f"\n📂 {task_type}:")
            print(f"   النموذج: {info['model_key']}")
            print(f"   الوصف: {info['description']}")
            print(f"   الحجم: {info['size']}")
            print(f"   نقاط القوة: {', '.join(info['strengths'])}")
            
    except Exception as e:
        print(f"❌ خطأ في عرض النماذج: {e}")

def main():
    """تشغيل جميع أمثلة نماذج البرمجة"""
    print("🚀 بدء تشغيل أمثلة نماذج البرمجة المتقدمة")
    print("=" * 70)
    
    # التحقق من وجود مفتاح API
    if not os.getenv("HUGGINGFACE_API_KEY"):
        print("❌ مفتاح HUGGINGFACE_API_KEY غير موجود!")
        print("يرجى إضافة مفتاح Hugging Face API في متغيرات البيئة.")
        return
    
    # عرض النماذج المتاحة
    display_coding_models()
    
    # قائمة المطورين المتخصصين
    developers = [
        ("مطور Python", create_python_developer),
        ("مطور الويب", create_web_developer),
        ("مطور قواعد البيانات", create_sql_developer),
        ("خبير الخوارزميات", create_algorithms_expert),
        ("مطور علوم البيانات", create_data_science_developer)
    ]
    
    results = {}
    
    for dev_name, dev_function in developers:
        print(f"\n🎯 تشغيل {dev_name}...")
        print("-" * 50)
        
        crew = dev_function()
        if crew:
            try:
                result = crew.kickoff()
                results[dev_name] = result
                print(f"✅ {dev_name} اكتمل بنجاح!")
            except Exception as e:
                print(f"❌ خطأ في تشغيل {dev_name}: {e}")
                results[dev_name] = f"خطأ: {e}"
        else:
            results[dev_name] = "فشل في الإعداد"
    
    # عرض النتائج النهائية
    print("\n" + "=" * 70)
    print("📋 ملخص نتائج نماذج البرمجة:")
    print("=" * 70)
    
    for dev_name, result in results.items():
        print(f"\n🔸 {dev_name}:")
        if isinstance(result, str) and ("خطأ" in result or "فشل" in result):
            print(f"   ❌ {result}")
        else:
            print(f"   ✅ تم بنجاح")
            # عرض جزء من النتيجة
            result_preview = str(result)[:300] + "..." if len(str(result)) > 300 else str(result)
            print(f"   📄 معاينة: {result_preview}")

if __name__ == "__main__":
    main()
