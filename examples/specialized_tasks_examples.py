#!/usr/bin/env python3
"""
أمثلة متخصصة لاستخدام نماذج Hugging Face في مهام محددة
"""

import os
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
from crewai.models.model_manager import HuggingFaceModelManager

def create_coding_crew():
    """إنشاء فريق متخصص في البرمجة"""
    print("🔧 إنشاء فريق البرمجة...")
    
    try:
        hf_manager = HuggingFaceModelManager()
        config = hf_manager.select_model(task_type="code")
        
        coding_llm = LLM(
            model=config["model"],
            api_key=config["api_key"],
            base_url=config["base_url"],
            temperature=config["temperature"]
        )
        
        # مطور Python
        python_developer = Agent(
            role='Python Developer',
            goal='كتابة كود Python عالي الجودة وحل المشاكل البرمجية',
            backstory='أنت مطور Python خبير مع سنوات من الخبرة في تطوير التطبيقات.',
            verbose=True,
            llm=coding_llm
        )
        
        # مهمة البرمجة
        coding_task = Task(
            description='اكتب دالة Python لحساب أرقام فيبوناتشي مع التحسين والتوثيق',
            agent=python_developer,
            expected_output='كود Python محسن مع التوثيق والأمثلة'
        )
        
        return Crew(
            agents=[python_developer],
            tasks=[coding_task],
            verbose=2,
            process=Process.sequential
        )
        
    except ValueError as e:
        print(f"❌ خطأ في إعداد فريق البرمجة: {e}")
        return None

def create_analysis_crew():
    """إنشاء فريق متخصص في التحليل"""
    print("📊 إنشاء فريق التحليل...")
    
    try:
        hf_manager = HuggingFaceModelManager()
        config = hf_manager.select_model(task_type="analysis")
        
        analysis_llm = LLM(
            model=config["model"],
            api_key=config["api_key"],
            base_url=config["base_url"],
            temperature=config["temperature"]
        )
        
        # محلل البيانات
        data_analyst = Agent(
            role='Data Analyst',
            goal='تحليل البيانات واستخراج الرؤى المفيدة',
            backstory='أنت محلل بيانات خبير متخصص في استخراج الأنماط والاتجاهات.',
            verbose=True,
            llm=analysis_llm
        )
        
        # مهمة التحليل
        analysis_task = Task(
            description='حلل اتجاهات الذكاء الاصطناعي في 2024 واكتب تقرير مفصل',
            agent=data_analyst,
            expected_output='تقرير تحليلي شامل مع الرؤى والتوصيات'
        )
        
        return Crew(
            agents=[data_analyst],
            tasks=[analysis_task],
            verbose=2,
            process=Process.sequential
        )
        
    except ValueError as e:
        print(f"❌ خطأ في إعداد فريق التحليل: {e}")
        return None

def create_arabic_crew():
    """إنشاء فريق متخصص في المحتوى العربي"""
    print("🇸🇦 إنشاء فريق المحتوى العربي...")
    
    try:
        hf_manager = HuggingFaceModelManager()
        config = hf_manager.select_model(task_type="arabic")
        
        arabic_llm = LLM(
            model=config["model"],
            api_key=config["api_key"],
            base_url=config["base_url"],
            temperature=config["temperature"]
        )
        
        # كاتب المحتوى العربي
        arabic_writer = Agent(
            role='كاتب المحتوى العربي',
            goal='إنتاج محتوى عربي عالي الجودة ومناسب ثقافياً',
            backstory='أنت كاتب محتوى عربي محترف مع فهم عميق للثقافة العربية واللغة.',
            verbose=True,
            llm=arabic_llm
        )
        
        # مهمة الكتابة العربية
        arabic_task = Task(
            description='اكتب مقال باللغة العربية عن أهمية الذكاء الاصطناعي في التعليم',
            agent=arabic_writer,
            expected_output='مقال عربي شامل ومفيد حول الذكاء الاصطناعي في التعليم'
        )
        
        return Crew(
            agents=[arabic_writer],
            tasks=[arabic_task],
            verbose=2,
            process=Process.sequential
        )
        
    except ValueError as e:
        print(f"❌ خطأ في إعداد فريق المحتوى العربي: {e}")
        return None

def create_chat_crew():
    """إنشاء فريق متخصص في المحادثة"""
    print("💬 إنشاء فريق المحادثة...")
    
    try:
        hf_manager = HuggingFaceModelManager()
        config = hf_manager.select_model(task_type="chat")
        
        chat_llm = LLM(
            model=config["model"],
            api_key=config["api_key"],
            base_url=config["base_url"],
            temperature=config["temperature"]
        )
        
        # مساعد المحادثة
        chat_assistant = Agent(
            role='Chat Assistant',
            goal='تقديم محادثات مفيدة وودودة للمستخدمين',
            backstory='أنت مساعد ذكي ودود يحب مساعدة الناس والإجابة على أسئلتهم.',
            verbose=True,
            llm=chat_llm
        )
        
        # مهمة المحادثة
        chat_task = Task(
            description='أجب على سؤال المستخدم: "ما هي أفضل الممارسات لتعلم البرمجة؟"',
            agent=chat_assistant,
            expected_output='إجابة شاملة ومفيدة مع نصائح عملية'
        )
        
        return Crew(
            agents=[chat_assistant],
            tasks=[chat_task],
            verbose=2,
            process=Process.sequential
        )
        
    except ValueError as e:
        print(f"❌ خطأ في إعداد فريق المحادثة: {e}")
        return None

def main():
    """تشغيل جميع الأمثلة المتخصصة"""
    print("🚀 بدء تشغيل الأمثلة المتخصصة لنماذج Hugging Face")
    print("=" * 60)
    
    # التحقق من وجود مفتاح API
    if not os.getenv("HUGGINGFACE_API_KEY"):
        print("❌ مفتاح HUGGINGFACE_API_KEY غير موجود!")
        print("يرجى إضافة مفتاح Hugging Face API في متغيرات البيئة.")
        return
    
    # قائمة الفرق المتخصصة
    crews = [
        ("فريق البرمجة", create_coding_crew),
        ("فريق التحليل", create_analysis_crew),
        ("فريق المحتوى العربي", create_arabic_crew),
        ("فريق المحادثة", create_chat_crew)
    ]
    
    results = {}
    
    for crew_name, crew_function in crews:
        print(f"\n🎯 تشغيل {crew_name}...")
        print("-" * 40)
        
        crew = crew_function()
        if crew:
            try:
                result = crew.kickoff()
                results[crew_name] = result
                print(f"✅ {crew_name} اكتمل بنجاح!")
            except Exception as e:
                print(f"❌ خطأ في تشغيل {crew_name}: {e}")
                results[crew_name] = f"خطأ: {e}"
        else:
            results[crew_name] = "فشل في الإعداد"
    
    # عرض النتائج النهائية
    print("\n" + "=" * 60)
    print("📋 ملخص النتائج:")
    print("=" * 60)
    
    for crew_name, result in results.items():
        print(f"\n🔸 {crew_name}:")
        if isinstance(result, str) and ("خطأ" in result or "فشل" in result):
            print(f"   ❌ {result}")
        else:
            print(f"   ✅ تم بنجاح")
            # عرض جزء من النتيجة
            result_preview = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
            print(f"   📄 معاينة: {result_preview}")

if __name__ == "__main__":
    main()
