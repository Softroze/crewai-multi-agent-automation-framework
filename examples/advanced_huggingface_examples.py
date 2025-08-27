"""
أمثلة متقدمة لاستخدام نماذج Hugging Face في مشروع CrewAI
"""

import os
from crewai.models.model_manager import HuggingFaceModelManager
from crewai.llm import LLM

def main():
    # إنشاء مدير نماذج Hugging Face
    hf_manager = HuggingFaceModelManager()
    
    # أمثلة على أنواع المهام
    task_types = ["general", "chat", "code", "analysis", "research", "arabic"]
    
    for task_type in task_types:
        try:
            # اختيار النموذج المناسب
            config = hf_manager.select_model(task_type)
            llm = LLM(
                model=config["model"],
                api_key=config["api_key"],
                base_url=config["base_url"],
                temperature=config["temperature"]
            )
            print(f"✅ تم إنشاء نموذج {task_type} بنجاح: {config['model']}")
        except ValueError as e:
            print(f"❌ فشل إنشاء نموذج {task_type}: {e}")

if __name__ == "__main__":
    main()
