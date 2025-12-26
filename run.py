import uvicorn
import os
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️  هشدار: کلید GROQ_API_KEY در فایل .env یافت نشد.")
        print("لطفاً فایل .env را ویرایش کنید و کلید خود را وارد کنید.")
    
    print("🚀 در حال راه‌اندازی دستیار هوشمند...")
    print("🌐 آدرس دسترسی: http://localhost:8000")
    
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
