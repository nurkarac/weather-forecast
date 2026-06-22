import google.generativeai as genai

# Buraya kendi uzun API anahtarını yapıştır
genai.configure(api_key="AQ.Ab8RN6KDBwclO5hAQxtBJjz9_n5m84iuwzrqtMCcvJxSCM3uBQ")

print("Kullanılabilir Modeller Aranıyor...\n")

# Hesabına tanımlı tüm modelleri listele
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)