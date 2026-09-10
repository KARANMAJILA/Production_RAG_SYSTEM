import os
from dotenv import load_dotenv


load_dotenv()

class Settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_CLUSTER_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    GOOGLE_GEMINI_KEY = os.getenv("GOOGLE_GEMINI_KEY")
    QDRANT_COLLECTION = "enterprise_rag"
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    PORTKEY_CONFIG = os.getenv("PORTKEY_CONFIG")
    QDRANT_URL = f"{QDRANT_CLUSTER_ENDPOINT}?api_key={QDRANT_API_KEY}"


settings = Settings()