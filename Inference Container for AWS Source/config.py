# config.py
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Config:
    ENV = os.getenv("ENV", "local")
    IS_LOCAL = ENV == "local"
    
    TMP_DIR = Path(os.getenv("TMP_DIR", "/tmp/")).resolve()
    S3_BUCKET = os.getenv("S3_BUCKET", "bspuserartifacts")