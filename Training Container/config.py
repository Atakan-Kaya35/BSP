import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    ENV = os.getenv("ENV", "production")
    IS_LOCAL = ENV == "local"
    
    TMP_DIR = Path(os.getenv("TMP_DIR", "./tmp")).resolve()
    S3_BUCKET = os.getenv("S3_BUCKET", "bspuserartifacts")
