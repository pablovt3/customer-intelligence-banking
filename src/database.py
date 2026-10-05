import os

from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine


PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / '.env')

DATABASE_URL = os.getenv('DATABASE_URL')

if DATABASE_URL is None:
    raise ValueError(
        'DATABASE_URL environment variable is not defined.'
    )

engine = create_engine(DATABASE_URL)