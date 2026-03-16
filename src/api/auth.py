from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt
import os
from dotenv import load_dotenv

load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
# Bcrypt — industry standard for password hashing
pwd_context = CryptContext(schemes=["bcrypt"],deprecated="auto")

def hash_password(password: str) -> str:
    # Get the plain password, return its hashed version
    return pwd_context.hash(password)

def verify_password(plain_password: str , hashed_password:str) -> bool:
    # Compare plain password with hashed password

    return pwd_context.verify(plain_password , hashed_password)


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode , SECRET_KEY , algorithm=ALGORITHM)


def decode_token(token: str ) -> dict:
    # Raises JWTError if token is invalid or expired
    return jwt.decode(token,SECRET_KEY,algorithms=[ALGORITHM])