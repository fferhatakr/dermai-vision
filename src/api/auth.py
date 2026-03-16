from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt
import os
from dotenv import load_dotenv
from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends, HTTPException, status

load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
# Bcrypt — industry standard for password hashing
pwd_context = CryptContext(schemes=["bcrypt"],deprecated="auto")
# We specify where the token came from.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/users/login")

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


def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = decode_token(token)
        email = payload.get("sub")
        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        return email
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    