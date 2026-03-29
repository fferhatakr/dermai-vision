from fastapi import APIRouter,Depends,HTTPException,status
from sqlalchemy.orm import Session
from src.api.database import get_db
from src.api import db_models
from src.api.auth import hash_password,verify_password,create_access_token
from pydantic import BaseModel

router = APIRouter(prefix="/users",tags=["users"])


class UserRegister(BaseModel):
    email: str
    password: str
    full_name: str



class UserLogin(BaseModel):
    email: str
    password:str


@router.post("/register")
def register(user: UserRegister,db: Session = Depends(get_db)):

    
    existing_user = db.query(db_models.User).filter(
        db_models.User.email == user.email
    ).first()


    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This email is already registered"
        )

    hashed = hash_password(user.password)


    new_user = db_models.User(
        email = user.email,
        hashed_password= hashed,
        full_name = user.full_name
    )

    db.add(new_user)
    db.commit()
    
    return {"message": "Registration successful", "email": user.email}

@router.post("/login")
def login(user: UserLogin,db:Session = Depends(get_db)):

    db_user = db.query(db_models.User).filter(
        db_models.User.email == user.email
    ).first()


    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid email or password"
        )

    if not verify_password(user.password,db_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid email or password"
        )
    token = create_access_token(data={"sub":db_user.email})

    return {"access_token":token,"token_type":"bearer"}
