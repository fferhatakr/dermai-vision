from fastapi import APIRouter,Depends,HTTPException,status
from sqlalchemy.orm import Session
from src.api.database import get_db
from src.api import db_models
from src.api.auth import hash_password,verify_password,create_access_token
from pydantic import BaseModel

router = APIRouter(prefix="/users",tags=["users"])

# Format of data received for registration
class UserRegister(BaseModel):
    email: str
    password: str
    full_name: str


# The format of the incoming data for input.
class UserLogin(BaseModel):
    email: str
    password:str


@router.post("/register")
def register(user: UserRegister,db: Session = Depends(get_db)):

    # Check if this email address is already registered.
    existing_user = db.query(db_models.User).filter(
        db_models.User.email == user.email
    ).first()

    # Return error if registered
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This email is already registered"
        )
    # Hash the password
    hashed = hash_password(user.password)

    # Create new user
    new_user = db_models.User(
        email = user.email,
        hashed_password= hashed,
        full_name = user.full_name
    )
    # Add to database and save
    db.add(new_user)
    db.commit()
    
    return {"message": "Registration successful", "email": user.email}

@router.post("/login")
def login(user: UserLogin,db:Session = Depends(get_db)):
    # Check if this email exists in the database.
    db_user = db.query(db_models.User).filter(
        db_models.User.email == user.email
    ).first()

    # Return an error if email not found
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid email or password"
        )
    # Check if the password is correct,
    if not verify_password(user.password,db_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid email or password"
        )
    # Generate tokens — put your email address inside
    token = create_access_token(data={"sub":db_user.email})

    return {"access_token":token,"token_type":"bearer"}
