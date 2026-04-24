from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.api.database import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer,primary_key=True,index=True)
    email = Column(String,unique=True,index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    is_active = Column(Boolean,default=True)
    created_at = Column(DateTime,default=func.now())

    analyses = relationship("Analysis",back_populates="user")

class Patient(Base):
    __tablename__ = "patients"
    id = Column(Integer,primary_key=True,index=True)
    full_name = Column(String)
    age = Column(Integer)
    sex = Column(String)
    anatom_site = Column(String)
    created_at = Column(DateTime,default=func.now())

    analyses = relationship("Analysis",back_populates="patient")


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer,primary_key=True,index=True)
    patient_id = Column(Integer,ForeignKey("patients.id"))
    user_id = Column(Integer,ForeignKey("users.id"))
    diagnosis = Column(String)
    confidence = Column(Float)
    is_risky = Column(Boolean)
    heatmap_base64 = Column(Text,nullable=True)
    created_at = Column(DateTime,default=func.now())

    patient = relationship("Patient",back_populates="analyses")
    user = relationship("User",back_populates="analyses")


class InferenceLog(Base):

    __tablename__ = "inference_logs"
    # nullable= This column may be empty
    id = Column(Integer, primary_key=True, index=True)
    patient_age = Column(Integer, nullable=True)
    patient_sex = Column(String, nullable=True)
    anatomical_site = Column(String, nullable=True)


    cnn_mel_prob = Column(Float, nullable=False)
    xgb_mel_prob = Column(Float, nullable=False)
    final_decision = Column(String, nullable=False)

    image_path = Column(String, nullable=False)
    #server_default=func.now -->> If the column is empty, automatically enter the current time  
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DoctorFeedback(Base):
    __tablename__ = "doctor_feedbacks"

    id = Column(Integer, primary_key=True, index=True)
    inference_id = Column(Integer, ForeignKey("inference_logs.id"),nullable=False)

    is_correct = Column(Boolean, nullable=False)
    corrected_diagnosis = Column(String, nullable=True)
    doctor_notes = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

