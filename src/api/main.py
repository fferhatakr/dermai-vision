from fastapi import FastAPI
import uvicorn
import src.api.models as ai_models
from src.api.database import create_tables
from src.api.routes import users
from src.api.routes import analyze
from src.api.routes import patients
from src.api.routes import report
from src.api.db_models import InferenceLog, DoctorFeedback

app = FastAPI(title="DermaScan AI - Full Debug Meta-Engine")
app.include_router(users.router)
app.include_router(analyze.router)
app.include_router(patients.router)
app.include_router(report.router)


@app.on_event("startup")
def startup():
    create_tables()
    ai_models.load_ai_models()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)