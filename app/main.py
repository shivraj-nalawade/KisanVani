import uvicorn
from fastapi import FastAPI
from app.api.webhook import router as webhook_router

app = FastAPI(title="Agri Voice Assistant")

# Include the Twilio webhook routes
app.include_router(webhook_router)

@app.get("/")
def read_root():
    return {"status": "Server is running"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)