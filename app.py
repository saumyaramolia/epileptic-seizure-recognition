from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# Add CORS support to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Custom exception class
class CustomException(Exception):
    def __init__(self, detail: str):
        self.detail = detail


# Custom exception handler
@app.exception_handler(CustomException)
async def custom_exception_handler(request, exc):
    return JSONResponse(status_code=400, content={"message": exc.detail})


# Route with error handling
@app.get("/example")
async def example():
    raise CustomException(detail="Custom error message")
