import uvicorn
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")  # Default to localhost if not set
    port = int(os.getenv("PORT", 8888))    # Default to 8000 if not set
    
    uvicorn.run("main:app", host=host, port=port, reload=True)
