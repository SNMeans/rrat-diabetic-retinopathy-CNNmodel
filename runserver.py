import uvicorn
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

if __name__ == "__main__":
    # Default to localhost if not set
    host = os.getenv("HOST", "127.0.0.1")

    # Default to 8888 if not set
    port = int(os.getenv("PORT", 8888))

    uvicorn.run("main:app", host=host, port=port, reload=True)
