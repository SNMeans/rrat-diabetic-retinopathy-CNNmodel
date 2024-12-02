from fastapi import FastAPI

import random

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"result": "Hello World"}


@app.post("/analyze")
def analyze_image():   
    random_int = random.randint(0, 4)  # Generate a random integer between 0 and 4 as a placeholder
    return {"result": random_int}
