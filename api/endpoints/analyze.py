from fastapi import APIRouter
import random

router = APIRouter()


@router.post("/")
def analyze_image():

    random_int = random.randint(
        0, 4
    )  # Generate a random integer between 0 and 4 as a placeholder
    return {"result": random_int}
