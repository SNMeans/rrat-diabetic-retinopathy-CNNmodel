from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def root():
    """
    Welcome to the Diabetic Retinopathy CNN Model API.

    This API endpoint will return a simple message welcoming users to the API.
    """
    return {"result": "Welcome to the Diabetic Retinopathy CNN Model API!"}
