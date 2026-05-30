from fastapi import FastAPI, UploadFile, File, HTTPException
# import uvicorn
from PIL import Image
import io
from src.inference.predict import predict_image

# creates the FastAPI application object, it does not run a server.
app = FastAPI(title="Waste Classification API")


@app.get("/")
def home():
    return {"message": "Waste Classification API Running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # validate image type
    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/jpg"
    ]:
        raise HTTPException(status_code=400, detail="Invalid image format")

    try:

        # read uploaded file
        contents = await file.read()

        # convert bytes to PIL image
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # run inference
        result = predict_image(image)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# optional, alternative: uvicorn src.api.app:app --reload
# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)