# from fastapi import FastAPI
# import pickle
# import numpy as np

# app = FastAPI()

# # Load model
# model = pickle.load(open("house_price_model.pkl", "rb"))

# @app.get("/predict")
# def predict(area: float, bedrooms: int):
#     prediction = model.predict(np.array([[area, bedrooms]]))
#     return {"Predicted Price": prediction[0]}

# from fastapi import FastAPI

# app = FastAPI()  # ✅ Make sure this line is present

# @app.get("/")
# def home():
#     return {"message": "FastAPI is running!"}

# @app.get("/predict")
# def predict(area: float, bedrooms: int):
#     # Dummy prediction logic (replace with your ML model)
#     predicted_price = (area * 3000) + (bedrooms * 50000)
#     return {"Predicted Price": predicted_price}


from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "FastAPI is running!"}

# Add this to run FastAPI when executing `python api.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
