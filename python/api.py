from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# 1. Root Route
@app.get("/")
def home():
    return {"message": "Hello Charan! Your FastAPI is working 🚀"}

# 2. Simple GET
@app.get("/hello")
def say_hello():
    return {"msg": "Hello from GET API!"}

# 3. Path Parameter
@app.get("/user/{name}")
def get_user(name: str):
    return {"message": f"Hello {name}"}

# 4. Query Parameters
@app.get("/add")
def add(a: int, b: int):
    return {"sum": a + b}

# 5. Body JSON (POST)
class Item(BaseModel):
    name: str
    price: float

@app.post("/item")
def create_item(item: Item):
    return {"message": "Item received", "data": item}
