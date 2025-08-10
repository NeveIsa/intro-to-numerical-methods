
import numpy as np
from fastapi import FastAPI

app = FastAPI()

@app.get("/bisection/{x}")
def read_item(x: float):
    return {'x':x, 'value': np.sin(x)}





