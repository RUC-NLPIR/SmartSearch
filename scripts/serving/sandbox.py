from fastapi import FastAPI
import contextlib
import io
from typing import Dict
from pydantic import BaseModel
from argparse import ArgumentParser

app = FastAPI()

class CodeRequest(BaseModel):
    env: str
    call: str
    timeout: int = 5

@app.post("/execute")
async def execute_code(request: CodeRequest) -> Dict:
    output = io.StringIO()
    result = None
    error = None

    print("-"*30)
    print(request.env)
    print(request.call)
    print("-"*30)

    try:
        with contextlib.redirect_stdout(output):
            exec_env = {}
            exec(compile(request.env, '<env>', 'exec'), exec_env)
            exec(compile(f"response = {request.call}", '<call>', 'exec'), exec_env)
            result = exec_env.get('response')
    except Exception as e:
        error = str(e)

    return {
        "output": output.getvalue(),
        "result": result,
        "error": error
    }

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--port", type=int, default=80)
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=args.port)
