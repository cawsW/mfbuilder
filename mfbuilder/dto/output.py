from pydantic import BaseModel

class OutputsConfig(BaseModel):
    write_input: bool = True
    run: bool = True
