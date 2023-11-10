from pydantic import BaseModel


class Text(BaseModel):
    text: str

    class Config:
        arbitrary_types_allowed = True