from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

class ClassificationResult(BaseModel):
    id: int = Field(..., description="Original ID of the data point")
    label: int = Field(..., ge=0, le=1, description="Predicted label (0 or 1)")

# 创建 parser
pyd_parser: PydanticOutputParser = PydanticOutputParser(pydantic_object=ClassificationResult)

# 对应的格式指令
pyd_format: str = pyd_parser.get_format_instructions()




if __name__ == "__main__":
    pass