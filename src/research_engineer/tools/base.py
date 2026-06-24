"""Base tool interface and utilities."""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

InputType = TypeVar("InputType", bound=BaseModel)
OutputType = TypeVar("OutputType", bound=BaseModel)


class ToolError(Exception):
    """Custom exception for tool errors."""

    def __init__(self, message: str, input_data: Any = None, cause: Exception = None):
        super().__init__(message)
        self.input_data = input_data
        self.cause = cause


class Tool(ABC, Generic[InputType, OutputType]):
    """Base interface for all tools with typed input/output."""

    @abstractmethod
    async def execute(self, input: InputType) -> OutputType:
        """Execute the tool and return output."""
        pass

    async def validate(self, input: InputType) -> bool:
        """Validate input before execution. Override for custom validation."""
        try:
            # Validate using Pydantic
            input.model_validate(input)
            return True
        except ValidationError:
            return False
        except Exception:
            return False

    async def __call__(self, input: InputType) -> OutputType:
        """Allow tool to be called as a function."""
        if not await self.validate(input):
            raise ToolError(f"Invalid input for {self.__class__.__name__}", input)
        return await self.execute(input)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
