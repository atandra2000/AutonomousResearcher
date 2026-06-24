"""AST domain models for Phase 2 - Repository Understanding Agent."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    """Represents a node in the repository knowledge graph."""

    id: str = Field(..., description="Unique node identifier")
    type: str = Field(..., description="Node type: class, function, module, config, data, model, dataset, optimizer, loss, metric")
    name: str = Field(..., description="Name of the node")
    file_path: str | None = Field(None, description="File path where node is defined")
    line_number: int | None = Field(None, description="Line number in file")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    description: str | None = Field(None, description="Brief description")
    properties: dict[str, Any] = Field(default_factory=dict, description="Node properties")


class GraphEdge(BaseModel):
    """Represents an edge (relationship) between nodes in the knowledge graph."""

    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    relationship: str = Field(..., description="Relationship type: uses, inherits, calls, depends_on, configures, trains, evaluates, imports, creates")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Edge metadata")
    context: str | None = Field(None, description="Context line where relationship occurs")


class ClassInfo(BaseModel):
    """Represents a class in Python AST."""

    name: str = Field(..., description="Class name")
    line_number: int = Field(..., description="Line number where class is defined")
    inherits: list[str] = Field(default_factory=list, description="Base classes")
    decorators: list[str] = Field(default_factory=list, description="Class decorators")
    methods: list["MethodInfo"] = Field(default_factory=list, description="Class methods")
    docstring: str | None = Field(None, description="Class docstring")
    is_abstract: bool = Field(default=False, description="Whether class is abstract")
    is_dataclass: bool = Field(default=False, description="Whether class is a dataclass")
    parent_module: str | None = Field(None, description="Parent module path")
    complexity_score: float = Field(default=0.0, description="Cyclomatic complexity")


class FunctionInfo(BaseModel):
    """Represents a function/method in Python AST."""

    name: str = Field(..., description="Function name")
    line_number: int = Field(..., description="Line number where function is defined")
    arguments: list["ArgumentInfo"] = Field(default_factory=list, description="Function arguments")
    decorators: list[str] = Field(default_factory=list, description="Function decorators")
    return_type: str | None = Field(None, description="Return type annotation")
    docstring: str | None = Field(None, description="Function docstring")
    is_async: bool = Field(default=False, description="Whether function is async")
    is_generator: bool = Field(default=False, description="Whether function is a generator")
    complexity_score: float = Field(default=0.0, description="Cyclomatic complexity")
    is_property: bool = Field(default=False, description="Whether function is a property getter")
    is_static: bool = Field(default=False, description="Whether function is static")
    is_classmethod: bool = Field(default=False, description="Whether function is a classmethod")


class ArgumentInfo(BaseModel):
    """Represents a function argument."""

    name: str = Field(..., description="Argument name")
    annotation: str | None = Field(None, description="Type annotation")
    default: str | None = Field(None, description="Default value")
    is_kwonly: bool = Field(default=False, description="Whether argument is keyword-only")
    is_vararg: bool = Field(default=False, description="Whether argument is *args")
    is_kwargs: bool = Field(default=False, description="Whether argument is **kwargs")


class MethodInfo(BaseModel):
    """Represents a method within a class."""

    name: str = Field(..., description="Method name")
    line_number: int = Field(..., description="Line number where method is defined")
    arguments: list["ArgumentInfo"] = Field(default_factory=list, description="Method arguments")
    decorators: list[str] = Field(default_factory=list, description="Method decorators")
    return_type: str | None = Field(None, description="Return type annotation")
    docstring: str | None = Field(None, description="Method docstring")


class ImportInfo(BaseModel):
    """Represents an import statement."""

    module: str = Field(..., description="Imported module name")
    names: list[str] = Field(default_factory=list, description="Imported names (objects, functions, classes)")
    line_number: int = Field(..., description="Line number of import")
    is_relative: bool = Field(default=False, description="Whether import is relative")
    level: int = Field(default=0, description="Relative import level")
    alias: str | None = Field(None, description="Import alias (if using 'as')")


class DecoratorInfo(BaseModel):
    """Represents a decorator applied to a class or function."""

    name: str = Field(..., description="Decorator name")
    line_number: int = Field(..., description="Line number of decorator")
    arguments: list[str] = Field(default_factory=list, description="Decorator arguments")
    kwarguments: dict[str, str] = Field(default_factory=dict, description="Decorator keyword arguments")


class TypeHintInfo(BaseModel):
    """Represents a type hint in function signature or variable annotation."""

    annotation: str = Field(..., description="Type hint string")
    line_number: int = Field(..., description="Line number of annotation")
    context: str = Field(default="unknown", description="Context: parameter, return, variable, attribute")
    is_optional: bool = Field(default=False, description="Whether type is Optional")
    generic_args: list[str] = Field(default_factory=list, description="Generic type arguments")


class ComplexityMetrics(BaseModel):
    """Represents code complexity metrics for a file or module."""

    cyclomatic_complexity: float = Field(default=0.0, description="Cyclomatic complexity")
    maintainability_index: float = Field(default=0.0, description="Maintainability index (0-100)")
    halstead_metrics: dict[str, float] = Field(default_factory=dict, description="Halstead complexity metrics")
    line_count: int = Field(default=0, description="Total lines of code")
    comment_ratio: float = Field(default=0.0, description="Comment lines / total lines")
    max_nesting_depth: int = Field(default=0, description="Maximum nesting depth")
    number_of_classes: int = Field(default=0, description="Number of classes")
    number_of_functions: int = Field(default=0, description="Number of functions")


class ASTOutput(BaseModel):
    """Output from AST analysis of a Python file."""

    file_path: str = Field(..., description="Path to analyzed file")
    classes: list[ClassInfo] = Field(default_factory=list, description="Classes in file")
    functions: list[FunctionInfo] = Field(default_factory=list, description="Functions in file")
    imports: list[ImportInfo] = Field(default_factory=list, description="Import statements")
    decorators: list[DecoratorInfo] = Field(default_factory=list, description="Decorators used")
    type_hints: dict[str, TypeHintInfo] = Field(default_factory=dict, description="Type hints found")
    complexity_metrics: ComplexityMetrics = Field(default_factory=ComplexityMetrics, description="Complexity metrics")
    parse_timestamp: datetime = Field(default_factory=datetime.now, description="AST parsing timestamp")
    line_count: int = Field(default=0, description="Total lines in file")
    parse_errors: list[str] = Field(default_factory=list, description="Any parsing errors encountered")


class ModuleInfo(BaseModel):
    """Represents a Python module/package."""

    name: str = Field(..., description="Module name")
    path: str = Field(..., description="File path")
    imports: list[ImportInfo] = Field(default_factory=list, description="Imports from this module")
    exports: list[str] = Field(default_factory=list, description="Exported symbols")
    line_count: int = Field(default=0, description="Lines of code")
    has_tests: bool = Field(default=False, description="Whether module has associated tests")
