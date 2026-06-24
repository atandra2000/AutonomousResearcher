"""AST Analyzer Tool for Phase 2 - Repository Understanding Agent."""

import ast
from datetime import datetime

from pydantic import BaseModel, Field

from research_engineer.models.ast_models import (
    ArgumentInfo,
    ASTOutput,
    ClassInfo,
    ComplexityMetrics,
    DecoratorInfo,
    FunctionInfo,
    ImportInfo,
    MethodInfo,
    TypeHintInfo,
)
from research_engineer.tools.base import Tool, ToolError


class ASTInput(BaseModel):
    """Input for AST analyzer tool."""

    file_path: str = Field(..., description="Path to Python file")
    content: str = Field(..., description="File content to analyze")
    extract_decorators: bool = Field(default=True, description="Extract decorators")
    extract_type_hints: bool = Field(default=True, description="Extract type hints")
    extract_docstrings: bool = Field(default=True, description="Extract docstrings")
    include_metrics: bool = Field(default=True, description="Include complexity metrics")


class ASTAnalysisTool(Tool[ASTInput, ASTOutput]):
    """Parse Python files with AST, extract classes, functions, decorators."""

    def __init__(self):
        self._cached_trees: dict[str, ast.AST] = {}

    def _parse(self, content: str) -> ast.Module:
        """Parse Python code into AST."""
        try:
            return ast.parse(content)
        except SyntaxError as e:
            raise ToolError(f"Syntax error in file: {e}", None)

    def _extract_node_line(self, node: ast.AST) -> int:
        """Extract line number from AST node."""
        return getattr(node, 'lineno', 0)

    def _extract_node_end_line(self, node: ast.AST) -> int:
        """Extract end line number from AST node."""
        return getattr(node, 'end_lineno', 0)

    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Extract decorator name from decorator node."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                return decorator.func.attr
        return str(decorator)

    def _extract_decorators(self, node: ast.AST) -> list[str]:
        """Extract decorator names from node."""
        decorators = []
        for dec in getattr(node, 'decorator_list', []):
            decorators.append(self._get_decorator_name(dec))
        return decorators

    def _extract_arguments(self, args: ast.arguments) -> list[ArgumentInfo]:
        """Extract function arguments."""
        arguments = []

        # Regular arguments with defaults
        defaults = args.defaults
        for i, arg in enumerate(args.posonlyargs + args.args):
            default_idx = i - len(args.posonlyargs) - len(args.args) + len(defaults)
            default = None
            if default_idx >= 0 and default_idx < len(defaults):
                default = ast.unparse(defaults[default_idx]) if hasattr(ast, 'unparse') else str(defaults[default_idx])

            ann = arg.annotation
            annotation = None
            if ann:
                annotation = ast.unparse(ann) if hasattr(ast, 'unparse') else str(ann)

            arguments.append(ArgumentInfo(
                name=arg.arg,
                annotation=annotation,
                default=default,
                is_vararg=False,
                is_kwargs=False,
            ))

        # *args
        if args.vararg:
            arguments.append(ArgumentInfo(
                name=args.vararg.arg,
                annotation=args.vararg.annotation,
                default=None,
                is_vararg=True,
                is_kwargs=False,
            ))

        # Keyword-only arguments
        for arg in args.kwonlyargs:
            default = None
            idx = args.kwonlyargs.index(arg)
            if idx < len(args.kw_defaults) and args.kw_defaults[idx]:
                default = ast.unparse(args.kw_defaults[idx]) if hasattr(ast, 'unparse') else str(args.kw_defaults[idx])

            ann = arg.annotation
            annotation = None
            if ann:
                annotation = ast.unparse(ann) if hasattr(ast, 'unparse') else str(ann)

            arguments.append(ArgumentInfo(
                name=arg.arg,
                annotation=annotation,
                default=default,
                is_kwonly=True,
                is_vararg=False,
                is_kwargs=False,
            ))

        # **kwargs
        if args.kwarg:
            arguments.append(ArgumentInfo(
                name=args.kwarg.arg,
                annotation=args.kwarg.annotation,
                default=None,
                is_vararg=False,
                is_kwargs=True,
            ))

        return arguments

    def _extract_function_info(self, node: ast.FunctionDef | ast.AsyncFunctionDef, file_path: str) -> FunctionInfo:
        """Extract information from a function definition."""
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)

        docstring = None
        if node.docstring:
            docstring = ast.get_docstring(node)

        return FunctionInfo(
            name=node.name,
            line_number=self._extract_node_line(node),
            arguments=self._extract_arguments(node.args),
            decorators=self._extract_decorators(node),
            return_type=return_type,
            docstring=docstring,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_generator=self._is_generator(node),
            is_property=self._is_property(node),
            is_static=self._is_static_method(node),
            is_classmethod=self._is_classmethod(node),
        )

    def _is_generator(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Check if function is a generator (contains yield)."""
        for child in ast.walk(node):
            if isinstance(child, (ast.Yield, ast.YieldFrom)):
                return True
        return False

    def _is_property(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Check if function is a property getter."""
        return 'property' in self._extract_decorators(node)

    def _is_static_method(self, node: ast.FunctionDef) -> bool:
        """Check if function is a static method."""
        return 'staticmethod' in self._extract_decorators(node)

    def _is_classmethod(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Check if function is a class method."""
        return 'classmethod' in self._extract_decorators(node)

    def _extract_class_info(self, node: ast.ClassDef, file_path: str) -> ClassInfo:
        """Extract information from a class definition."""
        inherits = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                inherits.append(base.id)
            elif isinstance(base, ast.Attribute):
                inherits.append(base.attr)

        docstring = None
        if node.docstring:
            docstring = ast.get_docstring(node)

        # Check for dataclass decorator
        decorators = self._extract_decorators(node)
        is_dataclass = any('dataclass' in d.lower() for d in decorators)
        is_abstract = 'abstractmethod' in decorators

        # Extract methods
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(MethodInfo(
                    name=item.name,
                    line_number=self._extract_node_line(item),
                    arguments=self._extract_arguments(item.args),
                    decorators=self._extract_decorators(item),
                    return_type=ast.unparse(item.returns) if item.returns else None,
                    docstring=ast.get_docstring(item),
                ))

        return ClassInfo(
            name=node.name,
            line_number=self._extract_node_line(node),
            inherits=inherits,
            decorators=decorators,
            methods=methods,
            docstring=docstring,
            is_abstract=is_abstract,
            is_dataclass=is_dataclass,
        )

    def _extract_import_info(self, node: ast.Import | ast.ImportFrom) -> ImportInfo:
        """Extract information from import statement."""
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
            return ImportInfo(
                module=None,
                names=names,
                line_number=self._extract_node_line(node),
                is_relative=False,
                level=0,
            )
        else:
            module = node.module or ''
            names = [alias.name for alias in node.names]
            return ImportInfo(
                module=module,
                names=names,
                line_number=self._extract_node_line(node),
                is_relative=node.level > 0,
                level=node.level,
            )

    def _extract_type_hints(self, node: ast.AST, content: str) -> dict[str, TypeHintInfo]:
        """Extract type hints from AST node."""
        type_hints = {}

        for child in ast.walk(node):
            if isinstance(child, ast.AnnAssign):
                var_name = None
                if isinstance(child.target, ast.Name):
                    var_name = child.target.id

                if var_name and child.annotation:
                    annotation = ast.unparse(child.annotation) if hasattr(ast, 'unparse') else str(child.annotation)
                    type_hints[var_name] = TypeHintInfo(
                        annotation=annotation,
                        line_number=self._extract_node_line(child),
                        context="variable",
                    )

            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Function return type
                if child.returns:
                    annotation = ast.unparse(child.returns) if hasattr(ast, 'unparse') else str(child.returns)
                    type_hints[f"{child.name}_return"] = TypeHintInfo(
                        annotation=annotation,
                        line_number=self._extract_node_line(child),
                        context="return",
                    )

                # Function argument types
                for arg in child.args.posonlyargs + child.args.args:
                    if arg.annotation:
                        annotation = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else str(arg.annotation)
                        type_hints[f"{child.name}_{arg.arg}"] = TypeHintInfo(
                            annotation=annotation,
                            line_number=self._extract_node_line(arg),
                            context="parameter",
                        )

                # *args and **kwargs
                if child.args.vararg and child.args.vararg.annotation:
                    annotation = ast.unparse(child.args.vararg.annotation) if hasattr(ast, 'unparse') else str(child.args.vararg.annotation)
                    type_hints[f"{child.name}_{child.args.vararg.arg}"] = TypeHintInfo(
                        annotation=annotation,
                        line_number=self._extract_node_line(child.args.vararg),
                        context="parameter",
                    )

                if child.args.kwarg and child.args.kwarg.annotation:
                    annotation = ast.unparse(child.args.kwarg.annotation) if hasattr(ast, 'unparse') else str(child.args.kwarg.annotation)
                    type_hints[f"{child.name}_{child.args.kwarg.arg}"] = TypeHintInfo(
                        annotation=annotation,
                        line_number=self._extract_node_line(child.args.kwarg),
                        context="parameter",
                    )

        return type_hints

    def _calculate_complexity(self, node: ast.AST, content: str) -> ComplexityMetrics:
        """Calculate code complexity metrics."""
        line_count = len(content.split('\n'))

        # Count comments
        comment_count = 0
        for line in content.split('\n'):
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                comment_count += 1

        # Calculate cyclomatic complexity
        cyclomatic = 1  # Base complexity
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                cyclomatic += 1
            elif isinstance(child, ast.ExceptHandler):
                cyclomatic += 1
            elif isinstance(child, ast.BoolOp):
                # Each and/or increases complexity
                cyclomatic += len(child.values) - 1
            elif isinstance(child, ast.comprehension):
                # For each 'if' clause in comprehension
                cyclomatic += len(child.ifs)

        # Calculate Halstead metrics (simplified)
        # Count operators and operands
        operators = set()
        operands = set()
        for child in ast.walk(node):
            if isinstance(child, ast.operator):
                operators.add(type(child).__name__)
            elif isinstance(child, ast.Name):
                operands.add(child.id)

        n1 = len(operators)  # Unique operators
        n2 = len(operands)   # Unique operands
        N1 = 0  # Total operators
        N2 = 0  # Total operands
        for child in ast.walk(node):
            if isinstance(child, ast.operator):
                N1 += 1
            elif isinstance(child, ast.Name):
                N2 += 1

        # Halstead volume
        Halstead_volume = 0
        try:
            Halstead_volume = (N1 + N2) * ((n1 + n2).bit_length())
        except:
            Halstead_volume = 0

        # Calculate maintainability index (simplified)
        # Based on Halstead volume, cyclomatic complexity, and LOC
        halstead = Halstead_volume
        complexity = cyclomatic
        loc = line_count

        # Simple formula: higher is better
        maintainability = max(0, 171 - 5.2 * (halstead ** 0.167) - 0.23 * complexity - 16.2 * (loc / 100))

        return ComplexityMetrics(
            cyclomatic_complexity=float(cyclomatic),
            maintainability_index=float(maintainability),
            halstead_metrics={
                'n1': n1,
                'n2': n2,
                'N1': N1,
                'N2': N2,
                'volume': halstead,
            },
            line_count=line_count,
            comment_ratio=comment_count / line_count if line_count > 0 else 0,
            max_nesting_depth=self._calculate_nesting_depth(node),
            number_of_classes=sum(1 for x in ast.walk(node) if isinstance(x, ast.ClassDef)),
            number_of_functions=sum(1 for x in ast.walk(node) if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))),
        )

    def _calculate_nesting_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth."""
        max_depth = current_depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.Try, ast.With)):
                depth = self._calculate_nesting_depth(child, current_depth + 1)
                max_depth = max(max_depth, depth)
            else:
                depth = self._calculate_nesting_depth(child, current_depth)
                max_depth = max(max_depth, depth)
        return max_depth

    async def execute(self, input: ASTInput) -> ASTOutput:
        """Execute AST analysis on Python file."""
        try:
            tree = self._parse(input.content)

            # Extract classes
            classes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(self._extract_class_info(node, input.file_path))

            # Extract functions (at module level and in classes)
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append(self._extract_function_info(node, input.file_path))

            # Extract imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(self._extract_import_info(node))

            # Extract decorators
            decorators = []
            if input.extract_decorators:
                for node in ast.walk(tree):
                    if hasattr(node, 'decorator_list'):
                        for dec in node.decorator_list:
                            decorators.append(DecoratorInfo(
                                name=self._get_decorator_name(dec),
                                line_number=self._extract_node_line(node),
                            ))

            # Extract type hints
            type_hints = {}
            if input.extract_type_hints:
                type_hints = self._extract_type_hints(tree, input.content)

            # Calculate complexity
            complexity = ComplexityMetrics()
            if input.include_metrics:
                complexity = self._calculate_complexity(tree, input.content)

            return ASTOutput(
                file_path=input.file_path,
                classes=classes,
                functions=functions,
                imports=imports,
                decorators=decorators,
                type_hints=type_hints,
                complexity_metrics=complexity,
                line_count=len(input.content.split('\n')),
                parse_timestamp=datetime.now(),
            )

        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Failed to analyze AST for {input.file_path}: {e}", input, e)

    async def close(self):
        """Close resources."""
        pass
