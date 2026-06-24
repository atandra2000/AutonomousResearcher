"""Parser tool for extracting sections from paper text."""

import re

from pydantic import BaseModel, Field

from research_engineer.models.paper import Paper
from research_engineer.tools.base import Tool, ToolError


class ParserInput(BaseModel):
    """Input for parser tool."""

    raw_content: str = Field(..., description="Raw paper text content")
    paper_metadata: Paper = Field(..., description="Paper metadata and context")


class ParserOutput(BaseModel):
    """Output from parser tool."""

    paper: Paper = Field(..., description="Paper with extracted content")
    sections: dict[str, str] = Field(default_factory=dict, description="Extracted sections")
    figures: list[dict] = Field(default_factory=list, description="Extracted figure markers")
    tables: list[dict] = Field(default_factory=list, description="Extracted table markers")
    equations: list[str] = Field(default_factory=list, description="Extracted equations")
    references: list[str] = Field(default_factory=list, description="Extracted references")


class PaperParserTool(Tool[ParserInput, ParserOutput]):
    """Parse unstructured paper content into structured sections."""

    SECTION_PATTERNS = {
        "abstract": [
            r"(?i)(abstract)\s*[:.\n]\s*(.*?)(?=(introduction|1\.|I\.|INTRODUCTION|$))",
            r"(?i)(abstract)\s*[:.\n]\s*(.*?)(?=(introduction|keywords|$))"
        ],
        "introduction": [
            r"(?i)(introduction)\s*[:.\n]\s*(.*?)(?=(related work|methods|experiments|conclusion|$))"
        ],
        "related_work": [
            r"(?i)(related work|background|previous work|literature review)\s*[:.\n]\s*(.*?)(?=(methods|experiments|results|$))"
        ],
        "methods": [
            r"(?i)(methods?|approach|methodology)\s*[:.\n]\s*(.*?)(?=(experiments|results|evaluation|$))"
        ],
        "model_architecture": [
            r"(?i)(model architecture|network architecture|architecture|model description)\s*[:.\n]\s*(.*?)(?=(training|training methodology|$))"
        ],
        "training_methodology": [
            r"(?i)(training methodology|training approach|training details|training settings)\s*[:.\n]\s*(.*?)(?=(evaluation|results|$))"
        ],
        "experiments": [
            r"(?i)(experiments|experimental setup|experimental results)\s*[:.\n]\s*(.*?)(?=(results|conclusion|$))"
        ],
        "results": [
            r"(?i)(results|evaluation results|experimental results)\s*[:.\n]\s*(.*?)(?=(conclusion|discussion|$))"
        ],
        "conclusion": [
            r"(?i)(conclusion)\s*[:.\n]\s*(.*)"
        ]
    }

    FIGURE_PATTERNS = [
        r"(?i)figure\s*#?\s*(\d+)",
        r"(?i)fig\.?\s*#?\s*(\d+)",
        r"(?i)fig\.?\s*(\d+)",
    ]

    TABLE_PATTERNS = [
        r"(?i)table\s*#?\s*(\d+)",
        r"(?i)tab\.?\s*#?\s*(\d+)",
        r"(?i)tab\.?\s*(\d+)",
    ]

    EQUATION_PATTERNS = [
        r"\$(.*?)\$",  # Inline equations $...$
        r"\\\[(.*?)\\\]",  # Display equations \\[...\\]
        r"\\\(.*?\\\)",  # Math mode \(...\)
    ]

    async def validate(self, input: ParserInput) -> bool:
        """Validate parser input."""
        return bool(
            input.raw_content and
            len(input.raw_content) > 100 and
            input.paper_metadata and
            input.paper_metadata.paper_id
        )

    async def _extract_section(self, content: str, pattern: str) -> tuple[str, str]:
        """Extract a section using regex pattern."""
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            section_name = match.group(1) if len(match.groups()) > 0 else "unknown"
            section_content = match.group(2) if len(match.groups()) > 1 else match.group(0)
            return section_name, section_content.strip()
        return None, None

    async def _find_figures(self, content: str) -> list[dict]:
        """Find figure references in content."""
        figures = []
        for pattern in self.FIGURE_PATTERNS:
            for match in re.finditer(pattern, content):
                figures.append({
                    "type": "figure",
                    "number": match.group(1),
                    "text": match.group(0)
                })
        return figures

    async def _find_tables(self, content: str) -> list[dict]:
        """Find table references in content."""
        tables = []
        for pattern in self.TABLE_PATTERNS:
            for match in re.finditer(pattern, content):
                tables.append({
                    "type": "table",
                    "number": match.group(1),
                    "text": match.group(0)
                })
        return tables

    async def _find_equations(self, content: str) -> list[str]:
        """Find equation markers in content."""
        equations = []
        for pattern in self.EQUATION_PATTERNS:
            for match in re.finditer(pattern, content):
                equation = match.group(1) if match.lastindex else match.group(0)
                equations.append(equation.strip())
        return equations

    async def _extract_abstract(self, content: str) -> str:
        """Extract abstract section."""
        # Try multiple patterns
        patterns = [
            r"(?i)(abstract)\s*[:.\n]\s*(.*?)(?=(introduction|1\.|I\.|INTRODUCTION|$))",
            r"(?i)(abstract)\s*[:.\n]\s*(.*?)(?=(introduction|keywords|$))",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(2).strip()
        return ""

    async def _extract_introduction(self, content: str) -> str:
        """Extract introduction section."""
        pattern = r"(?i)(introduction)\s*[:.\n]\s*(.*?)(?=(related work|methods|experiments|conclusion|$))"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(2).strip()
        return ""

    async def _extract_methods(self, content: str) -> str:
        """Extract methods section."""
        pattern = r"(?i)(methods?|approach)\s*[:.\n]\s*(.*?)(?=(experiments|results|conclusion|$))"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(2).strip()
        return ""

    async def _extract_results(self, content: str) -> str:
        """Extract results section."""
        pattern = r"(?i)(results|evaluation)\s*[:.\n]\s*(.*?)(?=(conclusion|$))"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(2).strip()
        return ""

    async def execute(self, input: ParserInput) -> ParserOutput:
        """Parse paper content into sections."""
        try:
            content = input.raw_content
            sections = {}

            # Extract key sections
            sections["abstract"] = await self._extract_abstract(content)
            sections["introduction"] = await self._extract_introduction(content)
            sections["methods"] = await self._extract_methods(content)
            sections["results"] = await self._extract_results(content)

            # Find figures, tables, equations
            figures = await self._find_figures(content)
            tables = await self._find_tables(content)
            equations = await self._find_equations(content)

            # Extract references section
            references = []
            ref_match = re.search(
                r"(?i)(references|bibliography)\s*[:.\n]\s*(.*)",
                content,
                re.DOTALL
            )
            if ref_match:
                # Split into individual references (simplified)
                ref_text = ref_match.group(2)
                references = [ref.strip() for ref in ref_text.split("\n") if ref.strip()]

            # Return parsed content
            return ParserOutput(
                paper=input.paper_metadata,
                sections=sections,
                figures=figures,
                tables=tables,
                equations=equations,
                references=references
            )

        except Exception as e:
            raise ToolError(f"Failed to parse paper: {e}", input, e)

    async def close(self):
        """Close resources."""
        pass
