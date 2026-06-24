"""LLM-powered analysis tool using LlamaIndex."""

from typing import Any

from llama_index.core import (
    Document,
    ServiceContext,
)

from .base import Tool, ToolError


class LLMAnalysisInput:
    """Input for LLM analysis tool."""

    code_snippet: str
    analysis_type: str
    context: str | None = None

    def __init__(
        self,
        code_snippet: str,
        analysis_type: str = "general",
        context: str | None = None,
    ):
        self.code_snippet = code_snippet
        self.analysis_type = analysis_type
        self.context = context


class LLMAnalysisOutput:
    """Output from LLM analysis tool."""

    analysis: str
    confidence: float
    entities: list[dict]
    recommendations: list[str]

    def __init__(
        self,
        analysis: str,
        confidence: float = 0.5,
        entities: list[dict] | None = None,
        recommendations: list[str] | None = None,
    ):
        self.analysis = analysis
        self.confidence = confidence
        self.entities = entities or []
        self.recommendations = recommendations or []


class LLMAnalysisTool(Tool[LLMAnalysisInput, LLMAnalysisOutput]):
    """LLM-powered analysis tool for code understanding."""

    def __init__(self, model_name: str = "llama3"):
        self._model_name = model_name
        self._service_context: Any = None

    async def execute(self, input: LLMAnalysisInput) -> LLMAnalysisOutput:
        """Analyze code using LLM."""
        try:
            import os

            from llama_index.llms.ollama import Ollama

            # Initialize LLM
            llm = Ollama(
                model=self._model_name,
                base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            )

            # Create documents from code
            documents = [Document(text=input.code_snippet)]

            # Create query engine
            from llama_index.core import VectorStoreIndex

            index = VectorStoreIndex.from_documents(
                documents,
                service_context=ServiceContext.from_defaults(llm=llm),
            )
            query_engine = index.as_query_engine()

            # Build prompt based on analysis type
            prompt = self._build_prompt(input)

            # Query LLM
            response = await query_engine.aquery(prompt)

            # Parse response
            return LLMAnalysisOutput(
                analysis=str(response),
                confidence=0.8,
                entities=[],
                recommendations=["Review analysis results", "Consider refactoring"],
            )

        except ImportError as e:
            raise ToolError(
                "LLM tools not installed. Run: pip install llama-index-llms-ollama",
                input,
                e,
            )
        except Exception as e:
            raise ToolError(f"LLM analysis failed: {e}", input, e)

    async def validate(self, input: LLMAnalysisInput) -> bool:
        """Validate input."""
        return bool(input.code_snippet) and len(input.code_snippet) > 0

    def _build_prompt(self, input: LLMAnalysisInput) -> str:
        """Build analysis prompt based on input."""
        prompts = {
            "general": f"""Analyze the following code and provide:
1. Purpose and functionality
2. Key components and their interactions
3. Potential improvements or issues

Code:
{input.code_snippet}""",
            "architecture": f"""Analyze the code architecture and provide:
1. Overall architecture patterns used
2. Component responsibilities
3. Dependencies and relationships

Code:
{input.code_snippet}""",
            "optimization": f"""Analyze for optimization opportunities:
1. Performance bottlenecks
2. Memory usage concerns
3. Scalability issues

Code:
{input.code_snippet}""",
        }
        return prompts.get(input.analysis_type, prompts["general"])


class LLMIntegrationTool(Tool[Any, dict]):
    """Integration tool that combines LLM with static analysis."""

    def __init__(self, llm_tool: LLMAnalysisTool | None = None):
        self._llm_tool = llm_tool or LLMAnalysisTool()
        self._cache: Any = None

    async def execute(self, input: dict) -> dict:
        """Execute integrated analysis."""
        # First run static analysis
        static_result = input.get("static_result", {})

        # Then run LLM analysis
        code_snippet = input.get("code_snippet", "")
        if code_snippet:
            llm_input = LLMAnalysisInput(
                code_snippet=code_snippet,
                analysis_type=input.get("analysis_type", "general"),
            )
            llm_result = await self._llm_tool.execute(llm_input)
        else:
            llm_result = LLMAnalysisOutput(analysis="No code snippet available")

        # Combine results
        return {
            "static_analysis": static_result,
            "llm_analysis": {
                "analysis": llm_result.analysis,
                "confidence": llm_result.confidence,
                "entities": llm_result.entities,
                "recommendations": llm_result.recommendations,
            },
            "combined_insights": self._combine_insights(static_result, llm_result),
        }

    async def validate(self, input: dict) -> bool:
        """Validate input."""
        return isinstance(input, dict)

    def _combine_insights(self, static: dict, llm: LLMAnalysisOutput) -> dict:
        """Combine static and LLM insights."""
        return {
            "static_matches_llm": self._match_insights(static, llm),
            "key_findings": self._extract_findings(static, llm),
            "priority_items": self._prioritize_issues(static, llm),
        }

    def _match_insights(self, static: dict, llm: LLMAnalysisOutput) -> bool:
        """Check if static and LLM insights align."""
        # Simple heuristic - check for common patterns
        static_classes = static.get("classes", [])
        llm_entities = llm.entities
        return len(static_classes) > 0 and len(llm_entities) > 0

    def _extract_findings(self, static: dict, llm: LLMAnalysisOutput) -> list[str]:
        """Extract key findings from both analyses."""
        findings = []
        if static.get("complexity", "low") == "high":
            findings.append("High complexity detected")
        if llm.confidence < 0.7:
            findings.append("Low confidence in LLM analysis")
        return findings

    def _prioritize_issues(self, static: dict, llm: LLMAnalysisOutput) -> list[dict]:
        """Prioritize issues to address."""
        return [
            {
                "issue": "Review high complexity areas",
                "priority": "high" if static.get("complexity") == "high" else "medium",
                "source": "static",
            },
        ]
