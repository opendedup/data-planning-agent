"""
Data Product Requirement Prompt Schema

Models for the structured Data PRP output format.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ExecutiveSummary(BaseModel):
    """Executive summary section of the Data PRP."""

    objective: str = Field(description="One-sentence summary of the business goal")
    target_audience: str = Field(description="Who will use this analysis")
    key_question: str = Field(description="The primary question to be answered")


class DataRequirements(BaseModel):
    """Data requirements section of the Data PRP."""

    key_metrics: List[str] = Field(
        default_factory=list, description="List of key metrics needed"
    )
    dimensions_and_breakdowns: List[str] = Field(
        default_factory=list, description="How data should be segmented"
    )
    filters: List[str] = Field(default_factory=list, description="Conditions and constraints")
    data_gaps: List["DataGap"] = Field(
        default_factory=list, description="List of identified data gaps"
    )


class DataGap(BaseModel):
    """Represents a single identified data gap."""

    gap_id: str = Field(description="Unique identifier for the data gap, e.g., 'gap_01'")
    description: str = Field(description="Explanation of what data is missing")
    target_view: str = Field(
        description="The target view affected by this gap, e.g., 'live_bet_performance'"
    )
    required_information: str = Field(
        description="Detailed description of the data needed to resolve the gap, including potential columns and their purpose."
    )


class SuccessCriteria(BaseModel):
    """Success criteria section of the Data PRP."""

    primary_metric: str = Field(description="Main success indicator")
    timeline: str = Field(description="Delivery and update expectations")


class DataProductRequirementPrompt(BaseModel):
    """
    Complete Data Product Requirement Prompt structure.

    This represents the final output that will be generated as markdown.
    """

    executive_summary: ExecutiveSummary = Field(description="Executive summary section")
    business_context: str = Field(
        description="Detailed paragraph explaining the business scenario"
    )
    data_requirements: DataRequirements = Field(description="Data requirements section")
    success_criteria: SuccessCriteria = Field(description="Success criteria section")

    def to_markdown(self) -> str:
        """
        Convert the Data PRP to markdown format.

        Returns:
            Formatted markdown string following the exact PRD specification
        """
        lines = []

        # Title
        lines.append("# Data Product Requirement Prompt")
        lines.append("")

        # 1. Executive Summary
        lines.append("## 1. Executive Summary")
        lines.append("")
        lines.append(f"* **Objective:** {self.executive_summary.objective}")
        lines.append(f"* **Target Audience:** {self.executive_summary.target_audience}")
        lines.append(f"* **Key Question:** {self.executive_summary.key_question}")
        lines.append("")

        # 2. Business Context
        lines.append("## 2. Business Context")
        lines.append("")
        lines.append(self.business_context)
        lines.append("")

        # 3. Data Requirements
        lines.append("## 3. Data Requirements")
        lines.append("")

        # 3.1. Key Metrics
        lines.append("### 3.1. Key Metrics")
        lines.append("")
        if self.data_requirements.key_metrics:
            for metric in self.data_requirements.key_metrics:
                lines.append(f"* {metric}")
        else:
            lines.append("* _To be determined_")
        lines.append("")

        # 3.2. Dimensions & Breakdowns
        lines.append("### 3.2. Dimensions & Breakdowns")
        lines.append("")
        if self.data_requirements.dimensions_and_breakdowns:
            for dimension in self.data_requirements.dimensions_and_breakdowns:
                lines.append(f"* {dimension}")
        else:
            lines.append("* _To be determined_")
        lines.append("")

        # 3.3. Filters
        lines.append("### 3.3. Filters")
        lines.append("")
        if self.data_requirements.filters:
            for filter_item in self.data_requirements.filters:
                lines.append(f"* {filter_item}")
        else:
            lines.append("* _No specific filters_")
        lines.append("")

        # 4. Success Criteria
        lines.append("## 4. Success Criteria")
        lines.append("")
        lines.append(f"* **Primary Metric:** {self.success_criteria.primary_metric}")
        lines.append(f"* **Timeline:** {self.success_criteria.timeline}")
        lines.append("")

        return "\n".join(lines)


class PRPAnalysisResult(BaseModel):
    """Structured result from the first-pass analysis of the user conversation."""

    target_views: List[dict] = Field(
        description="List of conceptual target views or tables that need to be created."
    )
    data_gaps: List[DataGap] = Field(
        description="List of identified data gaps that need to be resolved."
    )

