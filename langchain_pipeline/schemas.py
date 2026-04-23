"""
Pydantic output schemas for structured LLM output.

These classes describe exactly what each LLM stage must return. LangChain
uses them to make the model produce JSON with guaranteed field names and
types — no parsing surprises.
"""
from pydantic import BaseModel, Field
from typing import List


class LevelClassification(BaseModel):
    """Stage 1 output — which abstraction level(s) the change touches."""
    # the LLM puts scope labels here: parameter_level, method_level, class_level
    levels: List[str] = Field(
        description=(
            "List of abstraction levels present in the code change. "
            "Each must be one of: parameter_level, method_level, class_level. "
            "A change can have multiple levels."
        )
    )


class DetectedType(BaseModel):
    """One refactoring the LLM spotted, with the diff line it cites."""
    # e.g. 'Rename Method' for canonical, 'UnknownType: foo' for novel
    type_name: str = Field(description="Refactoring type name (e.g. 'Rename Method', 'Add Parameter', or 'UnknownType: brief description')")

    # the actual diff line the model points to as proof
    evidence: str = Field(description="Quoted diff line(s) supporting this detection")


class DetectedRefactorings(BaseModel):
    """Stage 2 output — two buckets: canonical 39 types vs novel (Unknown)."""
    # bucket A: matches one of the 39 canonical types
    defined: List[DetectedType] = Field(
        default_factory=list,
        description="Detected types from the known 39-type set, with evidence quotes."
    )

    # bucket B: novel patterns (the LLM couldn't name with the 39)
    undefined: List[DetectedType] = Field(
        default_factory=list,
        description="Novel patterns not in the known list. Use 'UnknownType: brief description' as type_name."
    )


class AdditionalType(BaseModel):
    """One extra type Stage 3 found that Stage 2 missed."""
    type_name: str = Field(description="Canonical refactoring type name")
    evidence: str = Field(description="Quoted diff line(s) supporting this detection")


class AdditionalDetections(BaseModel):
    """Stage 3 output — types added on top of Stage 2's list."""
    # empty list means Stage 2 already found everything
    additional: List[AdditionalType] = Field(
        default_factory=list,
        description="Additional refactoring types found. Empty list if none."
    )


class VerifiedType(BaseModel):
    """Stage 4: one detection plus a 0–100 confidence score and its evidence."""
    type_name: str = Field(description="Canonical refactoring type name")

    # the score we later threshold against in precision-mode
    confidence: int = Field(
        description="Confidence 0-100. 90+=certain, 60-89=probable, 30-59=uncertain, 0-29=very unlikely",
        ge=0, le=100,
    )
    evidence: str = Field(description="Quoted diff line(s) supporting the detection")


class VerifiedRefactoringsWithConfidence(BaseModel):
    """Stage 4 output — the final per-detection verdicts with confidences."""
    # if nothing survived verification the list is empty (valid)
    verified: List[VerifiedType] = Field(
        default_factory=list,
        description="List of verified detections with confidence scores. Empty if nothing survives.",
    )
