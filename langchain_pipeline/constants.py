"""
Constants, type mappings, confusion hints, and helper functions for
the LangChain refactoring-detection pipeline.

Standalone module -- only standard-library imports.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Set

# LEVEL -> TYPE MAPPING

PARAMETER_LEVEL_TYPES: List[str] = [
    "Add Parameter",
    "Remove Parameter",
    "Rename Parameter",
    "Reorder Parameter",
    "Split Parameter",
    "Localize Parameter",
    "Parameterize Variable",
    "Parameterize Attribute",
    "Change Variable Type",
]

METHOD_LEVEL_TYPES: List[str] = [
    "Extract Method",
    "Extract And Move Method",
    "Inline Method",
    "Move Method",
    "Move And Rename Method",
    "Rename Method",
    "Rename Variable",
    "Rename Attribute",
    "Extract Variable",
    "Inline Variable",
    "Replace Variable With Attribute",
    "Replace Attribute With Variable",
    "Split Conditional",
    "Invert Condition",
    "Move Code",
]

CLASS_LEVEL_TYPES: List[str] = [
    "Move Class",
    "Rename Class",
    "Move And Rename Class",
    "Extract Class",
    "Extract Subclass",
    "Extract Superclass",
    "Move Attribute",
    "Pull Up Method",
    "Push Down Method",
    "Pull Up Attribute",
    "Push Down Attribute",
    "Encapsulate Attribute",
    "Add Method Annotation",
    "Remove Method Annotation",
    "Add Class Annotation",
]

LEVEL_TYPES: Dict[str, List[str]] = {
    "parameter_level": PARAMETER_LEVEL_TYPES,
    "method_level": METHOD_LEVEL_TYPES,
    "class_level": CLASS_LEVEL_TYPES,
}

ALL_KNOWN_TYPES: Set[str] = set(
    PARAMETER_LEVEL_TYPES + METHOD_LEVEL_TYPES + CLASS_LEVEL_TYPES
)

# CONFUSION HINTS  (derived from Run 1-3 error analysis)

CONFUSION_HINTS: Dict[str, str] = {
    "Add Parameter": (
        "VERIFY DIRECTION: Read function signature in before/after. "
        "If def line LOSES a param -> Remove Parameter, not Add Parameter."
    ),
    "Remove Parameter": (
        "VERIFY DIRECTION: Read function signature in before/after. "
        "If def line GAINS a param -> Add Parameter, not Remove Parameter."
    ),
    "Rename Variable": (
        "VERIFY SCOPE: local var -> Rename Variable; "
        "def line param -> Rename Parameter; "
        "self.x/cls.x -> Rename Attribute."
    ),
    "Rename Attribute": (
        "VERIFY: Must be self.x or cls.x. "
        "Not local var (-> Rename Variable) or def param (-> Rename Parameter)."
    ),
    "Rename Parameter": (
        "VERIFY: Must appear in function def line as a parameter. "
        "Not local var (-> Rename Variable) or self.x (-> Rename Attribute)."
    ),
    "Split Conditional": (
        "HIGH FP: Only detect if one if/elif is split into TWO separate blocks. "
        "Not minor condition rewrites."
    ),
    "Extract Method": (
        "VERIFY: New function def must exist containing extracted code. "
        "Not in-place refactor within same function."
    ),
    "Inline Variable": (
        "VERIFY: Variable eliminated by substituting value inline. "
        "Not a rename."
    ),
    "Extract Variable": (
        "VERIFY: New variable introduced to name a sub-expression. "
        "Not a rename."
    ),
    "Move Method": (
        "VERIFY MOVE: Method must have moved between classes/files. "
        "Use get_full_code if diff is ambiguous."
    ),
    "Move Class": (
        "VERIFY MOVE: Class must have moved to another module. "
        "Use get_full_code if ambiguous."
    ),
    "Move And Rename Method": (
        "ONE atomic op -- not separate Move Method + Rename Method."
    ),
    "Move And Rename Class": (
        "ONE atomic op -- not separate Move Class + Rename Class."
    ),
    "Extract And Move Method": (
        "ONE atomic op -- not separate Extract Method + Move Method."
    ),
    "Extract Subclass": (
        "CHECK MISSED: New class inheriting from existing class "
        "with methods moved to it?"
    ),
    "Extract Superclass": (
        "CHECK MISSED: New parent class created to share methods "
        "between two classes?"
    ),
    "Invert Condition": (
        "CHECK MISSED: Boolean condition negated "
        "(x < y -> x >= y, or if-body swapped)?"
    ),
    "Pull Up Method": (
        "VERIFY HIERARCHY: Method moved FROM subclass TO superclass."
    ),
    "Push Down Method": (
        "VERIFY HIERARCHY: Method moved FROM superclass TO subclass."
    ),
}

# MISSING HINTS  (co-occurrence map for Stage 3B missing-type detector)

# Seeded from iter 1 GT co-occurrence: for each detected type, the top-3 types
# most commonly appearing in the same case in iter 1 ground truth. Used by
# Stage 3 Call B to probe for types Stage 2 may have missed.
MISSING_HINTS: Dict[str, List[str]] = {
    "Add Class Annotation":            ["Remove Method Annotation", "Move Method", "Extract Class"],
    "Add Method Annotation":           ["Remove Method Annotation", "Rename Method", "Extract Variable"],
    "Add Parameter":                   ["Remove Parameter", "Rename Variable", "Rename Parameter"],
    "Change Variable Type":            ["Rename Variable", "Extract Variable", "Add Parameter"],
    "Encapsulate Attribute":           ["Move And Rename Method", "Rename Variable", "Rename Method"],
    "Extract And Move Method":         ["Parameterize Variable", "Inline Variable", "Rename Variable"],
    "Extract Class":                   ["Move Method", "Move Attribute", "Add Parameter"],
    "Extract Method":                  ["Rename Variable", "Rename Parameter", "Remove Parameter"],
    "Extract Subclass":                ["Push Down Method", "Push Down Attribute", "Remove Parameter"],
    "Extract Superclass":              ["Pull Up Method", "Pull Up Attribute", "Move Code"],
    "Extract Variable":                ["Rename Variable", "Add Parameter", "Extract Method"],
    "Inline Method":                   ["Rename Variable", "Inline Variable", "Rename Parameter"],
    "Inline Variable":                 ["Rename Variable", "Extract Method", "Extract Variable"],
    "Invert Condition":                ["Extract Variable", "Rename Variable", "Extract Method"],
    "Localize Parameter":              ["Add Parameter", "Remove Parameter", "Rename Parameter"],
    "Move And Rename Class":           ["Move Class", "Rename Method", "Move Method"],
    "Move And Rename Method":          ["Remove Parameter", "Rename Variable", "Rename Parameter"],
    "Move Attribute":                  ["Move Method", "Extract Class", "Add Parameter"],
    "Move Class":                      ["Rename Method", "Move And Rename Class", "Extract Class"],
    "Move Code":                       ["Move Attribute", "Move Method", "Extract Class"],
    "Move Method":                     ["Move Attribute", "Add Parameter", "Remove Parameter"],
    "Parameterize Attribute":          ["Add Parameter", "Remove Parameter", "Rename Variable"],
    "Parameterize Variable":           ["Add Parameter", "Remove Parameter", "Rename Variable"],
    "Pull Up Attribute":               ["Extract Superclass", "Pull Up Method", "Inline Variable"],
    "Pull Up Method":                  ["Extract Superclass", "Pull Up Attribute", "Inline Variable"],
    "Push Down Attribute":             ["Push Down Method", "Extract Subclass", "Rename Class"],
    "Push Down Method":                ["Extract Subclass", "Push Down Attribute", "Add Parameter"],
    "Remove Method Annotation":        ["Add Method Annotation", "Extract Variable", "Remove Parameter"],
    "Remove Parameter":                ["Add Parameter", "Rename Parameter", "Rename Method"],
    "Rename Attribute":                ["Rename Method", "Add Parameter", "Rename Variable"],
    "Rename Class":                    ["Remove Parameter", "Rename Parameter", "Move Method"],
    "Rename Method":                   ["Remove Parameter", "Add Parameter", "Rename Variable"],
    "Rename Parameter":                ["Remove Parameter", "Rename Variable", "Add Parameter"],
    "Rename Variable":                 ["Add Parameter", "Extract Variable", "Rename Parameter"],
    "Reorder Parameter":               ["Add Parameter", "Remove Parameter", "Rename Method"],
    "Replace Attribute With Variable": ["Extract Variable", "Inline Variable", "Replace Variable With Attribute"],
    "Replace Variable With Attribute": ["Remove Parameter", "Rename Variable", "Extract Method"],
    "Split Conditional":               ["Extract Method", "Add Parameter"],
    "Split Parameter":                 ["Add Parameter", "Rename Parameter", "Extract Variable"],
}

# Types Stage 2 systematically under-detects. Always include these in
# Call 3B's candidate list (subject to cap), regardless of what Stage 2 found.
ALWAYS_CHECK_MISSING: List[str] = [
    "Invert Condition",
    "Split Conditional",
    "Extract Subclass",
    "Extract Superclass",
    "Encapsulate Attribute",
    "Move And Rename Method",
    "Move And Rename Class",
    "Extract And Move Method",
]

# Data-mined v2: top-4 GT co-occurrences from iter 1 + iter 2 (660 cases).
# Generated by experiments/A1_confusion_mining.py. Wider top-4 (vs curated
# top-3) and uses real frequency. Offline candidate-recall ablation
# (experiments/A2_candidate_recall.py) shows this lifts the Stage 3 recall
# ceiling from 40.4% to 45.2% at cap=7, and to 52.2% at cap=12.
MISSING_HINTS_MINED_V2: Dict[str, List[str]] = {
    "Add Class Annotation":            ["Remove Method Annotation", "Move Method", "Extract Class", "Push Down Attribute"],
    "Add Method Annotation":           ["Remove Method Annotation", "Rename Method", "Extract Variable", "Rename Variable"],
    "Add Parameter":                   ["Remove Parameter", "Rename Variable", "Rename Parameter", "Rename Method"],
    "Change Variable Type":            ["Rename Variable", "Extract Variable", "Add Parameter", "Move Class"],
    "Encapsulate Attribute":           ["Rename Variable", "Move And Rename Method", "Rename Method", "Add Parameter"],
    "Extract And Move Method":         ["Parameterize Variable", "Inline Variable", "Rename Variable", "Extract Variable"],
    "Extract Class":                   ["Move Method", "Move Attribute", "Add Parameter", "Move Code"],
    "Extract Method":                  ["Rename Variable", "Rename Parameter", "Remove Parameter", "Inline Variable"],
    "Extract Subclass":                ["Push Down Method", "Push Down Attribute", "Remove Parameter", "Rename Class"],
    "Extract Superclass":              ["Pull Up Attribute", "Pull Up Method", "Move Method", "Move Code"],
    "Extract Variable":                ["Rename Variable", "Add Parameter", "Extract Method", "Inline Variable"],
    "Inline Method":                   ["Rename Variable", "Inline Variable", "Rename Parameter", "Extract Method"],
    "Inline Variable":                 ["Rename Variable", "Extract Method", "Extract Variable", "Add Parameter"],
    "Invert Condition":                ["Extract Variable", "Rename Variable", "Extract Method", "Add Parameter"],
    "Localize Parameter":              ["Add Parameter", "Remove Parameter", "Rename Parameter", "Rename Variable"],
    "Move And Rename Class":           ["Move Class", "Rename Method", "Move Method", "Rename Parameter"],
    "Move And Rename Method":          ["Remove Parameter", "Rename Variable", "Rename Parameter", "Rename Method"],
    "Move Attribute":                  ["Move Method", "Extract Class", "Add Parameter", "Rename Variable"],
    "Move Class":                      ["Rename Method", "Move And Rename Class", "Extract Class", "Move Attribute"],
    "Move Code":                       ["Move Attribute", "Move Method", "Extract Class", "Remove Parameter"],
    "Move Method":                     ["Move Attribute", "Add Parameter", "Remove Parameter", "Extract Class"],
    "Parameterize Attribute":          ["Add Parameter", "Rename Variable", "Remove Parameter", "Inline Variable"],
    "Parameterize Variable":           ["Add Parameter", "Remove Parameter", "Rename Variable", "Extract Method"],
    "Pull Up Attribute":               ["Extract Superclass", "Inline Variable", "Move Method", "Pull Up Method"],
    "Pull Up Method":                  ["Extract Superclass", "Inline Variable", "Move Method", "Pull Up Attribute"],
    "Push Down Attribute":             ["Push Down Method", "Extract Subclass", "Rename Class", "Add Parameter"],
    "Push Down Method":                ["Extract Subclass", "Push Down Attribute", "Add Parameter", "Extract Variable"],
    "Remove Method Annotation":        ["Add Method Annotation", "Extract Variable", "Remove Parameter", "Add Parameter"],
    "Remove Parameter":                ["Add Parameter", "Rename Parameter", "Rename Method", "Rename Variable"],
    "Rename Attribute":                ["Add Parameter", "Rename Method", "Rename Variable", "Remove Parameter"],
    "Rename Class":                    ["Remove Parameter", "Rename Parameter", "Move Method", "Add Parameter"],
    "Rename Method":                   ["Remove Parameter", "Add Parameter", "Rename Variable", "Rename Parameter"],
    "Rename Parameter":                ["Remove Parameter", "Rename Variable", "Add Parameter", "Extract Method"],
    "Rename Variable":                 ["Add Parameter", "Extract Variable", "Rename Parameter", "Extract Method"],
    "Reorder Parameter":               ["Add Parameter", "Remove Parameter", "Rename Method", "Rename Variable"],
    "Replace Attribute With Variable": ["Extract Variable", "Inline Variable", "Replace Variable With Attribute", "Move And Rename Method"],
    "Replace Variable With Attribute": ["Remove Parameter", "Rename Variable", "Extract Method", "Inline Variable"],
    "Split Conditional":               ["Add Parameter", "Extract Method"],
    "Split Parameter":                 ["Add Parameter", "Rename Parameter", "Extract Variable", "Remove Parameter"],
}

# Data-mined v3: NPMI-ranked (Bouma 2009 "Normalized PMI"). Same source data
# as v2 (660 cases from iter 1+2) but ranked by NPMI instead of raw count.
# NPMI is bounded [-1, 1], handles rare events robustly, and suppresses
# high-frequency types (Add Parameter, Rename Variable) from cluttering every
# hint list. 39/39 types differ from v2 in their top-4 partners.
# Generated by experiments/A1_confusion_mining.py; copied from
# experiments/output/A1_mined_hints_v3.py.
MISSING_HINTS_MINED_V3: Dict[str, List[str]] = {
    "Add Class Annotation": ["Remove Method Annotation", "Change Return Type", "Add Parameter", "Remove Parameter"],
    "Add Method Annotation": ["Remove Method Annotation", "Localize Parameter", "Rename Method", "Extract Variable", "Change Return Type", "Add Parameter", "Rename Variable"],
    "Add Parameter": ["Parameterize Variable", "Split Parameter", "Reorder Parameter", "Change Return Type", "Inline Variable", "Rename Variable"],
    "Change Variable Type": ["Rename Class", "Rename Variable", "Move Class", "Extract Variable", "Change Return Type", "Add Method Annotation"],
    "Encapsulate Attribute": ["Move And Rename Method", "Rename Method", "Rename Variable", "Change Return Type"],
    "Extract And Move Method": ["Pull Up Attribute", "Pull Up Method", "Parameterize Variable", "Extract Superclass"],
    "Extract Class": ["Move Attribute", "Move Method", "Move Code", "Invert Condition", "Rename Variable", "Change Return Type", "Inline Variable"],
    "Extract Method": ["Inline Variable", "Parameterize Variable", "Parameterize Attribute", "Rename Parameter", "Change Return Type", "Rename Variable"],
    "Extract Subclass": ["Push Down Method", "Push Down Attribute", "Extract Superclass", "Rename Class", "Move Method", "Change Return Type"],
    "Extract Superclass": ["Pull Up Attribute", "Pull Up Method", "Extract Subclass", "Push Down Method", "Move Attribute", "Move And Rename Method", "Rename Variable"],
    "Extract Variable": ["Invert Condition", "Rename Variable", "Extract Method", "Push Down Method", "Change Return Type", "Inline Variable"],
    "Inline Method": ["Parameterize Attribute", "Rename Variable", "Replace Variable With Attribute", "Change Return Type", "Add Method Annotation"],
    "Inline Variable": ["Inline Method", "Extract Method", "Parameterize Attribute", "Replace Attribute With Variable", "Change Return Type", "Replace Variable With Attribute", "Add Method Annotation"],
    "Invert Condition": ["Extract Class", "Move Code", "Parameterize Variable", "Extract Method", "Inline Variable"],
    "Localize Parameter": ["Move Code", "Remove Parameter", "Add Method Annotation", "Move And Rename Method"],
    "Move And Rename Class": ["Move Class", "Move Method", "Move Attribute", "Rename Method"],
    "Move And Rename Method": ["Encapsulate Attribute", "Move Code", "Replace Variable With Attribute", "Extract Class"],
    "Move Attribute": ["Move Method", "Extract Class", "Move Code", "Move And Rename Method", "Replace Variable With Attribute", "Change Return Type", "Inline Variable"],
    "Move Class": ["Move And Rename Class", "Extract Class", "Change Variable Type", "Rename Method", "Inline Variable", "Move Attribute", "Rename Parameter"],
    "Move Code": ["Extract Class", "Move Attribute", "Move And Rename Method", "Extract Superclass", "Move Method"],
    "Move Method": ["Move Attribute", "Extract Class", "Move Code", "Remove Parameter", "Add Parameter", "Change Return Type", "Pull Up Method"],
    "Parameterize Attribute": ["Inline Method", "Parameterize Variable", "Inline Variable", "Add Parameter"],
    "Parameterize Variable": ["Parameterize Attribute", "Extract Method", "Extract And Move Method", "Add Parameter"],
    "Pull Up Attribute": ["Extract Superclass", "Pull Up Method", "Extract And Move Method", "Inline Variable"],
    "Pull Up Method": ["Extract Superclass", "Pull Up Attribute", "Extract And Move Method", "Inline Variable", "Move Attribute", "Move And Rename Method"],
    "Push Down Attribute": ["Extract Subclass", "Push Down Method", "Rename Class", "Remove Parameter"],
    "Push Down Method": ["Extract Subclass", "Push Down Attribute", "Extract Superclass", "Rename Class"],
    "Remove Method Annotation": ["Add Method Annotation", "Add Class Annotation", "Rename Class", "Inline Method"],
    "Remove Parameter": ["Parameterize Attribute", "Localize Parameter", "Rename Parameter", "Rename Method", "Change Return Type", "Inline Variable", "Parameterize Variable"],
    "Rename Attribute": ["Rename Class", "Replace Variable With Attribute", "Rename Method", "Move Attribute", "Change Return Type", "Add Parameter", "Inline Variable"],
    "Rename Class": ["Extract Subclass", "Push Down Attribute", "Push Down Method", "Change Variable Type", "Change Return Type", "Move Method", "Move And Rename Class"],
    "Rename Method": ["Remove Parameter", "Localize Parameter", "Reorder Parameter", "Move And Rename Method", "Change Return Type", "Change Method Access Modifier"],
    "Rename Parameter": ["Extract Method", "Split Parameter", "Rename Variable", "Remove Parameter", "Change Return Type", "Inline Variable", "Replace Attribute With Variable"],
    "Rename Variable": ["Inline Method", "Parameterize Attribute", "Rename Parameter", "Extract Method", "Change Return Type", "Inline Variable", "Extract Variable"],
    "Reorder Parameter": ["Add Parameter", "Remove Parameter", "Rename Method", "Move Method"],
    "Replace Attribute With Variable": ["Replace Variable With Attribute", "Inline Variable", "Extract Variable"],
    "Replace Variable With Attribute": ["Replace Attribute With Variable", "Move And Rename Method", "Inline Method", "Move Code"],
    "Split Conditional": ["Change Return Type", "Extract Variable", "Remove Parameter"],
    "Split Parameter": ["Rename Class", "Rename Parameter", "Add Parameter", "Parameterize Variable"],

}

# GPT-4o-mini's own missing-hints dict. Initialized from the
# pre-calibration V3 snapshot so GPT calibrates independently
# from qwen3.5's calibration path. Activate via
# LANGCHAIN_HINTS_VERSION=v3_gpt
MISSING_HINTS_GPT4MINI: Dict[str, List[str]] = {
    "Add Class Annotation": ["Remove Method Annotation", "Change Return Type", "Inline Variable", "Rename Attribute"],
    "Add Method": ["Change Return Type", "Rename Variable", "Move And Rename Method"],
    "Add Method Annotation": ["Remove Method Annotation", "Rename Method", "Extract Variable", "Change Return Type", "Inline Variable", "Rename Attribute"],
    "Add Parameter": ["Parameterize Variable", "Change Return Type", "Inline Variable", "Extract Variable"],
    "Change Variable Type": ["Rename Class", "Rename Variable", "Move Class", "Extract Variable", "Change Return Type", "Rename Attribute", "Inline Variable"],
    "Encapsulate Attribute": ["Move And Rename Method", "Rename Method", "Rename Variable", "Change Method Access Modifier"],
    "Extract And Move Method": ["Pull Up Attribute", "Pull Up Method", "Parameterize Variable", "Extract Superclass"],
    "Extract Class": ["Move Attribute", "Move Method", "Move Code", "Invert Condition", "Change Return Type", "Move And Rename Method", "Move Class"],
    "Extract Method": ["Inline Variable", "Parameterize Variable", "Parameterize Attribute", "Rename Parameter", "Change Return Type", "Extract And Move Method"],
    "Extract Subclass": ["Push Down Method", "Push Down Attribute", "Extract Superclass", "Rename Class", "Change Return Type", "Rename Parameter", "Move Method"],
    "Extract Superclass": ["Pull Up Method", "Extract Subclass", "Push Down Method", "Rename Parameter", "Extract And Move Method", "Move Method"],
    "Extract Variable": ["Invert Condition", "Rename Variable", "Change Return Type", "Rename Attribute", "Inline Variable"],
    "Inline Method": ["Inline Variable", "Rename Variable", "Replace Variable With Attribute", "Change Return Type", "Parameterize Variable"],
    "Inline Variable": ["Replace Attribute With Variable", "Change Return Type", "Parameterize Variable", "Move Method"],
    "Invert Condition": ["Extract Class", "Move Code", "Parameterize Variable", "Move Method"],
    "Localize Parameter": ["Move Code", "Remove Parameter", "Add Method Annotation", "Move And Rename Method", "Change Return Type", "Inline Variable", "Rename Attribute"],
    "Move And Rename Class": ["Move Class", "Move Method", "Rename Method", "Move Source Folder", "Rename Variable"],
    "Move And Rename Method": ["Encapsulate Attribute", "Move Code", "Replace Variable With Attribute", "Extract Class", "Change Method Access Modifier"],
    "Move Attribute": ["Move Method", "Move Code", "Move And Rename Method", "Rename Attribute", "Change Return Type", "Move Class"],
    "Move Class": ["Move And Rename Class", "Change Variable Type", "Rename Method", "Change Method Access Modifier", "Change Return Type", "Move Attribute"],
    "Move Code": ["Move And Rename Method", "Extract Superclass", "Extract Variable", "Change Return Type", "Parameterize Variable"],
    "Move Method": ["Move Code", "Remove Parameter", "Change Return Type", "Rename Attribute"],
    "Parameterize Attribute": ["Inline Method", "Parameterize Variable", "Inline Variable", "Add Parameter", "Change Return Type", "Rename Attribute"],
    "Parameterize Variable": ["Parameterize Attribute", "Extract Method", "Extract And Move Method", "Add Parameter", "Inline Variable", "Change Return Type", "Change Method Access Modifier"],
    "Pull Up Attribute": ["Extract Superclass", "Pull Up Method", "Extract And Move Method", "Inline Variable", "Change Return Type", "Move Method"],
    "Pull Up Method": ["Extract Superclass", "Extract And Move Method", "Inline Variable", "Change Return Type", "Rename Parameter"],
    "Push Down Attribute": ["Extract Subclass", "Push Down Method", "Rename Class", "Remove Parameter", "Change Return Type", "Move Method", "Rename Parameter"],
    "Push Down Method": ["Push Down Attribute", "Extract Superclass", "Rename Class", "Change Return Type", "Move Method", "Extract And Move Method"],
    "REMOVE": ["Move Class", "Change Return Type", "Rename Variable"],
    "Remove": ["Rename Variable"],
    "Remove Method": ["Change Return Type"],
    "Remove Method Annotation": ["Add Method Annotation", "Rename Class", "Change Return Type", "Inline Variable", "Rename Attribute"],
    "Remove Parameter": ["Rename Method", "Change Return Type", "Inline Variable", "Rename Attribute"],
    "Rename Attribute": ["Rename Class", "Replace Variable With Attribute", "Move Attribute", "Change Return Type", "Move Class", "Inline Variable"],
    "Rename Class": ["Push Down Attribute", "Push Down Method", "Change Variable Type", "Change Return Type", "Rename Attribute", "Move Method"],
    "Rename Method": ["Remove Parameter", "Change Return Type", "Inline Variable", "Change Method Access Modifier"],
    "Rename Parameter": ["Rename Variable", "Change Return Type", "Inline Variable", "Rename Attribute"],
    "Rename Variable": ["Rename Parameter", "Change Return Type", "Inline Variable", "Rename Attribute"],
    "Reorder Parameter": ["Add Parameter", "Remove Parameter", "Rename Method", "Move Method", "Change Return Type", "Inline Variable", "Rename Attribute"],
    "Replace Attribute With Variable": ["Replace Variable With Attribute", "Inline Variable", "Change Return Type"],
    "Replace Variable With Attribute": ["Replace Attribute With Variable", "Move And Rename Method", "Change Return Type", "Parameterize Variable", "Pull Up Attribute"],
    "Split Conditional": ["Extract Variable", "Inline Variable", "Extract And Move Method"],
    "Split Parameter": ["Rename Class", "Rename Parameter", "Add Parameter", "Parameterize Variable", "Inline Variable", "Change Return Type", "Rename Attribute"],

}

# Data-mined v2: NPMI-ranked FP→GT confusion partners (Bouma 2009).
# For each type T, lists types most often present in GT when the model
# wrongly predicted T. Used by Stage 4 verifier as data-mined direction
# rules complementing the curated CONFUSION_HINTS.
CONFUSION_HINTS_MINED_V2: Dict[str, List[str]] = {
    "Add Class Annotation":            ["Remove Parameter", "Add Parameter"],
    "Add Method Annotation":           [],  # no data
    "Add Parameter":                   ["Extract Subclass", "Pull Up Method", "Push Down Method", "Split Parameter"],
    "Change Variable Type":            ["Add Class Annotation", "Extract Class", "Inline Variable", "Move Method"],
    "Encapsulate Attribute":           ["Remove Parameter"],
    "Extract And Move Method":         ["Extract Subclass", "Move And Rename Method", "Push Down Method", "Localize Parameter"],
    "Extract Class":                   ["Extract Subclass", "Rename Class", "Extract Superclass", "Push Down Method"],
    "Extract Method":                  ["Extract And Move Method", "Reorder Parameter", "Parameterize Variable", "Localize Parameter"],
    "Extract Subclass":                ["Move Method", "Move Attribute", "Extract Superclass", "Pull Up Attribute"],
    "Extract Superclass":              ["Move Attribute", "Move Method", "Rename Class", "Inline Method"],
    "Extract Variable":                ["Split Conditional", "Localize Parameter", "Invert Condition", "Extract And Move Method"],
    "Inline Method":                   ["Rename Variable", "Encapsulate Attribute", "Rename Parameter", "Rename Method"],
    "Inline Variable":                 ["Rename Variable", "Add Parameter"],
    "Invert Condition":                [],  # no data
    "Localize Parameter":              [],  # no data
    "Move And Rename Class":           ["Move Attribute", "Move Method", "Rename Variable"],
    "Move And Rename Method":          ["Rename Method"],
    "Move Attribute":                  [],  # no data
    "Move Class":                      ["Move Method", "Remove Parameter"],
    "Move Code":                       ["Replace Variable With Attribute", "Invert Condition", "Localize Parameter", "Reorder Parameter"],
    "Move Method":                     ["Extract Subclass", "Push Down Method", "Push Down Attribute", "Move And Rename Method"],
    "Parameterize Attribute":          [],  # no data
    "Parameterize Variable":           ["Extract Class", "Move Method", "Extract Method", "Add Parameter"],
    "Pull Up Attribute":               [],  # no data
    "Pull Up Method":                  ["Rename Class", "Rename Parameter"],
    "Push Down Attribute":             [],  # no data
    "Push Down Method":                [],  # no data
    "Remove Method Annotation":        [],  # no data
    "Remove Parameter":                ["Split Parameter", "Localize Parameter", "Replace Variable With Attribute", "Change Variable Type"],
    "Rename Attribute":                ["Split Parameter", "Extract Superclass", "Remove Method Annotation", "Parameterize Variable"],
    "Rename Class":                    ["Change Variable Type", "Rename Method", "Add Method Annotation", "Inline Variable"],
    "Rename Method":                   ["Change Variable Type", "Inline Variable", "Move And Rename Method", "Extract Method"],
    "Rename Parameter":                ["Split Parameter", "Parameterize Variable", "Add Parameter", "Remove Parameter"],
    "Rename Variable":                 ["Extract Method", "Remove Parameter", "Add Parameter"],
    "Reorder Parameter":               ["Split Parameter", "Add Parameter", "Rename Attribute", "Rename Parameter"],
    "Replace Attribute With Variable": ["Remove Parameter", "Move Attribute"],
    "Replace Variable With Attribute": ["Move And Rename Method", "Remove Parameter", "Move Code", "Extract Class"],
    "Split Conditional":               ["Extract Variable", "Add Parameter", "Rename Method", "Remove Parameter"],
    "Split Parameter":                 [],  # no data
}

# v4: FP-Growth association rules over the full 14k train corpus, ranked
# by lift*confidence. See experiments/A1_mined_hints_v4_apriori.py.
MISSING_HINTS_MINED_V4: Dict[str, List[str]] = {
    'Add Class Annotation'             : [],
    'Add Method Annotation'            : ["Remove Method Annotation", "Rename Method", "Add Parameter", "Remove Parameter"],
    'Add Parameter'                    : ["Remove Parameter", "Rename Parameter", "Rename Method", "Extract Variable"],
    'Change Variable Type'             : ["Rename Class", "Move And Rename Class", "Rename Variable", "Move Class"],
    'Encapsulate Attribute'            : [],
    'Extract And Move Method'          : ["Parameterize Variable", "Rename Variable", "Inline Variable", "Extract Superclass"],
    'Extract Class'                    : ["Move Attribute", "Move Method", "Move And Rename Method", "Move Code"],
    'Extract Method'                   : ["Parameterize Variable", "Rename Variable", "Extract Variable", "Inline Variable"],
    'Extract Subclass'                 : [],
    'Extract Superclass'               : ["Pull Up Method", "Extract And Move Method", "Move Class", "Rename Method"],
    'Extract Variable'                 : ["Rename Variable", "Add Parameter", "Inline Variable", "Extract Method"],
    'Inline Method'                    : ["Rename Variable", "Remove Parameter", "Rename Parameter", "Rename Method"],
    'Inline Variable'                  : ["Rename Variable", "Extract Variable", "Remove Parameter", "Rename Parameter"],
    'Invert Condition'                 : ["Extract Variable", "Extract Method", "Rename Variable", "Inline Variable"],
    'Localize Parameter'               : ["Remove Parameter", "Rename Parameter", "Rename Method", "Add Parameter"],
    'Move And Rename Class'            : ["Move Class", "Change Variable Type", "Move Method", "Rename Method"],
    'Move And Rename Method'           : ["Move Method", "Extract Class", "Move Attribute", "Remove Parameter"],
    'Move Attribute'                   : ["Extract Class", "Move Method", "Move And Rename Method", "Move Code"],
    'Move Class'                       : ["Move And Rename Class", "Move Method", "Move And Rename Method", "Rename Method"],
    'Move Code'                        : ["Extract Class", "Move Attribute", "Move Method", "Move And Rename Method"],
    'Move Method'                      : ["Extract Class", "Move Attribute", "Move And Rename Method", "Remove Parameter"],
    'Parameterize Attribute'           : ["Add Parameter"],
    'Parameterize Variable'            : ["Extract Method", "Extract And Move Method", "Rename Variable", "Remove Parameter"],
    'Pull Up Attribute'                : [],
    'Pull Up Method'                   : ["Extract Superclass", "Add Parameter"],
    'Push Down Attribute'              : [],
    'Push Down Method'                 : [],
    'Remove Method Annotation'         : ["Add Method Annotation", "Remove Parameter", "Rename Method", "Move And Rename Method"],
    'Remove Parameter'                 : ["Rename Parameter", "Add Parameter", "Rename Method", "Move Method"],
    'Rename Attribute'                 : ["Rename Parameter", "Remove Parameter", "Rename Variable", "Rename Method"],
    'Rename Class'                     : ["Change Variable Type", "Rename Method", "Rename Attribute", "Remove Parameter"],
    'Rename Method'                    : ["Rename Parameter", "Remove Parameter", "Add Parameter", "Rename Variable"],
    'Rename Parameter'                 : ["Remove Parameter", "Rename Method", "Rename Variable", "Rename Attribute"],
    'Rename Variable'                  : ["Extract Variable", "Rename Parameter", "Inline Variable", "Rename Method"],
    'Reorder Parameter'                : ["Remove Parameter", "Rename Parameter", "Add Parameter", "Move Method"],
    'Replace Attribute With Variable'  : ["Replace Variable With Attribute", "Rename Attribute", "Rename Method", "Rename Variable"],
    'Replace Variable With Attribute'  : ["Replace Attribute With Variable", "Remove Parameter", "Move And Rename Method", "Extract Method"],
    'Split Conditional'                : ["Extract Variable", "Add Parameter", "Rename Variable"],
    'Split Parameter'                  : ["Add Parameter"],
}


def _active_missing_hints() -> Dict[str, List[str]]:
    """Return a missing-type hint dict based on env vars.

    Resolution order (first match wins):
      1. ``LANGCHAIN_HINTS_VERSION`` ∈ {curated, v2, v3, v4}: explicit override.
      2. ``LANGCHAIN_USE_MINED_HINTS=1``: legacy alias for v2.
      3. Default: curated MISSING_HINTS.
    """
    import os
    version = os.environ.get("LANGCHAIN_HINTS_VERSION", "").lower()
    if version == "v4":
        return MISSING_HINTS_MINED_V4
    if version == "v3_gpt":
        return MISSING_HINTS_GPT4MINI
    if version == "v3":
        return MISSING_HINTS_MINED_V3
    if version == "v2":
        return MISSING_HINTS_MINED_V2
    if version == "curated":
        return MISSING_HINTS
    if os.environ.get("LANGCHAIN_USE_MINED_HINTS", "").lower() in ("1", "true", "yes"):
        return MISSING_HINTS_MINED_V2
    return MISSING_HINTS


def _active_missing_cap(default: int = 7) -> int:
    """Return cap from LANGCHAIN_MISSING_CAP env var if set, else default."""
    import os
    raw = os.environ.get("LANGCHAIN_MISSING_CAP")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return default


def adaptive_cap(detected_count: int, base: int = 7) -> int:
    """Return a cap that grows with the Stage 2 detection complexity.

    Rationale: A2 ablation showed cap=12 helps recall on complex multi-
    refactoring cases but hurts simple cases by introducing noise.
    Adaptive cap captures most of the recall gain at lower average token cost.
    """
    if detected_count == 0:
        return base
    if detected_count <= 2:
        return max(4, base - 2)
    if detected_count <= 5:
        return base
    if detected_count <= 10:
        return base + 3
    return base + 5


def _active_adaptive_cap_enabled() -> bool:
    """Return True if LANGCHAIN_ADAPTIVE_CAP is set, else False."""
    import os
    return os.environ.get("LANGCHAIN_ADAPTIVE_CAP", "").lower() in ("1", "true", "yes")


def _active_drop_always_check() -> bool:
    """Return True if LANGCHAIN_DROP_ALWAYS_CHECK is set, else False.

    A2 + iter 4 LLM-behavior analysis showed ALWAYS_CHECK_MISSING types
    convert to wrong answers 85% of the time (vs ~67% for hint-derived
    candidates). Dropping them is the strongest single change available
    for Stage 3 candidate quality.
    """
    import os
    return os.environ.get("LANGCHAIN_DROP_ALWAYS_CHECK", "").lower() in ("1", "true", "yes")


def _active_hints_hybrid() -> bool:
    """Return True if LANGCHAIN_HINTS_HYBRID is set.

    When True, build_missing_candidates unions the curated MISSING_HINTS
    with whichever mined dict is selected by LANGCHAIN_HINTS_VERSION.
    Candidate vote count is the sum of appearances across both dicts, so
    types in BOTH rank higher (highest-confidence subset).
    """
    import os
    return os.environ.get("LANGCHAIN_HINTS_HYBRID", "").lower() in ("1", "true", "yes")

_V4_WEIGHTS_CACHE = None
_V4_TABLE_CACHE = None


def _load_v4_artifacts():
    """Lazy-load candidate_score_weights.json + type_assoc_v4.json once."""
    global _V4_WEIGHTS_CACHE, _V4_TABLE_CACHE
    if _V4_WEIGHTS_CACHE is None:
        import json
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        weights_path = root / "experiments" / "output" / "candidate_score_weights.json"
        table_path = root / "experiments" / "output" / "type_assoc_v4.json"
        with open(weights_path) as f:
            _V4_WEIGHTS_CACHE = json.load(f)
        with open(table_path) as f:
            _V4_TABLE_CACHE = json.load(f)
    return _V4_WEIGHTS_CACHE, _V4_TABLE_CACHE

# v4 max rule strength
def _v4_max_rule_strength(detected_set, target, table):
    best = 0.0
    detected_l = sorted(detected_set)
    keys = list(detected_l)
    for i in range(len(detected_l)):
        for j in range(i + 1, len(detected_l)):
            keys.append("|".join(sorted([detected_l[i], detected_l[j]])))
    for k in keys:
        for r in table.get(k, []):
            if r["consequent"] == target and r["score"] > best:
                best = r["score"]
    return best


def _build_v4(detected_types, stage1_levels):
    """Score-formula candidate builder used when LANGCHAIN_HINTS_VERSION=v4.

    Scores every non-detected known type by:
        alpha * max_rule_strength(detected -> t)
      + beta  * P(t | active_levels)
      + gamma * P(t globally)

    Returns the top per-level-cap candidates by score (>0). Replaces the
    union-then-truncate logic + ALWAYS_CHECK_MISSING list.
    """
    import os
    weights, table = _load_v4_artifacts()
    detected = set(detected_types)
    levels = list(stage1_levels) if stage1_levels else list(LEVEL_TYPES.keys())
    cap_per_lvl = weights["cap_per_level"]
    cap = max((cap_per_lvl.get(lvl, 5) for lvl in levels), default=5)

    raw_cap = os.environ.get("LANGCHAIN_MISSING_CAP")
    if raw_cap:
        try:
            cap = max(1, int(raw_cap))
        except ValueError:
            pass
    a, b, g = weights["alpha"], weights["beta"], weights["gamma"]
    pri_lvl = weights["type_priors_per_level"]
    pri_glb = weights["type_priors_global"]
    scored = []
    for t in ALL_KNOWN_TYPES:
        if t in detected:
            continue
        rule_s = _v4_max_rule_strength(detected, t, table) if detected else 0.0
        p_lvl = max((pri_lvl.get(lvl, {}).get(t, 0.0) for lvl in levels), default=0.0)
        p_glb = pri_glb.get(t, 0.0)
        s = a * rule_s + b * p_lvl + g * p_glb
        if s > 0:
            scored.append((s, t))
    scored.sort(reverse=True)
    return [t for _, t in scored[:cap]]


def build_missing_candidates(
    detected_types: List[str],
    stage1_levels: Optional[List[str]] = None,
    cap: Optional[int] = None,
) -> List[str]:
    """Compute the candidate missing-type list for Stage 3 Call B.

    When LANGCHAIN_HINTS_VERSION=v4: dispatch to the calibrated score formula
    (``_build_v4``); the ``cap`` argument is ignored in favour of the per-level
    cap stored in candidate_score_weights.json.

    Otherwise (v1/v2/v3 legacy): union-then-truncate of the active hint dict.

    Args:
        detected_types: Types from Stage 2 (defined only).
        stage1_levels: Levels selected by Stage 1; used as fallback when
            ``detected_types`` is empty.
        cap: Maximum number of candidates. If ``None``, reads from
            ``LANGCHAIN_MISSING_CAP`` env var (default 7), then optionally
            scales by ``adaptive_cap(detected_count)`` if
            ``LANGCHAIN_ADAPTIVE_CAP=1``.

    Env-var resolution chain:
        - ``LANGCHAIN_HINTS_VERSION`` (curated/v2/v3/v4) selects the active dict.
        - ``LANGCHAIN_HINTS_HYBRID=1`` unions curated with the active dict.
        - ``LANGCHAIN_DROP_ALWAYS_CHECK=1`` drops the ALWAYS_CHECK_MISSING list.
        - ``LANGCHAIN_ADAPTIVE_CAP=1`` scales cap by Stage 2 detection count.

    Behavior:
        - **Normal case** (detected_types non-empty): union of
          ``hints[t]`` for each detected type, plus ``ALWAYS_CHECK_MISSING``
          (unless dropped), minus already-detected types, capped at ``cap``.
        - **Empty fallback** (detected_types empty): all defined types in
          ``stage1_levels``, no cap.
    """
    import os
    if os.environ.get("LANGCHAIN_HINTS_VERSION", "").lower() == "v4":
        return _build_v4(detected_types, stage1_levels)

    if cap is None:
        base_cap = _active_missing_cap(default=7)
        if _active_adaptive_cap_enabled():
            cap = adaptive_cap(len(detected_types), base=base_cap)
        else:
            cap = base_cap
    hints = _active_missing_hints()
    use_hybrid = _active_hints_hybrid()
    drop_always = _active_drop_always_check()
    detected_set = set(detected_types)

    if not detected_set:
        if not stage1_levels:
            return []
        out: List[str] = []
        seen: set = set()
        for lvl in stage1_levels:
            for t in LEVEL_TYPES.get(lvl, []):
                if t not in seen:
                    seen.add(t)
                    out.append(t)
        return out

    counts: Dict[str, int] = defaultdict(int)
    for t in detected_types:
        for cand in hints.get(t, []):
            if cand not in detected_set:
                counts[cand] += 1
        if use_hybrid and hints is not MISSING_HINTS:
            for cand in MISSING_HINTS.get(t, []):
                if cand not in detected_set:
                    counts[cand] += 1

    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    primary = [c for c, _ in ordered]

    if drop_always:
        return primary[:cap]

    backup = [
        c for c in ALWAYS_CHECK_MISSING
        if c not in detected_set and c not in primary
    ]

    return (primary + backup)[:cap]

# TERM DICTIONARIES

ACTION_TERMS: Dict[str, str] = {
    "Rename": "Change the name of an element (variable, method, class, etc.) without altering its behavior.",
    "Move": "Relocate an element from one scope, class, or module to another.",
    "Extract": "Pull out a piece of code into a new, named element (method, variable, class).",
    "Inline": "Replace a named element with its definition or body, eliminating the indirection.",
    "Pull Up": "Move an element from a subclass into its superclass (up the hierarchy).",
    "Push Down": "Move an element from a superclass into a subclass (down the hierarchy).",
    "Add": "Introduce a new element (parameter, annotation) that did not exist before.",
    "Remove": "Delete an existing element (parameter, annotation) from the code.",
    "Replace": "Substitute one kind of element for another (e.g., variable with attribute).",
    "Parameterize": "Turn a hard-coded value or fixed element into a configurable parameter.",
    "Localize": "Move a declaration closer to its point of use to reduce scope.",
    "Encapsulate": "Hide direct access to an element behind getter/setter methods.",
    "Split": "Divide a single element into two or more separate elements.",
    "Invert": "Negate or reverse a boolean condition and swap the corresponding branches.",
    "Reorder": "Change the position/order of elements (e.g., parameter order) without renaming.",
    "Change": "Modify the type or kind of an element (e.g., variable type change).",
}

TARGET_TERMS: Dict[str, str] = {
    "Method": "A function defined inside a class (def method_name(self, ...)).",
    "Variable": "A local name bound inside a function body (x = ..., for x in ...).",
    "Parameter": "A name declared in a function's def line (def f(param): ...).",
    "Attribute": "A name accessed via self.x or cls.x -- instance or class state.",
    "Class": "A class definition (class Foo: ...).",
    "Subclass": "A class that inherits from another class (class Child(Parent): ...).",
    "Superclass": "A parent class from which other classes inherit.",
    "Code": "A block of statements not necessarily tied to one method or class.",
    "Conditional": "An if/elif/else block or boolean expression.",
    "Annotation": "A decorator (@decorator) or type annotation on a method, class, or attribute.",
}

# COMPOUND TYPE NOTES

COMPOUND_TYPE_NOTES: Dict[str, str] = {
    "Move And Rename Method": (
        "This is a single atomic refactoring operation where a method is "
        "simultaneously moved to a different class/module AND renamed. "
        "Do NOT report it as separate Move Method + Rename Method."
    ),
    "Move And Rename Class": (
        "This is a single atomic refactoring operation where a class is "
        "simultaneously moved to a different module AND renamed. "
        "Do NOT report it as separate Move Class + Rename Class."
    ),
    "Extract And Move Method": (
        "This is a single atomic refactoring operation where code is "
        "extracted into a new method AND that method is placed in a different "
        "class/module. Do NOT report it as separate Extract Method + Move Method."
    ),
}

# PYTHON DISAMBIGUATION

PYTHON_DISAMBIGUATION: str = (
    "In Python, distinguishing Rename Variable / Rename Parameter / "
    "Rename Attribute / Rename Method / Rename Class requires checking WHERE "
    "the renamed identifier lives:\n"
    "  - Rename Variable: a local name inside a function body "
    "(e.g., 'x = 1' -> 'y = 1').\n"
    "  - Rename Parameter: a name in the function's def line "
    "(e.g., 'def f(old):' -> 'def f(new):').\n"
    "  - Rename Attribute: a name accessed via self.x or cls.x "
    "(e.g., 'self.old' -> 'self.new').\n"
    "  - Rename Method: the name after 'def' inside a class "
    "(e.g., 'def old_method(self):' -> 'def new_method(self):').\n"
    "  - Rename Class: the name after 'class' "
    "(e.g., 'class OldName:' -> 'class NewName:').\n"
    "Always check the syntactic position before choosing the type."
)

# LEVEL-SPECIFIC DEFINITIONS (for prompt injection)

# Loaded lazily from refactoring_defs.json
_TYPE_DEFINITIONS: Optional[Dict[str, str]] = None


def _load_type_definitions() -> Dict[str, str]:
    """Load type definitions from refactoring_defs.json (once)."""
    global _TYPE_DEFINITIONS
    if _TYPE_DEFINITIONS is not None:
        return _TYPE_DEFINITIONS
    import json, os
    defs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refactoring_defs.json")
    with open(defs_path, "r") as f:
        raw = json.load(f)
    _TYPE_DEFINITIONS = {entry["refactoring_type"]: entry["definition"] for entry in raw}
    return _TYPE_DEFINITIONS


def build_level_definitions_xml(level: str) -> str:
    """Build XML block with type definitions + term glossary for a specific level.

    Returns XML like:
        <reference>
          <type_definitions>
            <definition type="Add Parameter">A new parameter is added...</definition>
            ...
          </type_definitions>
          <term_glossary>
            <term name="Add">Introduce a new element...</term>
            ...
          </term_glossary>
        </reference>
    """
    type_names = LEVEL_TYPES.get(level, [])
    if not type_names:
        return ""

    defs = _load_type_definitions()

    def_lines = []
    for t in type_names:
        d = defs.get(t, "")
        if d:
            def_lines.append(f'    <definition type="{t}">{d}</definition>')

    relevant_actions = set()
    relevant_targets = set()
    for name in type_names:
        words = name.replace(" And ", " ").split()
        for w in words:
            for action in ("Pull Up", "Push Down"):
                if action.startswith(w) and action in name:
                    relevant_actions.add(action)
            if w in ACTION_TERMS:
                relevant_actions.add(w)
            if w in TARGET_TERMS:
                relevant_targets.add(w)

    term_lines = []
    for a in sorted(relevant_actions):
        term_lines.append(f'    <term name="{a}" role="action">{ACTION_TERMS[a]}</term>')
    for t in sorted(relevant_targets):
        term_lines.append(f'    <term name="{t}" role="target">{TARGET_TERMS[t]}</term>')

    compound_lines = []
    for name in type_names:
        if name in COMPOUND_TYPE_NOTES:
            compound_lines.append(f'    <compound type="{name}">{COMPOUND_TYPE_NOTES[name]}</compound>')

    parts = ["<reference>"]
    if def_lines:
        parts.append("  <type_definitions>")
        parts.extend(def_lines)
        parts.append("  </type_definitions>")
    if term_lines:
        parts.append("  <term_glossary>")
        parts.extend(term_lines)
        parts.append("  </term_glossary>")
    if compound_lines:
        parts.append("  <compound_types>")
        parts.extend(compound_lines)
        parts.append("  </compound_types>")
    parts.append("</reference>")
    return "\n".join(parts)

# HELPER FUNCTIONS
def build_confusion_hints_text(detected_types: List[str]) -> str:
    """Build targeted confusion hints for the given detected types.

    Also checks for commonly missed types (Extract Subclass, Extract
    Superclass, Invert Condition) and compound types when move/rename
    types are detected.
    """
    hints: List[str] = []
    for t in detected_types:
        if t in CONFUSION_HINTS:
            hints.append(f"* **{t}**: {CONFUSION_HINTS[t]}")

    compound_check = {
        "Move Method", "Rename Method", "Move Class", "Rename Class",
        "Extract Method", "Move Code",
    }
    if any(t in compound_check for t in detected_types):
        for compound in [
            "Move And Rename Method",
            "Move And Rename Class",
            "Extract And Move Method",
        ]:
            if compound not in detected_types and compound in CONFUSION_HINTS:
                hints.append(
                    f"* **{compound}** (not detected -- confirm it's not present): "
                    f"{CONFUSION_HINTS[compound]}"
                )

    for missed in ["Extract Subclass", "Extract Superclass", "Invert Condition"]:
        if missed not in detected_types:
            hints.append(
                f"* **{missed}** (not detected -- check if missed): "
                f"{CONFUSION_HINTS[missed]}"
            )

    return "\n".join(hints) if hints else "No specific confusion hints for these types."


def parse_detected_types(types_str: str) -> List[str]:
    """Parse a comma-separated detected-types string into a list.

    Strips whitespace and filters out ``"None"`` / ``"N/A"`` sentinels.
    """
    if not types_str:
        return []
    if types_str.strip().lower() in ("none", "n/a", ""):
        return []
    types: List[str] = []
    for t in types_str.split(","):
        t = t.strip()
        if t and t.lower() not in ("none", "n/a"):
            types.append(t)
    return types


def merge_types(type_lists: List[List[str]]) -> List[str]:
    """Deduplicate across multiple type lists by normalized name."""
    seen_normalized: set = set()
    merged: List[str] = []
    for type_list in type_lists:
        for t in type_list:
            norm = t.lower().replace(" ", "").replace("_", "")
            if norm not in seen_normalized:
                seen_normalized.add(norm)
                merged.append(t)
    return merged


def parse_levels(levels_str: str) -> List[str]:
    """Parse and validate a comma-separated levels string.

    Returns only recognised level names (``parameter_level``,
    ``method_level``, ``class_level``).
    """
    if not levels_str:
        return []
    valid = set(LEVEL_TYPES.keys())
    levels: List[str] = []
    for part in levels_str.split(","):
        level = part.strip().lower().replace(" ", "_")
        if level in valid:
            levels.append(level)
    return levels


def example_has_level(ground_truth_types: List[str], level: str) -> bool:
    """Return ``True`` if any ground-truth type belongs to *level*."""
    level_set = set(LEVEL_TYPES.get(level, []))
    return any(t in level_set for t in ground_truth_types)


def get_level_for_type(refactoring_type: str) -> Optional[str]:
    """Return the level name for a refactoring type, or ``None`` if unknown."""
    for level, types in LEVEL_TYPES.items():
        if refactoring_type in types:
            return level
    return None
