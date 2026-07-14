"""Schema-constrained Galaxy Zoo judging for the Phase 1 caption audit."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Dict, Iterable, List, Literal, Mapping, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class GalaxyDecisionTree(BaseModel):
    """Released GalaxyBench response schema, reproduced field-for-field."""

    overall_shape: Literal["smooth", "featured-or-disk", "artifact"] = Field(
        ..., description="The overall galaxy shape classification"
    )
    roundness: Optional[
        Literal["round", "in-between", "cigar-shaped", "not-mentioned"]
    ] = Field(None, description="How rounded the galaxy is (only for smooth galaxies)")
    edge_on: Optional[
        Literal["edge-on-yes", "edge-on-no", "not-mentioned"]
    ] = Field(
        None,
        description="Whether the galaxy is viewed edge-on (only for featured galaxies)",
    )
    edge_on_bulge: Optional[
        Literal["boxy", "none", "rounded", "not-mentioned"]
    ] = Field(None, description="Shape of the bulge for edge-on galaxies")
    has_spiral_arms: Optional[
        Literal["has-spiral-arms-yes", "has-spiral-arms-no", "not-mentioned"]
    ] = Field(
        None,
        description=(
            "Whether the galaxy has visible spiral arms "
            "(only for non-edge-on featured galaxies)"
        ),
    )
    spiral_winding: Optional[
        Literal["tight", "medium", "loose", "not-mentioned"]
    ] = Field(
        None,
        description="How tightly wound the spiral arms are (only for galaxies with spiral arms)",
    )
    spiral_arm_count: Optional[
        Literal["1", "2", "3", "4", "more-than-4", "cant-tell", "not-mentioned"]
    ] = Field(
        None, description="Number of spiral arms (only for galaxies with spiral arms)"
    )
    bar: Optional[Literal["strong", "weak", "no", "not-mentioned"]] = Field(
        None,
        description="Strength of central bar feature (only for non-edge-on featured galaxies)",
    )
    bulge_size: Optional[
        Literal["dominant", "large", "moderate", "small", "none", "not-mentioned"]
    ] = Field(
        None,
        description="Size of the central bulge (only for non-edge-on featured galaxies)",
    )
    merging: Optional[
        Literal[
            "none",
            "minor-disturbance",
            "major-disturbance",
            "merger",
            "not-mentioned",
        ]
    ] = Field(
        None,
        description=(
            "Signs of disturbance, interaction, or merging "
            "(only for smooth and featured galaxies, NOT artifacts)"
        ),
    )


SCHEMA_JSON = json.dumps(GalaxyDecisionTree.model_json_schema(), separators=(",", ":"))
SCHEMA_SHA256 = hashlib.sha256(SCHEMA_JSON.encode("utf-8")).hexdigest()


# --- Branch-enforcing (nested) schema -------------------------------------
# The released schema above is flat: all fields live at one level and the
# decision-tree branching is enforced only by the prompt. That lets the judge
# mis-slot conditional answers -- e.g. emit bulge_size for an edge-on galaxy,
# or leave edge_on_bulge unfilled. The nested schema below makes the branch
# structural: each conditional answer exists ONLY inside the variant it belongs
# to, so an invalid slotting cannot be represented. Leaf category vocabularies
# are byte-for-byte the released ones, and build_nested_path reproduces the
# released node strings, so scores stay directly comparable to the flat run.

_ROUNDNESS = Literal["round", "in-between", "cigar-shaped", "not-mentioned"]
_EDGE_ON_BULGE = Literal["boxy", "none", "rounded", "not-mentioned"]
_SPIRAL_WINDING = Literal["tight", "medium", "loose", "not-mentioned"]
_SPIRAL_ARM_COUNT = Literal["1", "2", "3", "4", "more-than-4", "cant-tell", "not-mentioned"]
_BAR = Literal["strong", "weak", "no", "not-mentioned"]
_BULGE_SIZE = Literal["dominant", "large", "moderate", "small", "none", "not-mentioned"]
_MERGING = Literal["none", "minor-disturbance", "major-disturbance", "merger", "not-mentioned"]


class SpiralArmsYes(BaseModel):
    model_config = ConfigDict(extra="forbid")
    has_spiral_arms: Literal["has-spiral-arms-yes"]
    spiral_winding: _SPIRAL_WINDING
    spiral_arm_count: _SPIRAL_ARM_COUNT


class SpiralArmsNo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    has_spiral_arms: Literal["has-spiral-arms-no"]


class SpiralArmsNotMentioned(BaseModel):
    model_config = ConfigDict(extra="forbid")
    has_spiral_arms: Literal["not-mentioned"]


SpiralArms = Annotated[
    Union[SpiralArmsYes, SpiralArmsNo, SpiralArmsNotMentioned],
    Field(discriminator="has_spiral_arms"),
]


class EdgeOnYes(BaseModel):
    model_config = ConfigDict(extra="forbid")
    edge_on: Literal["edge-on-yes"]
    edge_on_bulge: _EDGE_ON_BULGE


class EdgeOnNo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    edge_on: Literal["edge-on-no"]
    spiral_arms: SpiralArms
    bar: _BAR
    bulge_size: _BULGE_SIZE


class EdgeOnNotMentioned(BaseModel):
    model_config = ConfigDict(extra="forbid")
    edge_on: Literal["not-mentioned"]


Orientation = Annotated[
    Union[EdgeOnYes, EdgeOnNo, EdgeOnNotMentioned],
    Field(discriminator="edge_on"),
]


class SmoothClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    overall_shape: Literal["smooth"]
    roundness: _ROUNDNESS
    merging: _MERGING


class FeaturedClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    overall_shape: Literal["featured-or-disk"]
    orientation: Orientation
    merging: _MERGING


class ArtifactClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    overall_shape: Literal["artifact"]


class NestedGalaxyDecisionTree(BaseModel):
    """Branch-enforcing counterpart of the released flat GalaxyDecisionTree."""

    model_config = ConfigDict(extra="forbid")
    classification: Annotated[
        Union[SmoothClassification, FeaturedClassification, ArtifactClassification],
        Field(discriminator="overall_shape"),
    ]


def _xgrammar_schema_json(model: type[BaseModel]) -> str:
    """JSON Schema for constrained decoding, rewritten for XGrammar.

    Pydantic serializes discriminated unions as ``oneOf`` plus a
    ``discriminator`` object. XGrammar's grammar compiler understands
    ``anyOf``/``$ref`` but not ``oneOf``/``discriminator``; because every branch
    carries a distinct ``overall_shape``/``edge_on``/``has_spiral_arms`` literal
    the alternatives stay mutually exclusive under ``anyOf``. Pydantic keeps the
    discriminator for validation; only the decoding grammar is rewritten.
    """
    schema = model.model_json_schema()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("discriminator", None)
            if "oneOf" in node:
                node["anyOf"] = node.pop("oneOf")
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    return json.dumps(schema, separators=(",", ":"))


NESTED_SCHEMA_JSON = _xgrammar_schema_json(NestedGalaxyDecisionTree)
NESTED_SCHEMA_SHA256 = hashlib.sha256(NESTED_SCHEMA_JSON.encode("utf-8")).hexdigest()


def model_vocab_size(model: Any, tokenizer: Any) -> int:
    """Resolve flat and multimodal/nested Transformers configurations."""
    for config in (model.config, getattr(model.config, "text_config", None)):
        value = getattr(config, "vocab_size", None)
        if value is not None:
            return int(value)
    return int(len(tokenizer))


@dataclass(frozen=True)
class MorphologyResult:
    object_id: str
    tree: Any
    judge_path: List[str]
    answers: Dict[str, str]
    raw_response: str
    schema_sha256: str = SCHEMA_SHA256

    def as_record(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "tree_json": self.tree.model_dump_json(exclude_none=False),
            "judge_path_json": json.dumps(self.judge_path),
            "answers_json": json.dumps(self.answers, sort_keys=True),
            "raw_response": self.raw_response,
            "schema_enforced": True,
            "schema_sha256": self.schema_sha256,
        }


def build_decision_tree_path(tree: GalaxyDecisionTree) -> List[str]:
    """Match the released GalaxyZooJudge path construction exactly."""
    path = [f"smooth-or-featured_{tree.overall_shape}"]
    if tree.overall_shape == "smooth":
        if tree.roundness:
            path.append(
                "how-rounded-not-mentioned"
                if tree.roundness == "not-mentioned"
                else f"how-rounded_{tree.roundness}"
            )
    elif tree.overall_shape == "featured-or-disk" and tree.edge_on:
        if tree.edge_on == "not-mentioned":
            path.append("disk-edge-on-not-mentioned")
        elif tree.edge_on == "edge-on-yes":
            path.append("disk-edge-on_yes")
            if tree.edge_on_bulge:
                path.append(
                    "edge-on-bulge-not-mentioned"
                    if tree.edge_on_bulge == "not-mentioned"
                    else f"edge-on-bulge_{tree.edge_on_bulge}"
                )
        elif tree.edge_on == "edge-on-no":
            path.append("disk-edge-on_no")
            if tree.has_spiral_arms:
                if tree.has_spiral_arms == "not-mentioned":
                    path.append("has-spiral-arms-not-mentioned")
                elif tree.has_spiral_arms == "has-spiral-arms-yes":
                    path.append("has-spiral-arms_yes")
                    if tree.spiral_winding:
                        path.append(
                            "spiral-winding-not-mentioned"
                            if tree.spiral_winding == "not-mentioned"
                            else f"spiral-winding_{tree.spiral_winding}"
                        )
                    if tree.spiral_arm_count:
                        path.append(
                            "spiral-arm-count-not-mentioned"
                            if tree.spiral_arm_count == "not-mentioned"
                            else f"spiral-arm-count_{tree.spiral_arm_count}"
                        )
                elif tree.has_spiral_arms == "has-spiral-arms-no":
                    path.append("has-spiral-arms_no")
            if tree.bar:
                path.append(
                    "bar-not-mentioned" if tree.bar == "not-mentioned" else f"bar_{tree.bar}"
                )
            if tree.bulge_size:
                path.append(
                    "bulge-size-not-mentioned"
                    if tree.bulge_size == "not-mentioned"
                    else f"bulge-size_{tree.bulge_size}"
                )
    if tree.overall_shape != "artifact" and tree.merging:
        path.append(
            "merging-not-mentioned"
            if tree.merging == "not-mentioned"
            else f"merging_{tree.merging}"
        )
    return path


def _answer(value: Optional[str]) -> str:
    return "not-stated" if value in {None, "not-mentioned"} else str(value)


def decision_tree_answers(tree: GalaxyDecisionTree) -> Dict[str, str]:
    """Normalize the released tree for the secondary per-question diagnostic."""
    answers = {
        "smooth-or-featured": tree.overall_shape,
        "how-rounded": "not-applicable",
        "disk-edge-on": "not-applicable",
        "edge-on-bulge": "not-applicable",
        "has-spiral-arms": "not-applicable",
        "spiral-winding": "not-applicable",
        "spiral-arm-count": "not-applicable",
        "bar": "not-applicable",
        "bulge-size": "not-applicable",
        "merging": "not-applicable",
    }
    if tree.overall_shape == "smooth":
        answers["how-rounded"] = _answer(tree.roundness)
        answers["merging"] = _answer(tree.merging)
    elif tree.overall_shape == "featured-or-disk":
        answers["merging"] = _answer(tree.merging)
        edge_on = _answer(tree.edge_on)
        answers["disk-edge-on"] = {
            "edge-on-yes": "yes",
            "edge-on-no": "no",
        }.get(edge_on, edge_on)
        if tree.edge_on == "edge-on-yes":
            answers["edge-on-bulge"] = _answer(tree.edge_on_bulge)
        elif tree.edge_on == "edge-on-no":
            spiral = _answer(tree.has_spiral_arms)
            answers["has-spiral-arms"] = {
                "has-spiral-arms-yes": "yes",
                "has-spiral-arms-no": "no",
            }.get(spiral, spiral)
            if tree.has_spiral_arms == "has-spiral-arms-yes":
                answers["spiral-winding"] = _answer(tree.spiral_winding)
                answers["spiral-arm-count"] = _answer(tree.spiral_arm_count)
            answers["bar"] = _answer(tree.bar)
            answers["bulge-size"] = _answer(tree.bulge_size)
    return answers


def parse_morphology_response(object_id: str, response: str) -> MorphologyResult:
    try:
        tree = GalaxyDecisionTree.model_validate_json(response.strip())
    except ValidationError as error:
        raise ValueError(f"Invalid schema-constrained response for {object_id}") from error
    return MorphologyResult(
        object_id=str(object_id),
        tree=tree,
        judge_path=build_decision_tree_path(tree),
        answers=decision_tree_answers(tree),
        raw_response=response,
    )


def build_nested_path(tree: NestedGalaxyDecisionTree) -> List[str]:
    """Reproduce the released node strings from the branch-enforcing schema.

    Emits the same node vocabulary as build_decision_tree_path; the only
    structural difference is that the nested schema always fills each branch's
    questions (adding a ``*-not-mentioned`` node instead of omitting it), which
    does not change the path-overlap score (denominator = human path length).
    """
    c = tree.classification
    path = [f"smooth-or-featured_{c.overall_shape}"]
    if isinstance(c, SmoothClassification):
        path.append(
            "how-rounded-not-mentioned"
            if c.roundness == "not-mentioned"
            else f"how-rounded_{c.roundness}"
        )
    elif isinstance(c, FeaturedClassification):
        orientation = c.orientation
        if isinstance(orientation, EdgeOnNotMentioned):
            path.append("disk-edge-on-not-mentioned")
        elif isinstance(orientation, EdgeOnYes):
            path.append("disk-edge-on_yes")
            path.append(
                "edge-on-bulge-not-mentioned"
                if orientation.edge_on_bulge == "not-mentioned"
                else f"edge-on-bulge_{orientation.edge_on_bulge}"
            )
        elif isinstance(orientation, EdgeOnNo):
            path.append("disk-edge-on_no")
            arms = orientation.spiral_arms
            if isinstance(arms, SpiralArmsNotMentioned):
                path.append("has-spiral-arms-not-mentioned")
            elif isinstance(arms, SpiralArmsYes):
                path.append("has-spiral-arms_yes")
                path.append(
                    "spiral-winding-not-mentioned"
                    if arms.spiral_winding == "not-mentioned"
                    else f"spiral-winding_{arms.spiral_winding}"
                )
                path.append(
                    "spiral-arm-count-not-mentioned"
                    if arms.spiral_arm_count == "not-mentioned"
                    else f"spiral-arm-count_{arms.spiral_arm_count}"
                )
            elif isinstance(arms, SpiralArmsNo):
                path.append("has-spiral-arms_no")
            path.append(
                "bar-not-mentioned" if orientation.bar == "not-mentioned" else f"bar_{orientation.bar}"
            )
            path.append(
                "bulge-size-not-mentioned"
                if orientation.bulge_size == "not-mentioned"
                else f"bulge-size_{orientation.bulge_size}"
            )
    if not isinstance(c, ArtifactClassification):
        path.append(
            "merging-not-mentioned"
            if c.merging == "not-mentioned"
            else f"merging_{c.merging}"
        )
    return path


def nested_decision_tree_answers(tree: NestedGalaxyDecisionTree) -> Dict[str, str]:
    """Per-question diagnostic answers, matching decision_tree_answers exactly."""
    answers = {
        question: "not-applicable"
        for question in (
            "smooth-or-featured",
            "how-rounded",
            "disk-edge-on",
            "edge-on-bulge",
            "has-spiral-arms",
            "spiral-winding",
            "spiral-arm-count",
            "bar",
            "bulge-size",
            "merging",
        )
    }
    c = tree.classification
    answers["smooth-or-featured"] = c.overall_shape
    if isinstance(c, SmoothClassification):
        answers["how-rounded"] = _answer(c.roundness)
        answers["merging"] = _answer(c.merging)
    elif isinstance(c, FeaturedClassification):
        answers["merging"] = _answer(c.merging)
        orientation = c.orientation
        if isinstance(orientation, EdgeOnNotMentioned):
            answers["disk-edge-on"] = "not-stated"
        elif isinstance(orientation, EdgeOnYes):
            answers["disk-edge-on"] = "yes"
            answers["edge-on-bulge"] = _answer(orientation.edge_on_bulge)
        elif isinstance(orientation, EdgeOnNo):
            answers["disk-edge-on"] = "no"
            arms = orientation.spiral_arms
            if isinstance(arms, SpiralArmsNotMentioned):
                answers["has-spiral-arms"] = "not-stated"
            elif isinstance(arms, SpiralArmsYes):
                answers["has-spiral-arms"] = "yes"
                answers["spiral-winding"] = _answer(arms.spiral_winding)
                answers["spiral-arm-count"] = _answer(arms.spiral_arm_count)
            elif isinstance(arms, SpiralArmsNo):
                answers["has-spiral-arms"] = "no"
            answers["bar"] = _answer(orientation.bar)
            answers["bulge-size"] = _answer(orientation.bulge_size)
    return answers


def parse_nested_morphology_response(object_id: str, response: str) -> MorphologyResult:
    try:
        tree = NestedGalaxyDecisionTree.model_validate_json(response.strip())
    except ValidationError as error:
        raise ValueError(f"Invalid schema-constrained response for {object_id}") from error
    return MorphologyResult(
        object_id=str(object_id),
        tree=tree,
        judge_path=build_nested_path(tree),
        answers=nested_decision_tree_answers(tree),
        raw_response=response,
        schema_sha256=NESTED_SCHEMA_SHA256,
    )


def resolve_schema(variant: str):
    """Return (schema_json, schema_sha256, parse_fn) for a Phase 1 schema variant."""
    if variant == "flat":
        return SCHEMA_JSON, SCHEMA_SHA256, parse_morphology_response
    if variant == "nested":
        return NESTED_SCHEMA_JSON, NESTED_SCHEMA_SHA256, parse_nested_morphology_response
    raise ValueError(f"Unknown schema variant: {variant!r}")


class GemmaMorphologyExtractor:
    """Gemma 4 judge with XGrammar-constrained decoding."""

    def __init__(
        self,
        model_path: str,
        prompt_template: str,
        dtype: str = "bfloat16",
        max_new_tokens: int = 1536,
        enable_thinking: bool = False,
        processor: Any = None,
        model: Any = None,
        schema_json: str = SCHEMA_JSON,
    ) -> None:
        if enable_thinking:
            raise ValueError("The primary Phase 1 judge must run with thinking disabled")
        import xgrammar as xgr
        from xgrammar.contrib.hf import LogitsProcessor

        if processor is None or model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            dtype_value = getattr(torch, dtype)
            processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=dtype_value,
                device_map="auto",
                local_files_only=True,
            ).eval()
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("Gemma processor does not expose its tokenizer")
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=model_vocab_size(model, tokenizer)
        )
        compiler = xgr.GrammarCompiler(tokenizer_info)
        self.compiled_grammar = compiler.compile_json_schema(schema_json)
        self.schema_sha256 = hashlib.sha256(schema_json.encode("utf-8")).hexdigest()
        self._logits_processor_class = LogitsProcessor
        self._xgrammar_version = getattr(xgr, "__version__", "unknown")
        self.processor = processor
        self.model = model
        # The released triple-quoted prompt has no trailing newline; text files do.
        self.prompt_template = prompt_template.rstrip("\r\n")
        self.max_new_tokens = int(max_new_tokens)
        self.enable_thinking = bool(enable_thinking)

    def generate_response(self, description: str) -> str:
        response, _ = self.generate_response_with_metadata(description)
        return response

    def generate_response_with_metadata(
        self, description: str
    ) -> tuple[str, Dict[str, Any]]:
        import torch

        prompt = self.prompt_template.replace("{description}", description)
        messages = [{"role": "user", "content": prompt}]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[-1]
        logits_processor = self._logits_processor_class(self.compiled_grammar)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                logits_processor=[logits_processor],
            )
        decoded = self.processor.decode(outputs[0][input_length:], skip_special_tokens=False)
        used_parse_response = hasattr(self.processor, "parse_response")
        parsed_type = None
        if used_parse_response:
            parsed = self.processor.parse_response(decoded)
            parsed_type = type(parsed).__name__
            response = _response_text(parsed)
        else:
            response = self.processor.decode(
                outputs[0][input_length:], skip_special_tokens=True
            )
        tokenizer = getattr(self.processor, "tokenizer", None)
        response = _strip_trailing_special_tokens(
            response, getattr(tokenizer, "all_special_tokens", ()) or ()
        )
        return response.strip(), {
            "decoded_with_special_tokens": decoded,
            "used_processor_parse_response": used_parse_response,
            "processor_parse_response_type": parsed_type,
            "schema_enforced": True,
            "schema_sha256": self.schema_sha256,
            "structured_output_engine": "xgrammar",
            "xgrammar_version": self._xgrammar_version,
        }


def _strip_trailing_special_tokens(text: str, special_tokens: Iterable[str]) -> str:
    """Remove trailing tokenizer special tokens (e.g. Gemma's ``<eos>``).

    transformers 5.12's ``processor.parse_response`` keeps the terminal special
    token in the returned content, which then breaks strict JSON schema parsing.
    The ``skip_special_tokens=True`` decode path never carries them, so this is a
    no-op there. Only exact trailing special-token strings are removed -- never
    other characters -- so it does not repair otherwise-malformed JSON.
    """
    specials = [token for token in special_tokens if token]
    changed = True
    while changed:
        changed = False
        text = text.rstrip()
        for token in specials:
            if text.endswith(token):
                text = text[: -len(token)]
                changed = True
                break
    return text


def _response_text(parsed: Any) -> str:
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, Mapping):
        for key in ("content", "text", "response"):
            if isinstance(parsed.get(key), str):
                return str(parsed[key])
    if isinstance(parsed, list):
        for item in reversed(parsed):
            if isinstance(item, Mapping):
                for key in ("content", "text", "response"):
                    if isinstance(item.get(key), str):
                        return str(item[key])
            elif isinstance(item, str):
                return item
    for attribute in ("content", "text", "response"):
        value = getattr(parsed, attribute, None)
        if isinstance(value, str):
            return value
    raise ValueError(f"Unsupported processor.parse_response result: {type(parsed).__name__}")


def append_morphology_results(
    extractor: Any,
    caption_rows: Iterable[Mapping[str, Any]],
    output_jsonl: Path,
    error_jsonl: Optional[Path] = None,
    max_error_rate: float = 0.0,
    parse_fn: Any = parse_morphology_response,
) -> Dict[str, Any]:
    if not 0.0 <= max_error_rate < 1.0:
        raise ValueError("max_error_rate must be in [0, 1)")
    rows = list(caption_rows)
    ids = [str(row["object_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Morphology input contains duplicate object_id values")
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    completed = _read_object_ids(output_jsonl)
    failed = _read_object_ids(Path(error_jsonl)) if error_jsonl is not None else set()
    if completed & failed:
        raise ValueError("Morphology outputs and errors overlap")
    unexpected = (completed | failed) - set(ids)
    if unexpected:
        raise ValueError(f"Unexpected morphology object_id={sorted(unexpected)[0]}")
    error_budget = math.floor(len(rows) * max_error_rate)
    if len(failed) > error_budget:
        raise RuntimeError("Existing morphology errors exceed the configured cap")

    error_handle = None
    if error_jsonl is not None:
        error_path = Path(error_jsonl)
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_handle = error_path.open("a", encoding="utf-8")
    try:
        with output_jsonl.open("a", encoding="utf-8") as output_handle:
            for row in rows:
                object_id = str(row["object_id"])
                if object_id in completed or object_id in failed:
                    continue
                description = str(row["description"])
                raw_response: Optional[str] = None
                try:
                    raw_response = extractor.generate_response(description)
                    result = parse_fn(object_id, raw_response)
                except Exception as error:
                    if error_handle is None:
                        raise
                    error_handle.write(
                        json.dumps(
                            {
                                "object_id": object_id,
                                "stage": "parse" if raw_response is not None else "generation",
                                "error_type": type(error).__name__,
                                "error": str(error),
                                "raw_response": raw_response,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    error_handle.flush()
                    failed.add(object_id)
                    if len(failed) > error_budget:
                        raise RuntimeError(
                            f"Morphology error cap exceeded: {len(failed)}/{len(rows)}"
                        ) from error
                    continue
                output_handle.write(json.dumps(result.as_record(), sort_keys=True) + "\n")
                output_handle.flush()
                completed.add(object_id)
    finally:
        if error_handle is not None:
            error_handle.close()

    attempted = len(completed) + len(failed)
    return {
        "input_rows": len(rows),
        "completed_rows": len(completed),
        "error_rows": len(failed),
        "attempted_rows": attempted,
        "pending_rows": len(rows) - attempted,
        "error_rate": 0.0 if attempted == 0 else len(failed) / attempted,
        "max_error_rate": max_error_rate,
    }


def _read_object_ids(path: Path) -> set[str]:
    if not Path(path).exists():
        return set()
    return {
        str(json.loads(line)["object_id"])
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _append_jsonl(path: Optional[Path], record: Mapping[str, Any]) -> None:
    if path is None:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
        handle.flush()


def calibration_metrics(
    extractor: Any,
    calibration_path: Path,
    response_jsonl: Optional[Path] = None,
    error_jsonl: Optional[Path] = None,
    continue_on_error: bool = False,
    parse_fn: Any = parse_morphology_response,
) -> Dict[str, Any]:
    records = [
        json.loads(line)
        for line in Path(calibration_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("Judge calibration set is empty")
    answer_total = 0
    answer_correct = 0
    row_exact = 0
    row_records = []
    parse_errors = 0
    for record in records:
        object_id = str(record["object_id"])
        description = str(record["description"])
        expected = {str(key): str(value) for key, value in record["expected"].items()}
        answer_total += len(expected)
        raw_response = None
        generation_metadata: Dict[str, Any] = {}
        try:
            if hasattr(extractor, "generate_response_with_metadata"):
                raw_response, generation_metadata = extractor.generate_response_with_metadata(
                    description
                )
            else:
                raw_response = extractor.generate_response(description)
            _append_jsonl(
                response_jsonl,
                {
                    "object_id": object_id,
                    "description": description,
                    "raw_response": raw_response,
                    "generation_metadata": generation_metadata,
                },
            )
            result = parse_fn(object_id, raw_response)
        except Exception as error:
            parse_errors += 1
            _append_jsonl(
                error_jsonl,
                {
                    "object_id": object_id,
                    "description": description,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "raw_response": raw_response,
                    "generation_metadata": generation_metadata,
                },
            )
            row_records.append(
                {
                    "object_id": object_id,
                    "parse_valid": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            if continue_on_error:
                continue
            raise
        correctness = {
            key: result.answers.get(key) == value for key, value in expected.items()
        }
        answer_correct += sum(correctness.values())
        exact = all(correctness.values())
        row_exact += int(exact)
        row_records.append(
            {
                "object_id": object_id,
                "parse_valid": True,
                "exact": exact,
                "correctness": correctness,
                "answers": result.answers,
                "judge_path": result.judge_path,
            }
        )
    return {
        "rows": len(records),
        "row_exact_accuracy": row_exact / len(records),
        "answers": answer_total,
        "answer_accuracy": 0.0 if answer_total == 0 else answer_correct / answer_total,
        "parse_errors": parse_errors,
        "schema_enforced": True,
        "schema_sha256": SCHEMA_SHA256,
        "records": row_records,
    }
