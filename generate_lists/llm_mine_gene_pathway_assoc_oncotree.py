import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, get_args, get_origin, Union, Optional, Type

import google.generativeai as genai
import typer
from dotenv import load_dotenv
from google.generativeai.types import GenerationConfig
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()
YOUR_API_KEY = os.getenv("LLM_API_KEY")
if not YOUR_API_KEY:
    raise RuntimeError("LLM_API_KEY is not set in environment (.env). Please set LLM_API_KEY and try again.")
genai.configure(api_key=YOUR_API_KEY)


class CodeReferences(BaseModel):
    NCIt: str
    UMLS: str


class AssociatedPathways(BaseModel):
    ar_signaling: Literal["yes", "no"]
    ar_and_steroid_synthesis_enzymes: Literal["yes", "no"]
    steroid_inactivating_genes: Literal["yes", "no"]
    down_regulated_by_androgen: Literal["yes", "no"]
    rtk_ras_pi3k_akt_signaling: Literal["yes", "no"]
    rb_pathway: Literal["yes", "no"]
    cell_cycle_pathway: Literal["yes", "no"]
    hippo_pathway: Literal["yes", "no"]
    myc_pathway: Literal["yes", "no"]
    notch_pathway: Literal["yes", "no"]
    nrf2_pathway: Literal["yes", "no"]
    pi3k_pathway: Literal["yes", "no"]
    rtk_ras_pathway: Literal["yes", "no"]
    tp53_pathway: Literal["yes", "no"]
    wnt_pathway: Literal["yes", "no"]
    cell_cycle_control: Literal["yes", "no"]
    p53_signaling: Literal["yes", "no"]
    notch_signaling: Literal["yes", "no"]
    dna_damage_response: Literal["yes", "no"]
    other_growth_proliferation_signaling: Literal["yes", "no"]
    survival_cell_death_regulation_signaling: Literal["yes", "no"]
    telomere_maintenance: Literal["yes", "no"]
    rtk_signaling_family: Literal["yes", "no"]
    pi3k_akt_mtor_signaling: Literal["yes", "no"]
    ras_raf_mek_erk_jnk_signaling: Literal["yes", "no"]
    angiogenesis: Literal["yes", "no"]
    folate_transport: Literal["yes", "no"]
    invasion_and_metastasis: Literal["yes", "no"]
    tgf_β_pathway: Literal["yes", "no"]
    oncogenes_associated_with_epithelial_ovarian_cancer: Literal["yes", "no"]
    regulation_of_ribosomal_protein_synthesis_and_cell_growth: Literal["yes", "no"]


class GeneInfo(BaseModel):
    association_strength: Literal[
        "very strong", "strong", "moderate", "weak", "very weak"
    ]
    reference: str
    mutations: List[str]
    mutation_origin: Literal["germline/somatic", "somatic"]
    diagnostic_implication: str
    therapeutic_relevance: str


class AssociatedGene(BaseModel):
    gene_symbol: str
    gene_info: GeneInfo


class GenerateLists(BaseModel):
    cancer_name: str
    other_codes_used_for_data_gathering: CodeReferences

    associated_genes: List[AssociatedGene] = Field(
        ..., description="List of gene symbols and their associated data"
    )
    molecular_subtypes: List[str]
    associated_pathways: AssociatedPathways

    model_config = ConfigDict(validate_by_name=True)


# -----------------------
# Schema generation (backend-agnostic)
# -----------------------
def _is_optional(annotation) -> bool:
    origin = get_origin(annotation)
    if origin is Union:
        return type(None) in get_args(annotation)
    return False


def _map_primitive(annotation) -> Dict[str, Any]:
    # Map common primitive annotations to JSON schema types
    if annotation is str:
        return {"type": "string"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    # Fallback
    return {"type": "string"}


def _literal_to_enum(annotation) -> Dict[str, Any]:
    try:
        args = get_args(annotation)
        enum_values = [a for a in args if a is not None]
        # Normalize to strings for schema compatibility
        return {"type": "string", "enum": [str(v) for v in enum_values]}
    except Exception:
        return {"type": "string"}


def generate_json_schema(model_cls: Type[BaseModel]) -> Dict[str, Any]:
    """
    Generate a generic JSON schema (simple form) from a Pydantic BaseModel class.
    Compatible with Pydantic v1/v2 internal field representations.
    This schema is intended for human/LLM guidance and local validation (via Pydantic).
    """
    schema: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    # Obtain fields mapping in a Pydantic v2/v1 compatible way
    fields = None
    # Pydantic v2: model_cls.model_fields (mapping name -> FieldInfo)
    if hasattr(model_cls, "model_fields"):
        fields = getattr(model_cls, "model_fields")
    # Pydantic v1: model_cls.__fields__ (mapping name -> ModelField)
    elif hasattr(model_cls, "__fields__"):
        fields = getattr(model_cls, "__fields__")
    else:
        fields = {}

    for field_name, field in fields.items():
        # Try to discover the annotation/type for the field from known places
        annotation = None
        # Pydantic v2 FieldInfo-like object has .annotation
        if hasattr(field, "annotation"):
            annotation = getattr(field, "annotation")
        # Pydantic v1 ModelField has .outer_type_ or .type_
        elif hasattr(field, "outer_type_"):
            annotation = getattr(field, "outer_type_")
        elif hasattr(field, "type_"):
            annotation = getattr(field, "type_")
        # Some representations may have dict-like entries
        elif isinstance(field, dict) and "annotation" in field:
            annotation = field["annotation"]

        # As last resort, try to read from model __annotations__
        if annotation is None:
            try:
                ann = getattr(model_cls, "__annotations__", {})
                annotation = ann.get(field_name, None)
            except Exception:
                annotation = None

        if annotation is None:
            schema["properties"][field_name] = {"type": "string"}
            # treat as required by default
            schema["required"].append(field_name)
            continue

        optional = _is_optional(annotation)
        origin = get_origin(annotation)

        # Lists / arrays
        if origin is list or origin is List:
            args = get_args(annotation)
            if args:
                item_type = args[0]
                # Nested BaseModel in list
                if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                    schema["properties"][field_name] = {
                        "type": "array",
                        "items": generate_json_schema(item_type),
                    }
                else:
                    # primitive list item
                    if get_origin(item_type) is Literal or getattr(item_type, "__origin__", None) is Literal:
                        schema["properties"][field_name] = {
                            "type": "array",
                            "items": _literal_to_enum(item_type),
                        }
                    else:
                        schema["properties"][field_name] = {
                            "type": "array",
                            "items": _map_primitive(item_type),
                        }
            else:
                schema["properties"][field_name] = {"type": "array", "items": {"type": "string"}}

        # Union / Optional (non-list)
        elif origin is Union:
            args = [a for a in get_args(annotation) if a is not type(None)]
            if len(args) == 1:
                # Single non-None type
                chosen = args[0]
                if isinstance(chosen, type) and issubclass(chosen, BaseModel):
                    schema["properties"][field_name] = generate_json_schema(chosen)
                elif get_origin(chosen) is list:
                    # list union (rare)
                    list_args = get_args(chosen)
                    if list_args:
                        item_type = list_args[0]
                        schema["properties"][field_name] = {
                            "type": "array",
                            "items": _map_primitive(item_type),
                        }
                    else:
                        schema["properties"][field_name] = {"type": "array", "items": {"type": "string"}}
                elif get_origin(chosen) is Literal or getattr(chosen, "__origin__", None) is Literal:
                    schema["properties"][field_name] = _literal_to_enum(chosen)
                else:
                    schema["properties"][field_name] = _map_primitive(chosen)
            else:
                # Multiple non-None types: fallback to string
                schema["properties"][field_name] = {"type": "string"}

        # Literal
        elif get_origin(annotation) is Literal or getattr(annotation, "__origin__", None) is Literal:
            schema["properties"][field_name] = _literal_to_enum(annotation)

        # Nested BaseModel
        elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
            schema["properties"][field_name] = generate_json_schema(annotation)

        # Primitive types
        else:
            schema["properties"][field_name] = _map_primitive(annotation)

        if not optional:
            schema["required"].append(field_name)

    return schema


# Auto-generate JSON schema from the Pydantic model
schema_json = generate_json_schema(GenerateLists)
print("Generated Schema:\n", json.dumps(schema_json, indent=2))


# -----------------------
# LLM client abstraction (GenAI adapter + LiteLLM stub)
# -----------------------
class LLMClient:
    def supports_schema(self) -> bool:
        """Whether this backend accepts a server-side schema parameter."""
        return False

    def generate(self, prompt: str, temperature: float, schema: Optional[Dict[str, Any]] = None) -> str:
        """Return raw model text."""
        raise NotImplementedError


class GenAIClient(LLMClient):
    def __init__(self, model_name: str):
        self.model_name = model_name
        # default generation_config without schema; caller may pass schema to generate()
        self.base_generation_config = GenerationConfig(temperature=0.0)

    def supports_schema(self) -> bool:
        return True

    def generate(self, prompt: str, temperature: float, schema: Optional[Dict[str, Any]] = None) -> str:
        # If schema provided, create a GenerationConfig that includes it
        if schema is not None:
            cfg = GenerationConfig(
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=schema,
            )
            model = genai.GenerativeModel(model_name=self.model_name, generation_config=cfg)
        else:
            cfg = GenerationConfig(temperature=temperature)
            model = genai.GenerativeModel(model_name=self.model_name, generation_config=cfg)

        response = model.generate_content(prompt)

        # Try common attributes to get text
        if hasattr(response, "text") and isinstance(response.text, str):
            return response.text
        if hasattr(response, "candidates") and response.candidates:
            cand = response.candidates[0]
            if hasattr(cand, "content"):
                return cand.content
            if hasattr(cand, "text"):
                return cand.text
        # fallback to string representation
        return str(response)


class LiteLLMClient(LLMClient):
    """
    Placeholder/stub for LiteLLM. Implement this with the actual LiteLLM SDK calls.
    For now, this raises NotImplementedError to avoid accidental usage without implementation.
    """
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key

    def supports_schema(self) -> bool:
        # Update this when LiteLLM backend supports server-side schema enforcement
        return False

    def generate(self, prompt: str, temperature: float, schema: Optional[Dict[str, Any]] = None) -> str:
        raise NotImplementedError("LiteLLM client not implemented. Implement LiteLLMClient.generate() using the chosen LiteLLM SDK.")


# -----------------------
# Helpers for robust JSON extraction from model text
# -----------------------
def _find_json_substring(text: str) -> Optional[str]:
    """
    Find and return the first top-level JSON object substring in text.
    Uses brace counting to find a balanced {...} substring.
    Returns None if not found/parsable.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                # quick validation
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    return None
    return None


def extract_json_from_response_text(text: str) -> str:
    """
    Given raw model text, try to return a JSON string.
    Raises ValueError if no JSON could be extracted/parsed.
    """
    # Try direct parse
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # Try to find a JSON substring using brace matching
    candidate = _find_json_substring(text)
    if candidate:
        return candidate

    raise ValueError("No valid JSON found in model response text.")


# -----------------------
# CLI and main flow
# -----------------------
PROMPT_TEMPLATE = """You are an expert in clinical cancer genetics, specifically in gene-disease and pathway-disease curations (for hereditary and sporadic cancers). Based on scientific literature in PubMed, current genetic testing practices in oncology clinics, gene-disease association curations in ClinGen, OMIM, GeneReviews, and similar expert or peer reviewed resoursces, and public tumor sequencing databases such as cBioPortal, and COSMIC, list the genes and pathways, mutations in which are associated with {cancer_name} ({oncotree_code}). Different ontologies have different terms/codes to depict the same cancer sub-type. {oncotree_code} is the OncoTree code that is the same as {ncit_code} (NCIt) and {umls_code} (UMLS). Use these codes to gather as much literature/data as possible to provide a comprehensive list of genes and pathways in JSON structured format. The associated gene list should be ranked by strength and likelihood of association such that the first gene in the list has the strongest association with the cancer type and the last gene in the list has the weakest association with the cancer type. The gene list should be of high quality, accurate, and should not exceed 50 in count. The JSON should have top-level keys: "oncotree_code", "cancer_name" (full name of the code), "other_codes_used_for_data_gathering" (dictionary with keys NCIt and UMLS), "associated_genes" (a list of dictionaries - one dictionary for every associated gene, having top level keys of 'gene_symbol' and 'gene_info'. 'gene_symbol' should be only 1 gene per key. 'gene_info' is a dictionary with keys and values formatted as follows: 1. 'association_strength', value: classified as 'very strong', 'strong', 'moderate', 'weak', or 'very weak' association of this particular gene and cancer type depending on the quality and quantity of resources used to associate the gene and cancer type, 2. 'reference', value: resource(s) used to infer the gene-cancer type association (if multiple citations, then separate instances by '|'), 3. 'mutations', value: list of types of mutations in the gene that is associated with the given cancer type (such as truncating, splice, missense gain of function, missense-loss of function, missense-neomorphic, missense-hypo-/hyper-morphic, deletion, duplication, fusion, copy number variant, structural variant, complex rearrangements, methylation, and so on relevant to the gene-cancer type association), 4. 'mutation_origin', value: MUST be either "germline/somatic" OR "somatic" where 'germline/somatic' indicates that the cancer mutation in this gene can be present in the germline as cancer predisposing or arise somatically over time (so includes both 'germline' and 'somatic' options in 1 category only), 'somatic' indicates that the cancer mutation in this gene is only of somatic origin and not seen in the germline, 5. 'diagnostic_implication', value: clinical implication of the gene as to whether it is used to diagnose the cancer type, for example, the gene KRAS is associated with PAAD: 'diagnostic: missense mutations in KRAS are associated with PAAD and used for diagnosis.' Limit to 1 sentence, 6. 'therapeutic_relevance', value: if gene mutation informs decision making for therapeutic strategy, for example, for the association of KRAS and PAAD, 'clinical trials such as NCT07020221 are actively testing inhibitors of the actionable missense mutation KRAS G12D which is frequent in PAAD. Effect on immunotherapy is ....'), "molecular_subtypes", values: This should be a list of expression-based, genomic, or histological molecular subtypes known to occur in {cancer_name}. These subtypes should be informative for clinical decision-making, such as guiding treatment selection or predicting prognosis. Please use descriptive names or standard nomenclature for the subtypes, and prioritize those with known clinical implications, and "associated_pathways" (a dictionary with keys being each pathway name in the list: ['ar_signaling', 'ar_and_steroid_synthesis_enzymes', 'steroid_inactivating_genes', 'down_regulated_by_androgen', 'rtk_ras_pi3k_akt_signaling', 'rb_pathway', 'cell_cycle_pathway', 'hippo_pathway', 'myc_pathway', 'notch_pathway', 'nrf2_pathway', 'pi3k_pathway', 'rtk_ras_pathway', 'tp53_pathway', 'wnt_pathway', 'cell_cycle_control', 'p53_signaling', 'notch_signaling', 'dna_damage_response', 'other_growth_proliferation_signaling', 'survival_cell_death_regulation_signaling', 'telomere_maintenance', 'rtk_signaling_family', 'pi3k_akt_mtor_signaling', 'ras_raf_mek_erk_jnk_signaling', 'angiogenesis', 'folate_transport', 'invasion_and_metastasis', 'tgf_β_pathway', 'oncogenes_associated_with_epithelial_ovarian_cancer', 'regulation_of_ribosomal_protein_synthesis_and_cell_growth'] and the value being 'yes' if associated with cancer sub-type or 'no' if pathway not associated with cancer sub-type)."""

app = typer.Typer()


@app.command()
def generate_lists(
    input_oncotree: Path = typer.Option(
        ..., "--input_oncotree_filepath", "-i", help="Path to the OncoTree JSON file"
    ),
    output_lists: Path = typer.Option(
        ..., "--output_filepath", "-o", help="Path and name for output JSON file"
    ),
    llm_model: str = typer.Option(
        "--model_name",
        "-model",
        help="enter the string name of the LLM model to be used",
    ),
    temperature: float = typer.Option(
        ...,
        "--input_LLM_temperature",
        "-temp",
        help="Temperature setting for LLM: value between 0 to 1",
    ),
):
    typer.echo(f"Input file path: {input_oncotree}")

    if not input_oncotree.exists():
        typer.echo(f"File not found: {input_oncotree}")
        raise typer.Exit(code=1)

    with input_oncotree.open("r") as f:
        oncotree = json.load(f)

    # Initialize the output dictionary
    oncotree_codes_info: Dict[str, Dict[str, Optional[str]]] = {}

    for item in oncotree:
        if item["code"] not in {"COAD", "NSCLC", "PAAD", "DSRCT", "BRCA", "MNM"}:
            continue
        code = item["code"]
        name = item["name"]
        umls = (
            item["externalReferences"]["UMLS"][0]
            if "UMLS" in item.get("externalReferences", {})
            else None
        )
        ncit = (
            item["externalReferences"]["NCI"][0]
            if "NCI" in item.get("externalReferences", {})
            else None
        )

        # Add the extracted information to the output dictionary
        oncotree_codes_info[code] = {"name": name, "NCIt": ncit, "UMLS": umls}

    # Choose LLM client: if model name suggests Gemini/Google, use GenAIClient; otherwise user must implement LiteLLMClient
    if llm_model and ("gemini" in llm_model.lower() or "google" in llm_model.lower()):
        client: LLMClient = GenAIClient(model_name=llm_model)
    else:
        # You can implement LiteLLMClient.generate() to support other backends.
        client = LiteLLMClient(model_name=llm_model, api_key=YOUR_API_KEY)

    all_results: Dict[str, Any] = {}  # A dictionary to store all the AI's answers

    for oncotree_code, details in oncotree_codes_info.items():
        # Fill in the placeholders in the prompt template
        current_prompt = PROMPT_TEMPLATE.format(
            cancer_name=details["name"],
            oncotree_code=oncotree_code,
            ncit_code=details["NCIt"],
            umls_code=details["UMLS"],
        )

        try:
            # Pass schema to backend only if supported; otherwise perform local validation
            schema_for_backend = schema_json if client.supports_schema() else None
            raw_text = client.generate(current_prompt, temperature=temperature, schema=schema_for_backend)

            # Extract JSON string robustly from raw text
            try:
                json_str = extract_json_from_response_text(raw_text)
            except ValueError as ve:
                raise RuntimeError(f"Failed to extract JSON from model response: {ve}\nRaw response: {raw_text[:1000]}")

            parsed_json_data_dict = json.loads(json_str)

            # Validate with Pydantic
            parsed_model = GenerateLists(**parsed_json_data_dict)

            # Store the structured data
            all_results[oncotree_code] = parsed_model.model_dump()

        except Exception as e:
            print(f"  Error processing {oncotree_code}: {e}")
            # Log additional response diagnostics if available
            # (Note: GenAI SDK's response object is handled inside GenAIClient)
            all_results[oncotree_code] = {"error": str(e), "details_provided": details}

    print(all_results)

    with open(output_lists, "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    app()
