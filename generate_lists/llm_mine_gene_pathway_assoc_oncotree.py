import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, get_args

import typer
from dotenv import load_dotenv
from litellm import completion
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

if "USE_LITELLM_PROXY" not in os.environ:
    YOUR_API_KEY = os.getenv("LLM_API_KEY")
    os.environ["GOOGLE_API_KEY"] = YOUR_API_KEY


class AssociatedPathways(BaseModel):
    cell_cycle_pathway: Literal["yes", "no"]
    hippo_pathway: Literal["yes", "no"]
    myc_pathway: Literal["yes", "no"]
    notch_pathway: Literal["yes", "no"]
    nrf2_pathway: Literal["yes", "no"]
    pi3k_pathway: Literal["yes", "no"]
    tgf_beta_pathway: Literal["yes", "no"]  # fixed unicode beta issue
    rtk_ras_pathway: Literal["yes", "no"]
    tp53_pathway: Literal["yes", "no"]
    wnt_pathway: Literal["yes", "no"]


class GeneInfo(BaseModel):
    association_strength: Literal[
        "very strong", "strong", "moderate", "weak", "very weak"
    ]
    reference: str
    mutations: List[str]
    mutation_origin: Literal["germline/somatic", "somatic", "germline"]
    diagnostic_implication: str
    therapeutic_relevance: str


class AssociatedGene(BaseModel):
    gene_symbol: str
    gene_info: GeneInfo


class GenerateGeneLists(BaseModel):
    cancer_name: str
    associated_genes: List[AssociatedGene] = Field(
        ..., description="List of gene symbols and their associated data"
    )
    model_config = ConfigDict(validate_by_name=True)


class GeneratePathwayLists(BaseModel):
    cancer_name: str
    associated_pathways: AssociatedPathways
    model_config = ConfigDict(validate_by_name=True)


class GenerateMolecularSubtypeLists(BaseModel):
    cancer_name: str
    molecular_subtypes: List[str]
    model_config = ConfigDict(validate_by_name=True)


def generate_json_schema(model: BaseModel) -> Dict[str, Any]:
    schema = {"type": "object", "properties": {}, "required": []}

    fields = model.model_fields if hasattr(model, "model_fields") else model.__fields__

    for field_name, field in fields.items():
        field_type = field.annotation

        if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
            list_arg = get_args(field_type)[0]
            if isinstance(list_arg, type) and issubclass(list_arg, BaseModel):
                nested_schema = generate_json_schema(list_arg)
                schema["properties"][field_name] = {
                    "type": "array",
                    "items": nested_schema,
                }
            else:
                schema["properties"][field_name] = {
                    "type": "array",
                    "items": {"type": "string"},
                }

        elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
            schema["properties"][field_name] = generate_json_schema(field_type)

        else:
            schema["properties"][field_name] = {"type": "string"}

    schema["required"] = list(fields.keys())
    return schema


schema_json_genes = generate_json_schema(GenerateGeneLists)
schema_json_pathways = generate_json_schema(GeneratePathwayLists)
schema_json_molecularsubtypes = generate_json_schema(GenerateMolecularSubtypeLists)


PROMPT_TEMPLATE_GENES = """You are an expert in clinical cancer genetics ...
Return **strict JSON** without trailing commas, unescaped quotes, or comments. Ensure it parses with `json.loads()`."""


PROMPT_TEMPLATE_PATHWAYS = """You are an expert in clinical cancer genetics ...
Return **strict JSON** without trailing commas, unescaped quotes, or comments. Ensure it parses with `json.loads()`."""


PROMPT_TEMPLATE_MOLECULARSUBTYPES = """You are an expert in clinical cancer genetics ...
Return **strict JSON** without trailing commas, unescaped quotes, or comments. Ensure it parses with `json.loads()`."""


def retry_with_backoff(func, max_retries=5, base_delay=1, jitter=True):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            wait_time = base_delay * (2**attempt)
            if jitter:
                wait_time += random.uniform(0, 1)
            typer.echo(
                f"ERROR: Attempt {attempt+1} failed: {e}. Retrying in {wait_time:.2f}s..."
            )
            time.sleep(wait_time)
    raise Exception(f"Failed after {max_retries} retries")


def call_llm_with_retry(model, messages, temperature):
    def api_call():
        return completion(
            model=model,
            messages=messages,
            temperature=temperature,
        )

    return retry_with_backoff(api_call, max_retries=5, base_delay=1)


def try_parse_json(output: str) -> dict:
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", output, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def repair_with_llm(broken_output: str, llm_model: str) -> dict:
    repair_prompt = f"""
    The following JSON is invalid or malformed. Please fix it and return only valid JSON:

    {broken_output}
    """
    response = call_llm_with_retry(
        model=llm_model,
        messages=[{"role": "user", "content": repair_prompt}],
        temperature=0,
    )
    fixed = response.choices[0].message.content
    return try_parse_json(fixed)


app = typer.Typer()


@app.command()
def generate_lists(
    input_oncotree: Path = typer.Option(
        ..., "--input_oncotree_filepath", "-i"
    ),
    output_lists: Path = typer.Option(
        ..., "--output_filepath", "-o"
    ),
    llm_model: str = typer.Option(
        "gpt-4o-mini",
        "--model_name",
        "-m",
    ),
    temperature: float = typer.Option(
        0.25,
        "--input_LLM_temperature",
        "-t",
    ),
    codes: List[str] = typer.Option(
        None,
        "--codes",
        "-c",
    ),
    all_codes: bool = typer.Option(
        False,
        "--all",
        "-a",
    ),
    genes_flag: bool = typer.Option(False, "--genes", "-g"),
    pathways_flag: bool = typer.Option(False, "--pathways", "-p"),
    molecular_flag: bool = typer.Option(False, "--molecular", "-ms"),
):
    if not any([genes_flag, pathways_flag, molecular_flag]):
        typer.echo(
            "ERROR: You must specify at least one of --genes, --pathways, or --molecular"
        )
        raise typer.Exit(code=1)

    if not input_oncotree.exists():
        typer.echo(f"INFO: File not found: {input_oncotree}")
        raise typer.Exit(code=1)

    with input_oncotree.open("r") as f:
        oncotree = json.load(f)

    if all_codes:
        target_codes = {item["code"] for item in oncotree}
    elif codes:
        target_codes = set(codes)
    else:
        target_codes = {"COAD", "NSCLC", "PAAD", "DSRCT", "BRCA", "MNM"}

    oncotree_codes_info = {}
    for item in oncotree:
        if item["code"] not in target_codes:
            continue
        code = item["code"]
        name = item["name"]
        umls = item.get("externalReferences", {}).get("UMLS", [None])[0]
        ncit = item.get("externalReferences", {}).get("NCI", [None])[0]
        oncotree_codes_info[code] = {"name": name, "NCIt": ncit, "UMLS": umls}

    if not oncotree_codes_info:
        typer.echo("ERROR: No matching OncoTree codes found.")
        raise typer.Exit(code=1)

    all_results = {}
    total = len(oncotree_codes_info)

    for idx, (oncotree_code, details) in enumerate(
        oncotree_codes_info.items(), start=1
    ):
        typer.echo(f"[{idx}/{total}] Processing {oncotree_code}...")

        if genes_flag:
            current_prompt = PROMPT_TEMPLATE_GENES.format(
                cancer_name=details["name"],
                oncotree_code=oncotree_code,
                ncit_code=details["NCIt"],
                umls_code=details["UMLS"],
            )

            try:
                response = call_llm_with_retry(
                    model=llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Return only strict JSON.",
                        },
                        {"role": "user", "content": current_prompt},
                    ],
                    temperature=temperature,
                )

                raw_output = response.choices[0].message.content

                try:
                    parsed_json = try_parse_json(raw_output)
                except Exception:
                    parsed_json = repair_with_llm(raw_output, llm_model)

                parsed_model = GenerateGeneLists(**parsed_json)

                all_results.setdefault("genes", {})[
                    oncotree_code
                ] = parsed_model.model_dump()

            except Exception as e:
                typer.echo(f"ERROR: {oncotree_code}: {e}")

        time.sleep(2)

    with open(output_lists, "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    app()
