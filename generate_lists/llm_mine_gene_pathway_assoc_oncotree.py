@app.command()
def generate_lists(
    input_oncotree: Path = typer.Option(
        ..., "--input_oncotree_filepath", "-i", help="Path to the OncoTree JSON file"
    ),
    output_lists: Path = typer.Option(
        ..., "--output_filepath", "-o", help="Path and name for output JSON file"
    ),
    llm_model: str = typer.Option(
        ..., "--model_name", "-model",
        help="Enter the string name of the LLM model to be used",
    ),
    temperature: float = typer.Option(
        ..., "--input_LLM_temperature", "-temp",
        help="Temperature setting for LLM: value between 0 to 1",
    ),
    oncotree_codes: List[str] = typer.Option(
        ..., "--oncotree_codes", "-codes",
        help="Enter OncoTree codes separated by space (Example: COAD NSCLC BRCA)"
    ),
):

    typer.echo(f"Input file path: {input_oncotree}")
    typer.echo(f"Selected OncoTree Codes: {oncotree_codes}")

    if not input_oncotree.exists():
        typer.echo(f"File not found: {input_oncotree}")
        raise typer.Exit(code=1)

    with input_oncotree.open("r") as f:
        oncotree = json.load(f)

    # Initialize output dictionary
    oncotree_codes_info = {}

    for item in oncotree:

        # ✅ USER INPUT FILTER (Replaced Hardcoding)
        if item["code"] not in set(oncotree_codes):
            continue

        code = item["code"]
        name = item["name"]

        umls = (
            item["externalReferences"]["UMLS"][0]
            if "UMLS" in item["externalReferences"]
            else None
        )

        ncit = (
            item["externalReferences"]["NCI"][0]
            if "NCI" in item["externalReferences"]
            else None
        )

        oncotree_codes_info[code] = {
            "name": name,
            "NCIt": ncit,
            "UMLS": umls
        }

    generation_config = GenerationConfig(
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=schema_json,
    )

    model = genai.GenerativeModel(
        model_name=llm_model,
        generation_config=generation_config,
    )

    all_results = {}

    for oncotree_code, details in oncotree_codes_info.items():

        current_prompt = PROMPT_TEMPLATE.format(
            cancer_name=details["name"],
            oncotree_code=oncotree_code,
            ncit_code=details["NCIt"],
            umls_code=details["UMLS"],
        )

        try:
            response = model.generate_content(current_prompt)

            parsed_json_data_dict = json.loads(response.text)
            parsed_model = GenerateLists(**parsed_json_data_dict)

            all_results[oncotree_code] = parsed_model.model_dump()

        except Exception as e:
            print(f"Error processing {oncotree_code}: {e}")

            if "response" in locals() and hasattr(response, "prompt_feedback"):
                print(f"Prompt Feedback: {response.prompt_feedback}")

            if (
                "response" in locals()
                and hasattr(response, "candidates")
                and response.candidates
            ):
                print(
                    f"Finish Reason: {response.candidates[0].finish_reason}"
                )

            all_results[oncotree_code] = {
                "error": str(e),
                "details_provided": details
            }

    print(all_results)

    with open(output_lists, "w") as f:
        json.dump(all_results, f, indent=2)
