"""STTM generation agent — unified autonomous ReAct version.

A single autonomous STTM agent holds four tools:
  - inspect_context_tool       : previews source data context for any layer
  - generate_bronze_sttm_tool  : generates Bronze ingestion rules
  - generate_silver_sttm_tool  : generates Silver cleansing rules
  - generate_gold_sttm_tool    : generates Gold materialisation rules

The orchestrator sends a goal stating which STTM to generate. The agent
inspects the relevant context, decides which generation tool matches the
request, executes it, and returns the saved STTM CSV path.

Business intent is consumed ONLY by Gold STTM generation. Bronze and Silver
are intent-agnostic — Bronze maps every source column as-is, Silver applies
standard cleansing rules to every Bronze column.

I/O contract:
    generate_bronze_sttm(profile_path, run_id, task_description) -> str
    generate_silver_sttm(bronze_output_paths, bronze_sttm_path, run_id, task_description) -> str
    generate_gold_sttm(silver_output_paths, silver_sttm_path, business_intent, run_id, task_description) -> str
"""

import json
import os
import pandas as pd
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from core.config import STTM_DIR, LLM_PROVIDER, ANTHROPIC_API_KEY, ANTHROPIC_MODEL, GROQ_API_KEY, GROQ_MODEL, GOOGLE_API_KEY, GEMINI_MODEL, GITHUB_TOKEN, GITHUB_MODEL, GITHUB_BASE_URL
from core.audit import AuditLogger
from core.observability import AgentTrace


# ---------------------------------------------------------------------------
# Unified autonomous STTM agent prompt
# ---------------------------------------------------------------------------

STTM_AGENT_PROMPT = """You are an autonomous Data Engineering Architect specialising in Source-to-Target
Mapping (STTM) generation for Medallion data pipelines. You operate independently:
you receive a goal from the orchestrator, inspect the relevant data context, decide
which STTM generation tool is appropriate, execute it, and verify the output.

## Your operating mode — follow this EXACT sequence every time

1. THINK: Read the task carefully. Determine:
   - Which layer's STTM is being requested: Bronze, Silver, or Gold?
   - What source data context is available to inspect?

2. INSPECT: Call inspect_context_tool FIRST to preview the source data — profile JSON
   for Bronze, Parquet column metadata for Silver/Gold. State your observations about
   the columns, types, and mappings you will need to generate.

3. PLAN: Based on the inspection, write your explicit plan:
   - Which generation tool will you call (bronze / silver / gold) and why?
   - Roughly how many STTM rows will you generate?
   - What transformation patterns will you apply (direct pass-through, type casting,
     null handling, joins, aggregations)?

4. ACT: Call EXACTLY ONE generation tool matching the requested layer:
   - **generate_bronze_sttm_tool** → Bronze ingestion rules from raw CSV profile.
   - **generate_silver_sttm_tool** → Silver cleansing rules from Bronze Parquet outputs.
   - **generate_gold_sttm_tool**   → Gold materialisation rules from Silver Parquet outputs.

5. VERIFY: Confirm the STTM CSV was saved. Report the output path and row count.

## Available tools

- **inspect_context_tool**: Previews the source data relevant to the requested layer.
  For Bronze: summarises the data profile JSON (column names, types, stats, semantics).
  For Silver: summarises Bronze Parquet column metadata and approved columns.
  For Gold: summarises Silver Parquet column metadata and approved columns.
  Always call this FIRST.

- **generate_bronze_sttm_tool**: Generates a complete Bronze STTM CSV covering every
  raw source column. This layer should include all columns from the source file adn loaded as-is to create bronze layer. It is a direct data dump of all the columns from all the source files provided in the bronze layer. Includes _load_timestamp and _source_file metadata rows.
  Returns JSON: {"sttm_path": "...", "row_count": N}.

- **generate_silver_sttm_tool**: Generates a complete Silver STTM CSV with null
  handling, deduplication, type casting, date standardisation, and surrogate key.
  Returns JSON: {"sttm_path": "...", "row_count": N}.

- **generate_gold_sttm_tool**: Generates a complete Gold STTM CSV with join rules,
  renames, aggregations, and surrogate key for analytics-ready tables.
  Returns JSON: {"sttm_path": "...", "row_count": N}.

## STTM rules by layer — apply these exactly

### Bronze
- "Direct" for pass-through columns, "Indirect" for renamed/type-validated columns.
- Add metadata rows: _load_timestamp ("Current UTC timestamp injected at load time")
  and _source_file ("Source file path injected at load time").
- Do NOT add a surrogate key — that belongs in Silver.
- Each row: source_schema, source_table, source_column, target_schema, target_table,
  target_column, transformation_type, transformation_logic. All STTM files (Bronze, Silver,
  Gold) use this same 8-column structure.

### Silver
- Same 8-column structure as Bronze: source_schema, source_table, source_column,
  target_schema, target_table, target_column, transformation_type, transformation_logic.
- FIRST row must be the surrogate key: source_schema="raw", source_table="<source_table>",
  source_column="" (empty), target_schema="cleansed", target_table="<source_table>",
  target_column="pk_<table_stem>_silver_id", transformation_type="Indirect",
  transformation_logic="Auto-generated sequential surrogate primary key starting from 1".
- Apply null handling (drop/fill mean/median/mode/constant), deduplication, type
  casting, date standardisation to YYYY-MM-DD, text normalisation (strip, lower, etc.).
- For id columns: type casting ONLY — no null handling.

### Gold
- Same 8-column structure as Bronze and Silver.
- FIRST row must be the surrogate key: source_schema="cleansed", source_table="",
  source_column="" (empty), target_schema="analytics", target_table="<gold_table_name>",
  target_column="pk_gold_id", transformation_type="Indirect",
  transformation_logic="Auto-generated sequential surrogate primary key starting from 1".
- Join Silver tables on matching key columns where applicable.
- Use "Direct" / "Passthrough" for columns needing no transformation.
- Build queryable tables — do NOT pre-aggregate for the business question.

## Important
- Return ONLY one generation tool call per task — Bronze, Silver, OR Gold.
- Do not call more than one generation tool in a single task.
- Output format from generation tools is JSON; read it for confirmation.
- Bronze and Silver are intent-agnostic: do NOT filter, prioritise, or shape
  rules based on any business question. Map every column mechanically.
- Gold is intent-driven: shape target tables to serve the business intent."""


# ---------------------------------------------------------------------------
# Pure Python context helpers — no LLM
# ---------------------------------------------------------------------------

def _prepare_bronze_context(profile_path: str) -> dict:
    """Read the dataset profile JSON produced by the profiler."""
    with open(profile_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _prepare_silver_context(bronze_output_paths: list[str], bronze_sttm_path: str) -> list[dict]:
    """Load Bronze Parquet metadata filtered to STTM-approved columns."""
    sttm_df = pd.read_csv(bronze_sttm_path).fillna("")
    if "target_column" not in sttm_df.columns:
        raise ValueError(
            f"Invalid STTM file '{bronze_sttm_path}': missing 'target_column' column. "
            f"Expected columns: source_schema, source_table, source_column, target_schema, "
            f"target_table, target_column, transformation_type, transformation_logic. "
            f"Found columns: {list(sttm_df.columns)}"
        )
    approved_cols = set(sttm_df["target_column"].unique())
    result = []
    for bp in bronze_output_paths:
        df = pd.read_parquet(bp)
        kept = [c for c in df.columns if c in approved_cols or c.startswith("_")]
        df = df[kept] if kept else df.iloc[:, :0]
        result.append({
            "filename": os.path.basename(bp),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "sample": df.head(5).to_dict(orient="records"),
        })
    return result


def _prepare_gold_context(silver_output_paths: list[str], silver_sttm_path: str) -> list[dict]:
    """Load Silver Parquet metadata filtered to STTM-approved columns."""
    sttm_df = pd.read_csv(silver_sttm_path).fillna("")
    if "target_column" not in sttm_df.columns:
        raise ValueError(
            f"Invalid STTM file '{silver_sttm_path}': missing 'target_column' column. "
            f"Expected columns: source_schema, source_table, source_column, target_schema, "
            f"target_table, target_column, transformation_type, transformation_logic. "
            f"Found columns: {list(sttm_df.columns)}"
        )
    approved_cols = set(sttm_df["target_column"].unique())
    result = []
    for sp in silver_output_paths:
        df = pd.read_parquet(sp)
        kept = [c for c in df.columns if c in approved_cols or c.startswith("_")]
        df = df[kept] if kept else df.iloc[:, :0]
        result.append({
            "filename": os.path.basename(sp),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "sample": df.head(5).to_dict(orient="records"),
        })
    return result


def _extract_sttm_rows(result: dict) -> list[dict]:
    """Scan agent message history (reverse order) for a JSON array of STTM rows."""
    for msg in reversed(result.get("messages", [])):
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            continue
        text = content
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        text = text.strip()
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1:
            continue
        try:
            rows = json.loads(text[start: end + 1])
            if isinstance(rows, list) and rows:
                return rows
        except (json.JSONDecodeError, ValueError):
            continue
    return []


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _make_llm():
    import os
    provider = os.environ.get("LLM_PROVIDER", LLM_PROVIDER).strip().lower()
    if provider == "claude":
        from langchain_anthropic import ChatAnthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)
        model = os.environ.get("ANTHROPIC_MODEL", ANTHROPIC_MODEL)
        print(f"[LLM] sttm | model={model} | api_key_set={bool(api_key)}")
        return ChatAnthropic(api_key=api_key, model=model)
    if provider == "github":
        import httpx
        from langchain_openai import ChatOpenAI
        token = os.environ.get("GITHUB_TOKEN", GITHUB_TOKEN)
        model = os.environ.get("GITHUB_MODEL", GITHUB_MODEL)
        print(f"[LLM] sttm | model={model} | token_set={bool(token)}")
        return ChatOpenAI(
            api_key=token,
            model=model,
            base_url="https://models.inference.ai.azure.com",
            http_client=httpx.Client(),
            http_async_client=httpx.AsyncClient(),
        )
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model=GEMINI_MODEL)


# ---------------------------------------------------------------------------
# Unified tool factory — all 4 tools built from the caller's context
# ---------------------------------------------------------------------------

def _make_sttm_tools(
    profile_path: str | None,
    bronze_output_paths: list[str] | None,
    bronze_sttm_path: str | None,
    silver_output_paths: list[str] | None,
    silver_sttm_path: str | None,
    business_intent: str | None,
    run_id: str,
    scratchpad: dict,
):
    """Build all four STTM tools bound to the caller's context via closure.

    Only the context relevant to the requested layer will be populated; the
    others will be None and the agent should not call those generation tools.
    """

    @tool
    def inspect_context_tool(confirmation: str = "execute") -> str:
        """Preview the source data context for the STTM layer being generated.

        For Bronze: summarises the data profile JSON (columns, types, stats, analysis).
        For Silver: summarises Bronze Parquet column metadata (approved columns only).
        For Gold: summarises Silver Parquet column metadata (approved columns only).
        Call this FIRST to understand what you will be mapping.
        Returns a JSON summary of the available source data context.
        """
        if profile_path:
            context = _prepare_bronze_context(profile_path)
            return json.dumps({"layer": "bronze", "profile": context}, default=str)
        if bronze_output_paths and bronze_sttm_path:
            context = _prepare_silver_context(bronze_output_paths, bronze_sttm_path)
            return json.dumps({"layer": "silver", "bronze_tables": context}, default=str)
        if silver_output_paths and silver_sttm_path:
            context = _prepare_gold_context(silver_output_paths, silver_sttm_path)
            return json.dumps({"layer": "gold", "silver_tables": context}, default=str)
        return json.dumps({"error": "No source context available"})

    @tool
    def generate_bronze_sttm_tool(confirmation: str = "execute") -> str:
        """Generate a complete Bronze STTM CSV from the raw data profile.

        Covers every source column with ingestion rules (rename, type cast, metadata).
        Adds _load_timestamp and _source_file metadata rows. Does NOT add surrogate keys.
        Returns JSON: {"sttm_path": "path/to/file.csv", "row_count": N}.
        Only call this when the orchestrator has requested a Bronze STTM.
        """
        if not profile_path:
            return json.dumps({"error": "No profile_path available for Bronze STTM"})

        context = _prepare_bronze_context(profile_path)
        context_tool_result = json.dumps(context, default=str)

        # Run a focused inner agent to generate the STTM rows from context.
        # Bronze is intent-agnostic: map every source column as-is.
        inner_prompt = (
            f"Generate a complete Bronze STTM JSON array from this profile context.\n"
            f"Profile context:\n{context_tool_result[:6000]}\n\n"
            "Bronze is intent-agnostic: cover EVERY source column mechanically — "
            "do not filter, prioritise, or omit any column based on perceived relevance.\n"
            "Return ONLY a valid JSON array of STTM rows. Each row must have: "
            "source_schema, source_table, source_column, target_schema, target_table, "
            "target_column, transformation_type, transformation_logic — all STTMs use these 8 columns. "
            "No markdown fences, no prose."
        )
        llm = _make_llm()
        response = llm.invoke(inner_prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        # Strip fences if present
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        raw = raw.strip()
        start, end = raw.find("["), raw.rfind("]")
        rows = []
        if start != -1 and end != -1:
            try:
                rows = json.loads(raw[start: end + 1])
            except (json.JSONDecodeError, ValueError):
                rows = []

        sttm_path = str(STTM_DIR / f"sttm_bronze_{run_id[:8]}.csv")
        pd.DataFrame(rows).to_csv(sttm_path, index=False)
        scratchpad["sttm_path"] = sttm_path
        print(f"[STTM] Bronze STTM saved: {sttm_path} ({len(rows)} rows)")
        return json.dumps({"sttm_path": sttm_path, "row_count": len(rows)})

    @tool
    def generate_silver_sttm_tool(confirmation: str = "execute") -> str:
        """Generate a complete Silver STTM CSV from Bronze Parquet outputs.

        Covers every Bronze column with cleansing rules (null handling, deduplication,
        type casting, date standardisation, surrogate key as first row).
        Returns JSON: {"sttm_path": "path/to/file.csv", "row_count": N}.
        Only call this when the orchestrator has requested a Silver STTM.
        """
        if not (bronze_output_paths and bronze_sttm_path):
            return json.dumps({"error": "No bronze_output_paths/bronze_sttm_path available for Silver STTM"})

        context = _prepare_silver_context(bronze_output_paths, bronze_sttm_path)
        context_str = json.dumps(context, default=str)

        # Silver is intent-agnostic: apply standard cleansing to every Bronze column.
        inner_prompt = (
            f"Generate a complete Silver STTM JSON array from this Bronze output metadata.\n"
            f"Bronze metadata:\n{context_str[:6000]}\n\n"
            "Silver is intent-agnostic: apply standard cleansing rules to EVERY Bronze column — "
            "do not filter or prioritise based on any business question.\n"
            "STRUCTURE: Each row must have 8 columns: source_schema, source_table, source_column, "
            "target_schema, target_table, target_column, transformation_type, transformation_logic.\n"
            "RULES: First row = surrogate key (source_schema='raw', source_table='<table_name>', "
            "source_column='', target_schema='cleansed', target_table='<table_name>', "
            "target_column='pk_<stem>_silver_id', transformation_type='Indirect', "
            "transformation_logic='Auto-generated sequential surrogate primary key starting from 1'). "
            "For all other rows: source_schema='raw', target_schema='cleansed', source_table and "
            "target_table should match the Bronze table name. Apply null handling, type casting, "
            "deduplication, date standardisation. For id columns: type casting only. "
            "Return ONLY a valid JSON array. No markdown fences, no prose."
        )
        llm = _make_llm()
        response = llm.invoke(inner_prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        raw = raw.strip()
        start, end = raw.find("["), raw.rfind("]")
        rows = []
        if start != -1 and end != -1:
            try:
                rows = json.loads(raw[start: end + 1])
            except (json.JSONDecodeError, ValueError):
                rows = []

        sttm_path = str(STTM_DIR / f"sttm_silver_{run_id[:8]}.csv")
        pd.DataFrame(rows).to_csv(sttm_path, index=False)
        scratchpad["sttm_path"] = sttm_path
        print(f"[STTM] Silver STTM saved: {sttm_path} ({len(rows)} rows)")
        return json.dumps({"sttm_path": sttm_path, "row_count": len(rows)})

    @tool
    def generate_gold_sttm_tool(confirmation: str = "execute") -> str:
        """Generate a complete Gold STTM CSV from Silver Parquet outputs.

        Covers every Silver column with materialisation rules (joins, renames,
        aggregations, passthrough, surrogate key as first row).
        Returns JSON: {"sttm_path": "path/to/file.csv", "row_count": N}.
        Only call this when the orchestrator has requested a Gold STTM.
        """
        if not (silver_output_paths and silver_sttm_path):
            return json.dumps({"error": "No silver_output_paths/silver_sttm_path available for Gold STTM"})
        if not business_intent:
            return json.dumps({"error": "business_intent is required for Gold STTM generation"})

        context = _prepare_gold_context(silver_output_paths, silver_sttm_path)
        context_str = json.dumps(context, default=str)

        inner_prompt = (
            f"Generate a complete Gold STTM JSON array from this Silver output metadata.\n"
            f"Business intent: {business_intent}\n"
            f"Silver metadata:\n{context_str[:6000]}\n\n"
            "STRUCTURE: Each row must have 8 columns: source_schema, source_table, source_column, "
            "target_schema, target_table, target_column, transformation_type, transformation_logic.\n"
            "RULES: First row = surrogate key (source_schema='cleansed', source_table='', "
            "source_column='', target_schema='analytics', target_table='<gold_table_name>', "
            "target_column='pk_gold_id', transformation_type='Indirect', "
            "transformation_logic='Auto-generated sequential surrogate primary key starting from 1'). "
            "Join Silver tables on matching key columns. "
            "Use Direct/Passthrough for unchanged columns. Build queryable tables, do NOT pre-aggregate. "
            "Return ONLY a valid JSON array. No markdown fences, no prose."
        )
        llm = _make_llm()
        response = llm.invoke(inner_prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        raw = raw.strip()
        start, end = raw.find("["), raw.rfind("]")
        rows = []
        if start != -1 and end != -1:
            try:
                rows = json.loads(raw[start: end + 1])
            except (json.JSONDecodeError, ValueError):
                rows = []

        sttm_path = str(STTM_DIR / f"sttm_gold_{run_id[:8]}.csv")
        pd.DataFrame(rows).to_csv(sttm_path, index=False)
        scratchpad["sttm_path"] = sttm_path
        print(f"[STTM] Gold STTM saved: {sttm_path} ({len(rows)} rows)")
        return json.dumps({"sttm_path": sttm_path, "row_count": len(rows)})

    return inspect_context_tool, generate_bronze_sttm_tool, generate_silver_sttm_tool, generate_gold_sttm_tool


# ---------------------------------------------------------------------------
# Shared agent runner
# ---------------------------------------------------------------------------

def _run_sttm_agent(
    trace_name: str,
    run_id: str,
    task_description: str,
    tools: list,
    scratchpad: dict,
    audit_action: str,
    audit_kwargs: dict,
    expected_filename_fragment: str,
) -> str:
    """Instantiate the unified STTM agent, invoke it, extract and return STTM path."""
    trace = AgentTrace(trace_name, run_id)
    trace.set_input(**audit_kwargs)

    audit = AuditLogger(run_id)
    audit.log("sttm_generator", audit_action, **audit_kwargs)

    llm = _make_llm()
    agent = create_agent(llm, tools, system_prompt=STTM_AGENT_PROMPT)

    try:
        result = agent.invoke({"messages": [HumanMessage(content=task_description)]})
    except Exception as e:
        trace.fail(str(e))
        raise

    messages = result.get("messages", [])
    trace.extract_from_messages(messages)

    # Primary: path captured by the generation tool via scratchpad
    sttm_path = scratchpad.get("sttm_path", "")

    # Fallback: scan messages for the path string if scratchpad was not populated
    if not sttm_path:
        for msg in reversed(messages):
            content = getattr(msg, "content", "")
            if isinstance(content, str) and expected_filename_fragment in content:
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "sttm_path" in parsed:
                        sttm_path = parsed["sttm_path"]
                        break
                except (json.JSONDecodeError, ValueError):
                    pass

    audit.log("sttm_generator", audit_action.replace("started", "completed"), output_file=sttm_path)
    trace.set_output(sttm_path=sttm_path).complete()
    return sttm_path


# ---------------------------------------------------------------------------
# Public entry points — I/O contract UNCHANGED
# ---------------------------------------------------------------------------

def generate_bronze_sttm(
    profile_path: str,
    run_id: str,
    task_description: str,
) -> str:
    """Bronze STTM agent entry point — autonomous ReAct version.

    Bronze is intent-agnostic. Every source column is mapped mechanically.

    Args:
        profile_path: Path to the combined profile JSON from the profiler.
        run_id: Unique identifier for this pipeline run.
        task_description: High-level goal message from the orchestrator.

    Returns:
        str: Path to the saved Bronze STTM CSV.
    """
    print(f"[STTM] Generating Bronze STTM for run_id: {run_id}")
    scratchpad: dict = {}
    tools = list(_make_sttm_tools(
        profile_path=profile_path,
        bronze_output_paths=None,
        bronze_sttm_path=None,
        silver_output_paths=None,
        silver_sttm_path=None,
        business_intent=None,
        run_id=run_id,
        scratchpad=scratchpad,
    ))
    return _run_sttm_agent(
        trace_name="sttm_bronze",
        run_id=run_id,
        task_description=task_description,
        tools=tools,
        scratchpad=scratchpad,
        audit_action="started_bronze",
        audit_kwargs={"profile_path": profile_path},
        expected_filename_fragment=f"sttm_bronze_{run_id[:8]}",
    )


def generate_silver_sttm(
    bronze_output_paths: list[str],
    bronze_sttm_path: str,
    run_id: str,
    task_description: str,
) -> str:
    """Silver STTM agent entry point — autonomous ReAct version.

    Silver is intent-agnostic. Standard cleansing rules are applied to every column.

    Args:
        bronze_output_paths: Bronze Parquet file paths to use as source schema context.
        bronze_sttm_path: Approved Bronze STTM CSV (used to filter to approved columns).
        run_id: Unique identifier for this pipeline run.
        task_description: High-level goal message from the orchestrator.

    Returns:
        str: Path to the saved Silver STTM CSV.
    """
    print(f"[STTM] Generating Silver STTM for run_id: {run_id}")
    scratchpad: dict = {}
    tools = list(_make_sttm_tools(
        profile_path=None,
        bronze_output_paths=bronze_output_paths,
        bronze_sttm_path=bronze_sttm_path,
        silver_output_paths=None,
        silver_sttm_path=None,
        business_intent=None,
        run_id=run_id,
        scratchpad=scratchpad,
    ))
    return _run_sttm_agent(
        trace_name="sttm_silver",
        run_id=run_id,
        task_description=task_description,
        tools=tools,
        scratchpad=scratchpad,
        audit_action="started_silver",
        audit_kwargs={"bronze_paths": bronze_output_paths},
        expected_filename_fragment=f"sttm_silver_{run_id[:8]}",
    )


def generate_gold_sttm(
    silver_output_paths: list[str],
    silver_sttm_path: str,
    business_intent: str,
    run_id: str,
    task_description: str,
) -> str:
    """Gold STTM agent entry point — autonomous ReAct version.

    Args:
        silver_output_paths: Silver Parquet file paths to use as source schema context.
        silver_sttm_path: Approved Silver STTM CSV (used to filter to approved columns).
        business_intent: Analytical goal guiding Gold table structure.
        run_id: Unique identifier for this pipeline run.
        task_description: High-level goal message from the orchestrator.

    Returns:
        str: Path to the saved Gold STTM CSV.
    """
    print(f"[STTM] Generating Gold STTM for run_id: {run_id}")
    scratchpad: dict = {}
    tools = list(_make_sttm_tools(
        profile_path=None,
        bronze_output_paths=None,
        bronze_sttm_path=None,
        silver_output_paths=silver_output_paths,
        silver_sttm_path=silver_sttm_path,
        business_intent=business_intent,
        run_id=run_id,
        scratchpad=scratchpad,
    ))
    return _run_sttm_agent(
        trace_name="sttm_gold",
        run_id=run_id,
        task_description=task_description,
        tools=tools,
        scratchpad=scratchpad,
        audit_action="started_gold",
        audit_kwargs={"silver_paths": silver_output_paths, "business_intent": business_intent},
        expected_filename_fragment=f"sttm_gold_{run_id[:8]}",
    )
