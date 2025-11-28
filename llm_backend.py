"""
llm_backend.py
Proper free LLM backend connector with canonical-safe prompting
Supports: mock, HuggingFace Inference API, local LLM
"""

import os
import json
import requests
import pandas as pd
from typing import List, Dict

# ============================================================================
# CONFIGURATION
# ============================================================================

def get_llm_status() -> Dict:
    """Get current LLM backend configuration"""
    mode = os.getenv("VERO_LLM_MODE", "mock")
    
    status = {"mode": mode}
    
    if mode == "hf_inference":
        status["hf_model"] = os.getenv("VERO_HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
        status["hf_token_set"] = bool(os.getenv("HF_TOKEN"))
    elif mode == "local":
        status["local_url"] = os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1/chat/completions")
    
    return status

# ============================================================================
# CANONICAL CONTEXT BUILDER (CRITICAL FOR ADIP)
# ============================================================================

def build_entity_context(entity_id: str, 
                        canonical_df: pd.DataFrame, 
                        entity_clusters_df: pd.DataFrame,
                        events_df: pd.DataFrame = None) -> Dict:
    """
    Build structured context for a canonical entity
    ADIP Principle: LLM only reads canonical + curated context
    
    Returns:
        {
            "canonical": {EntityID, CanonicalName, EntityType, ...},
            "sources": [{RecordID, SourceSystem, SourceName, ...}],
            "events": [{EventType, EventDate, EventDetails}]
        }
    """
    # 1. Fetch canonical row
    canon_rows = canonical_df[canonical_df["EntityID"] == entity_id]
    if canon_rows.empty:
        raise ValueError(f"EntityID {entity_id} not found in canonical_entities")
    
    canon_row = canon_rows.iloc[0]
    canonical_name = canon_row["CanonicalName"]
    entity_type = canon_row.get("EntityType", "Unknown")
    primary_district = canon_row.get("PrimaryDistrict", "Unknown")
    
    # Parse aliases
    aliases_raw = canon_row.get("Aliases", "")
    if isinstance(aliases_raw, str):
        alias_list = [a.strip() for a in aliases_raw.split("|") if a.strip()]
        if ";" in aliases_raw:  # Alternative separator
            alias_list = [a.strip() for a in aliases_raw.split(";") if a.strip()]
    elif isinstance(aliases_raw, list):
        alias_list = aliases_raw
    else:
        alias_list = []
    
    # 2. Get all cluster/source rows
    cluster_rows = entity_clusters_df[entity_clusters_df["EntityID"] == entity_id].copy()
    
    # 3. Build compact structured view of sources
    sources_summary = []
    for _, r in cluster_rows.iterrows():
        sources_summary.append({
            "RecordID": r.get("RecordID", ""),
            "SourceSystem": r.get("SourceSystem", ""),
            "SourceName": r.get("SourceName", ""),
            "SourceDistrict": r.get("SourceDistrict", ""),
            "IngestBatch": r.get("IngestID", ""),
            "BatchLabel": r.get("BatchLabel", "")
        })
    
    # 4. Event timeline (if any)
    events_summary = []
    if events_df is not None and not events_df.empty and "EntityID" in events_df.columns:
        ev_rows = events_df[events_df["EntityID"] == entity_id]
        for _, e in ev_rows.iterrows():
            events_summary.append({
                "EventType": e.get("EventType", ""),
                "EventDate": str(e.get("EventTime", e.get("EventDate", ""))),
                "EventDetails": e.get("Summary", e.get("EventDetails", ""))
            })
    
    return {
        "canonical": {
            "EntityID": entity_id,
            "CanonicalName": canonical_name,
            "EntityType": entity_type,
            "PrimaryDistrict": primary_district,
            "Aliases": alias_list,
            "RecordCount": canon_row.get("RecordCount", len(sources_summary)),
            "DataQualityScore": canon_row.get("DataQualityScore", "N/A")
        },
        "sources": sources_summary,
        "events": events_summary
    }

# ============================================================================
# PROMPT CONSTRUCTION WITH CANONICAL RULES
# ============================================================================

def build_prompt_from_context(context: Dict, question: str) -> List[Dict]:
    """
    Build chat-style prompt enforcing canonical name rules
    
    Returns:
        List of message dicts: [{"role": "system", "content": "..."}, ...]
    """
    canon = context["canonical"]
    canonical_name = canon["CanonicalName"]
    aliases = canon.get("Aliases", [])
    
    system_msg = f"""You are a data assistant on top of a humanitarian identity resolution system.

You will answer questions about ONE canonical entity.

CRITICAL RULES:
1. Always refer to the entity using its CANONICAL_NAME: "{canonical_name}".
2. You may mention aliases ONCE in parentheses at first reference, like:
   {canonical_name} (also recorded as: {", ".join(aliases[:3])})
   but do NOT use aliases again afterwards.
3. Ignore spelling differences or noise in source names.
4. Base your answers on the structured data provided (sources and events).
5. If something is unclear or missing, say so explicitly. Do NOT invent facts.
6. Keep answers concise: 2-3 paragraphs maximum."""
    
    # Compact context for the model
    sources_text_lines = []
    for s in context["sources"][:10]:  # Limit to 10 for token efficiency
        sources_text_lines.append(
            f"- [{s.get('SourceSystem')}] Name='{s.get('SourceName')}', "
            f"District='{s.get('SourceDistrict')}', Batch='{s.get('BatchLabel', 'N/A')}'"
        )
    sources_text = "\n".join(sources_text_lines) if sources_text_lines else "No source records available."
    
    events_text_lines = []
    for e in context["events"][:10]:  # Limit to 10
        events_text_lines.append(
            f"- [{e.get('EventDate')}] {e.get('EventType')}: {e.get('EventDetails')}"
        )
    events_text = "\n".join(events_text_lines) if events_text_lines else "No recorded events."
    
    context_msg = f"""CANONICAL ENTITY:
- ID: {canon["EntityID"]}
- CANONICAL_NAME: {canonical_name}
- ENTITY_TYPE: {canon["EntityType"]}
- PRIMARY_DISTRICT: {canon["PrimaryDistrict"]}
- ALIASES: {", ".join(aliases) if aliases else "None"}
- TOTAL_RECORDS: {canon.get("RecordCount", 0)}
- DATA_QUALITY: {canon.get("DataQualityScore", "N/A")}

SOURCE RECORDS (deduplicated from):
{sources_text}

EVENTS:
{events_text}"""
    
    messages = [
        {"role": "system", "content": system_msg.strip()},
        {"role": "user", "content": context_msg.strip()},
        {"role": "user", "content": f"User question: {question}"}
    ]
    
    return messages

# ============================================================================
# LLM CALLERS
# ============================================================================

def call_llm(messages: List[Dict]) -> str:
    """Route to appropriate LLM backend based on configuration"""
    status = get_llm_status()
    mode = status["mode"]
    
    if mode == "mock":
        return _call_llm_mock(messages)
    elif mode == "hf_inference":
        return _call_llm_hf(messages, status)
    elif mode == "local":
        return _call_llm_local(messages, status)
    else:
        return "❌ Unsupported LLM mode. Set VERO_LLM_MODE to 'mock', 'hf_inference', or 'local'."

def _call_llm_mock(messages: List[Dict]) -> str:
    """Mock mode - deterministic fake answer for demos"""
    # Extract canonical name from messages
    context_msg = messages[1]["content"] if len(messages) > 1 else ""
    
    if "CANONICAL_NAME:" in context_msg:
        name = context_msg.split("CANONICAL_NAME:")[1].split("\n")[0].strip()
    else:
        name = "this entity"
    
    return f"""Based on the canonical entity data, {name} has been successfully deduplicated across multiple data sources through the VERO identity resolution system.

The entity's data quality score and source representation indicate good coverage across the available data ecosystem. All spelling variations and duplicate records have been consolidated into this single canonical identity.

This provides a reliable foundation for reporting, analytics, and decision-making without the noise of multiple conflicting records."""

def _call_llm_hf(messages: List[Dict], status: Dict) -> str:
    """HuggingFace Inference API"""
    hf_token = os.getenv("HF_TOKEN")
    model_id = status["hf_model"]
    
    if not hf_token:
        return "⚠️ HF_TOKEN not set. Running in mock mode.\n\n" + _call_llm_mock(messages)
    
    # Join messages into a single prompt for text generation models
    prompt = ""
    for m in messages:
        if m["role"] == "system":
            prompt += "System: " + m["content"] + "\n\n"
        elif m["role"] == "user":
            prompt += m["content"] + "\n\n"
    
    prompt += "Assistant:"
    
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.3,
            "top_p": 0.9,
            "return_full_text": False
        }
    }
    
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        
        # HF returns list of dicts with 'generated_text'
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("generated_text", "").strip()
        elif isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"].strip()
        else:
            return f"⚠️ Unexpected response format: {data}"
    
    except requests.exceptions.Timeout:
        return "⏱️ Request timed out. HuggingFace model may be loading. Wait 20-30 seconds and try again."
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 503:
            return "⏳ Model is loading on HuggingFace (cold start). Please wait 20-30 seconds and try again."
        return f"❌ API Error {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return f"❌ Error calling HuggingFace: {str(e)}"

def _call_llm_local(messages: List[Dict], status: Dict) -> str:
    """Local LLM (LM Studio, Ollama, etc.)"""
    local_url = status["local_url"]
    
    payload = {
        "model": "local-model",
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 512
    }
    
    try:
        resp = requests.post(local_url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        
        # OpenAI-compatible schema
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"].strip()
        else:
            return f"⚠️ Unexpected local LLM response: {data}"
    
    except Exception as e:
        return f"❌ Error calling local LLM at {local_url}: {str(e)}\n\nMake sure your local LLM server is running."

# ============================================================================
# POST-PROCESSING
# ============================================================================

def normalize_names_in_answer(answer: str, canonical_name: str, aliases: List[str]) -> str:
    """Replace any alias mentions with canonical name"""
    text = answer
    for alias in aliases:
        if alias and alias.lower() != canonical_name.lower():
            # Case-insensitive replacement
            import re
            pattern = re.compile(re.escape(alias), re.IGNORECASE)
            text = pattern.sub(canonical_name, text)
    return text

# ============================================================================
# PUBLIC API
# ============================================================================

def answer_entity_question(entity_id: str,
                          question: str,
                          canonical_df: pd.DataFrame,
                          entity_clusters_df: pd.DataFrame,
                          events_df: pd.DataFrame = None) -> str:
    """
    Main entrypoint for LLM Q&A on canonical entities
    
    Parameters:
        entity_id: Canonical EntityID
        question: User's question
        canonical_df: Canonical entities table
        entity_clusters_df: Entity clusters with full lineage
        events_df: Optional events table
    
    Returns:
        LLM answer with canonical name enforced
    """
    if not question.strip():
        return "Please enter a question."
    
    try:
        # 1) Build context
        context = build_entity_context(
            entity_id=entity_id,
            canonical_df=canonical_df,
            entity_clusters_df=entity_clusters_df,
            events_df=events_df
        )
        
        canonical_name = context["canonical"]["CanonicalName"]
        aliases = context["canonical"]["Aliases"]
        
        # 2) Build prompt
        messages = build_prompt_from_context(context, question)
        
        # 3) Call LLM
        raw_answer = call_llm(messages)
        
        # 4) Normalize names
        final_answer = normalize_names_in_answer(raw_answer, canonical_name, aliases)
        
        return final_answer
    
    except ValueError as e:
        return f"❌ Error: {str(e)}"
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"
