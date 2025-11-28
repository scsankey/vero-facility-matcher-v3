"""
storage.py
Batch storage abstraction for streaming-like ingestion workflow
Stores all ingested records with full audit trail
"""

import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path

STORAGE_DIR = Path("data")
STORAGE_FILE = STORAGE_DIR / "raw_records.csv"

# Schema for raw_records
RAW_RECORDS_COLUMNS = [
    "IngestID",
    "SourceSystem",
    "BatchLabel",
    "RecordID",
    "RawPayload",
    "IngestedAt"
]

def init_storage():
    """Ensure storage directory and file exist"""
    STORAGE_DIR.mkdir(exist_ok=True)
    
    if not STORAGE_FILE.exists():
        # Create empty storage file with schema
        empty_df = pd.DataFrame(columns=RAW_RECORDS_COLUMNS)
        empty_df.to_csv(STORAGE_FILE, index=False)
        print(f"✅ Created storage file: {STORAGE_FILE}")
    else:
        print(f"✅ Storage file exists: {STORAGE_FILE}")

def add_batch(source_system: str, df: pd.DataFrame, batch_label: str) -> str:
    """
    Add a batch of records to storage
    
    Parameters:
        source_system: "Gov", "NGO", "WhatsApp"
        df: DataFrame with source records
        batch_label: User-provided label for this batch
    
    Returns:
        ingest_id: Unique identifier for this batch
    """
    # Generate unique IngestID
    timestamp = datetime.now().isoformat()
    ingest_id = f"{timestamp}_{source_system.lower()}_{len(df)}"
    
    # Load existing storage
    if STORAGE_FILE.exists():
        storage_df = pd.read_csv(STORAGE_FILE)
    else:
        storage_df = pd.DataFrame(columns=RAW_RECORDS_COLUMNS)
    
    # Convert each row to storage format
    new_rows = []
    for idx, row in df.iterrows():
        new_rows.append({
            "IngestID": ingest_id,
            "SourceSystem": source_system,
            "BatchLabel": batch_label,
            "RecordID": str(row.get("RecordID", idx)),
            "RawPayload": row.to_json(date_format='iso'),          
            "IngestedAt": timestamp
        })
    
    # Append new rows
    new_df = pd.DataFrame(new_rows)
    storage_df = pd.concat([storage_df, new_df], ignore_index=True)
    
    # Save
    storage_df.to_csv(STORAGE_FILE, index=False)
    
    print(f"✅ Added batch '{batch_label}': {len(new_rows)} records (IngestID: {ingest_id})")
    
    return ingest_id

def load_all_raw() -> pd.DataFrame:
    """Load all raw records from storage"""
    if not STORAGE_FILE.exists():
        return pd.DataFrame(columns=RAW_RECORDS_COLUMNS)
    
    return pd.read_csv(STORAGE_FILE)

def get_ingestion_history() -> pd.DataFrame:
    """Get summary of all ingestion batches"""
    all_raw = load_all_raw()
    
    if len(all_raw) == 0:
        return pd.DataFrame(columns=["IngestID", "SourceSystem", "BatchLabel", "RecordCount", "IngestedAt"])
    
    history = all_raw.groupby(["IngestID", "SourceSystem", "BatchLabel", "IngestedAt"]).size().reset_index()
    history.rename(columns={0: "RecordCount"}, inplace=True)
    history = history.sort_values("IngestedAt", ascending=False)
    
    return history

def clear_storage():
    """
    Clear all stored data (developer/debug only)
    WARNING: This deletes all ingested batches
    """
    if STORAGE_FILE.exists():
        STORAGE_FILE.unlink()
        print("⚠️ Storage cleared")
    init_storage()

def get_storage_stats() -> dict:
    """Get statistics about stored data"""
    all_raw = load_all_raw()
    
    if len(all_raw) == 0:
        return {
            "total_records": 0,
            "total_batches": 0,
            "sources": {},
            "earliest_ingest": None,
            "latest_ingest": None
        }
    
    return {
        "total_records": len(all_raw),
        "total_batches": all_raw["IngestID"].nunique(),
        "sources": all_raw["SourceSystem"].value_counts().to_dict(),
        "earliest_ingest": all_raw["IngestedAt"].min(),
        "latest_ingest": all_raw["IngestedAt"].max()
    }
