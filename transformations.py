"""
transformations.py
Reshape raw storage records into logical source tables for vero_engine
"""

import pandas as pd
import json

def reshape_raw_to_logical_sources(all_raw_df: pd.DataFrame):
    """
    Transform raw_records storage into three logical source DataFrames
    
    Input:
        all_raw_df: DataFrame with columns [IngestID, SourceSystem, RawPayload, ...]
    
    Output:
        tuple: (gov_all_df, ngo_all_df, wa_all_df)
    """
    
    if len(all_raw_df) == 0:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame()
        )
    
    # Parse RawPayload JSON
    def parse_payload(payload_str):
        try:
            return json.loads(payload_str)
        except:
            return {}
    
    all_raw_df["parsed_payload"] = all_raw_df["RawPayload"].apply(parse_payload)
    
    # Separate by SourceSystem
    gov_raw = all_raw_df[all_raw_df["SourceSystem"] == "Gov"].copy()
    ngo_raw = all_raw_df[all_raw_df["SourceSystem"] == "NGO"].copy()
    wa_raw = all_raw_df[all_raw_df["SourceSystem"] == "WhatsApp"].copy()
    
    # Reconstruct Gov DataFrame
    if len(gov_raw) > 0:
        gov_records = []
        for _, row in gov_raw.iterrows():
            payload = row["parsed_payload"]
            gov_records.append({
                "RecordID": payload.get("RecordID", row["RecordID"]),
                "OfficialFacilityName": payload.get("OfficialFacilityName", ""),
                "AltName": payload.get("AltName", ""),
                "District": payload.get("District", ""),
                "AdminLevel2": payload.get("AdminLevel2", ""),
                "GPS_Lat": payload.get("GPS_Lat", None),
                "GPS_Lon": payload.get("GPS_Lon", None),
                # Preserve any additional fields
                **{k: v for k, v in payload.items() 
                   if k not in ["RecordID", "OfficialFacilityName", "AltName", "District"]}
            })
        gov_all_df = pd.DataFrame(gov_records)
    else:
        gov_all_df = pd.DataFrame(columns=["RecordID", "OfficialFacilityName", "District"])
    
    # Reconstruct NGO DataFrame
    if len(ngo_raw) > 0:
        ngo_records = []
        for _, row in ngo_raw.iterrows():
            payload = row["parsed_payload"]
            ngo_records.append({
                "RecordID": payload.get("RecordID", row["RecordID"]),
                "FarmerID": payload.get("FarmerID", ""),
                "FarmerName": payload.get("FarmerName", ""),
                "FacilityName": payload.get("FacilityName", ""),
                "District": payload.get("District", ""),
                "Phone": payload.get("Phone", ""),
                "Village": payload.get("Village", ""),
                "Coop": payload.get("Coop", ""),
                "Gender": payload.get("Gender", ""),
                **{k: v for k, v in payload.items() 
                   if k not in ["RecordID", "FacilityName", "District", "Phone"]}
            })
        ngo_all_df = pd.DataFrame(ngo_records)
    else:
        ngo_all_df = pd.DataFrame(columns=["RecordID", "FacilityName", "District"])
    
    # Reconstruct WhatsApp DataFrame
    if len(wa_raw) > 0:
        wa_records = []
        for _, row in wa_raw.iterrows():
            payload = row["parsed_payload"]
            wa_records.append({
                "RecordID": payload.get("RecordID", row["RecordID"]),
                "ChatID": payload.get("ChatID", ""),
                "ContactName": payload.get("ContactName", ""),
                "RelatedFacility": payload.get("RelatedFacility", payload.get("Lookup", "")),
                "DistrictNote": payload.get("DistrictNote", payload.get("District", "")),
                "Phone": payload.get("Phone", ""),
                "LocationNickname": payload.get("LocationNickname", ""),
                **{k: v for k, v in payload.items() 
                   if k not in ["RecordID", "RelatedFacility", "DistrictNote", "Phone"]}
            })
        wa_all_df = pd.DataFrame(wa_records)
    else:
        wa_all_df = pd.DataFrame(columns=["RecordID", "RelatedFacility", "DistrictNote"])
    
    return gov_all_df, ngo_all_df, wa_all_df

def enrich_with_ingest_metadata(entity_clusters_df: pd.DataFrame, all_raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add IngestID and BatchLabel to entity_clusters for full audit trail
    
    Parameters:
        entity_clusters_df: Output from vero_engine
        all_raw_df: Raw storage records
    
    Returns:
        Enhanced entity_clusters with IngestID and BatchLabel
    """
    if len(entity_clusters_df) == 0 or len(all_raw_df) == 0:
        return entity_clusters_df
    
    # Create lookup from RecordID to IngestID/BatchLabel
    lookup = all_raw_df[["RecordID", "IngestID", "BatchLabel"]].drop_duplicates()
    
    # Merge
    enriched = entity_clusters_df.merge(
        lookup,
        on="RecordID",
        how="left"
    )
    
    return enriched
