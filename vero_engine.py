"""
vero_engine.py
VERO Entity Resolution Engine - EARE Multi-Entity Standard v4.0

EARE Framework:
E = Entities: facilities, people, districts, shipments, etc.
A = Attributes: dynamic key-value metadata per entity
R = Relationships: links between entities (scaffolded)
E = Events: time-based facts per entity (scaffolded)

Core Principles:
1. Multi-entity support via EntityType
2. Backward compatible with facility-only usage
3. Type-safe clustering (never mix entity types)
4. Extensible via explicit configs
5. Canonical entities with dynamic attributes
"""

import pandas as pd
import numpy as np
import networkx as nx
from rapidfuzz import fuzz, distance
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

SCHEMA_VERSION = 1
FEATURE_COLS = [
    "jw_name", "lev_name", "token_sim", "starts_with", "ends_with", "nickname",
    "jw_alt_a", "jw_alt_b", "jw_alt_both", "cos_embed",
    "district_exact", "district_fuzzy",
    "phone_exact", "phone_partial", "phone_both_missing",
]

HIGH_CONFIDENCE_THRESHOLD = 0.90
MEDIUM_CONFIDENCE_THRESHOLD = 0.75
NAME_SIMILARITY_THRESHOLD = 0.85
DISTRICT_BLOCKING_THRESHOLD = 0.75

# ============================================================================
# DATA PREPARATION (MULTI-ENTITY / EARE READY)
# ============================================================================

def prepare_unified_dataset(
    gov_df=None,
    ngo_df=None,
    whatsapp_df=None,
    extra_entity_sources=None
):
    """
    Build a unified raw-record table from multiple sources.

    Each row = one raw record (facility / person / district / shipment / etc.)

    Unified schema:

        RecordID        : str  (unique within its source system)
        EntityType      : str  ('facility', 'person', 'district', 'shipment', ...)
        Name            : str  (primary name)
        AltName         : str or None
        District        : str or None
        Phone           : str or None
        SourceSystem    : str  ('Gov', 'NGO', 'WhatsApp', 'HMIS', 'HR', ...)
        Attributes_JSON : str  (JSON serialised dict of extra attributes)
        # Derived/cleaned columns added later:
        #   name_clean, alt_a_clean, alt_b_clean, district_clean, phone_clean

    Parameters
    ----------
    gov_df : pd.DataFrame or None
        Expected columns: ['RecordID', 'OfficialFacilityName', 'District', ...]
        Mapped as EntityType='facility', SourceSystem='Gov'.

    ngo_df : pd.DataFrame or None
        Expected columns: ['RecordID', 'FacilityName', 'District', 'Phone', ...]
        Mapped as EntityType='facility', SourceSystem='NGO'.

    whatsapp_df : pd.DataFrame or None
        Expected columns: ['RecordID', 'RelatedFacility', 'DistrictNote', 'Phone', 'LocationNickname', ...]
        Mapped as EntityType='facility', SourceSystem='WhatsApp'.

    extra_entity_sources : list[dict] or None
        Optional list of configs for MORE entity types.
        Each item MUST have this shape:

        {
            "df": <pd.DataFrame>,
            "entity_type": "person" | "district" | "shipment" | ...,
            "source_system": "HR" | "Bank" | "WFP" | ...,
            "name_col": "FullName",           # main name column
            "alt_name_col": "AltNameCol",     # or None
            "district_col": "DistrictCol",    # or None
            "phone_col": "PhoneCol",          # or None
            "record_id_col": "RecordID",      # unique per source
            "extra_attribute_cols": ["Gender", "Nationality", "Occupation"]
        }

    Returns
    -------
    unified : pd.DataFrame
        Unified table with EARE-ready schema for Entities + Attributes.
    """
    import json

    rows = []

    # ---------- 1) Built-in facility sources (backward compatible) ----------

    if gov_df is not None:
        for _, r in gov_df.iterrows():
            attrs = {}
            # Any extra columns we want to carry into Attributes
            for col in gov_df.columns:
                if col not in ["RecordID", "OfficialFacilityName", "AltName", "District"]:
                    attrs[col] = r.get(col, None)
            rows.append({
                "RecordID": str(r["RecordID"]),
                "EntityType": "facility",
                "Name": r.get("OfficialFacilityName", ""),
                "AltName": r.get("AltName", None),
                "District": r.get("District", None),
                "Phone": None,
                "SourceSystem": "Gov",
                "Attributes_JSON": json.dumps(attrs, default=str)
            })

    if ngo_df is not None:
        for _, r in ngo_df.iterrows():
            attrs = {}
            for col in ngo_df.columns:
                if col not in ["RecordID", "FacilityName", "District", "Phone"]:
                    attrs[col] = r.get(col, None)
            rows.append({
                "RecordID": str(r["RecordID"]),
                "EntityType": "facility",
                "Name": r.get("FacilityName", ""),
                "AltName": None,
                "District": r.get("District", None),
                "Phone": r.get("Phone", None),
                "SourceSystem": "NGO",
                "Attributes_JSON": json.dumps(attrs, default=str)
            })

    if whatsapp_df is not None:
        for _, r in whatsapp_df.iterrows():
            attrs = {}
            for col in whatsapp_df.columns:
                if col not in ["RecordID", "RelatedFacility", "DistrictNote", "Phone", "LocationNickname"]:
                    attrs[col] = r.get(col, None)
            rows.append({
                "RecordID": str(r["RecordID"]),
                "EntityType": "facility",
                "Name": r.get("RelatedFacility", ""),
                "AltName": r.get("LocationNickname", None),
                "District": r.get("DistrictNote", None),
                "Phone": r.get("Phone", None),
                "SourceSystem": "WhatsApp",
                "Attributes_JSON": json.dumps(attrs, default=str)
            })

    # ---------- 2) Extra multi-entity sources (EARE-ready) ----------

    if extra_entity_sources:
        import json
        for cfg in extra_entity_sources:
            df = cfg["df"]
            ent_type = cfg["entity_type"]
            src = cfg["source_system"]
            name_col = cfg["name_col"]
            alt_name_col = cfg.get("alt_name_col")
            district_col = cfg.get("district_col")
            phone_col = cfg.get("phone_col")
            record_id_col = cfg["record_id_col"]
            extra_cols = cfg.get("extra_attribute_cols", [])

            for _, r in df.iterrows():
                attrs = {}
                for col in extra_cols:
                    attrs[col] = r.get(col, None)

                rows.append({
                    "RecordID": str(r[record_id_col]),
                    "EntityType": ent_type,
                    "Name": r.get(name_col, ""),
                    "AltName": r.get(alt_name_col, None) if alt_name_col else None,
                    "District": r.get(district_col, None) if district_col else None,
                    "Phone": r.get(phone_col, None) if phone_col else None,
                    "SourceSystem": src,
                    "Attributes_JSON": json.dumps(attrs, default=str)
                })

    unified = pd.DataFrame(rows)

    # ---------- 3) Derived cleaned fields (used by matcher) ----------

    unified["Name"] = unified["Name"].fillna("")
    unified["AltName"] = unified["AltName"].fillna("")
    unified["District"] = unified["District"].fillna("")
    unified["Phone"] = unified["Phone"].fillna("")

    unified["name_clean"] = (unified["Name"] + " " + unified["AltName"]).str.strip().str.lower()
    unified["alt_a_clean"] = unified["Name"].str.lower()
    unified["alt_b_clean"] = unified["AltName"].str.lower()
    unified["district_clean"] = unified["District"].str.lower().str.strip()
    unified["phone_clean"] = unified["Phone"].astype(str).str.replace(r"\D", "", regex=True)

    return unified

# ============================================================================
# STAGE 2: FEATURE COMPUTATION
# ============================================================================

def extract_facility_type(name: str) -> str:
    """Extract facility type for matching"""
    name_lower = str(name).lower()
    
    if 'district hospital' in name_lower:
        return 'district_hospital'
    elif 'hospital' in name_lower:
        return 'hospital'
    elif 'rural health centre' in name_lower or 'rhc' in name_lower:
        return 'rural_health_centre'
    elif 'health centre' in name_lower or 'health center' in name_lower or ' hc' in name_lower:
        return 'health_centre'
    elif 'clinic' in name_lower:
        return 'clinic'
    elif 'health post' in name_lower:
        return 'health_post'
    else:
        return 'unknown'

def nickname_score(name_a: str, name_b: str) -> int:
    """Simple nickname heuristic"""
    tokens_a = name_a.split()
    tokens_b = name_b.split()
    if not tokens_a or not tokens_b:
        return 0
    if tokens_a[0] == tokens_b[0]:
        return 1
    if tokens_a[0] and tokens_b[0] and tokens_a[0][0] == tokens_b[0][0]:
        return 1
    return 0

def compute_pair_features(rec_a, rec_b, id_to_emb):
    """
    Compute similarity features for entity resolution
    Handles both Series and dict inputs
    """
    # Convert Series to dict if needed
    if isinstance(rec_a, pd.Series):
        rec_a = rec_a.to_dict()
    if isinstance(rec_b, pd.Series):
        rec_b = rec_b.to_dict()
    
    # Extract cleaned fields
    name_a = rec_a["name_clean"]
    name_b = rec_b["name_clean"]
    
    # Name similarity metrics
    jw_name = distance.JaroWinkler.normalized_similarity(name_a, name_b)
    lev_name = distance.Levenshtein.normalized_similarity(name_a, name_b)
    token_sim = fuzz.token_set_ratio(name_a, name_b) / 100.0
    
    # Token-based features
    tokens_a = name_a.split()
    tokens_b = name_b.split()
    starts_with = int(len(tokens_a) > 0 and len(tokens_b) > 0 and tokens_a[0] == tokens_b[0])
    ends_with = int(len(tokens_a) > 0 and len(tokens_b) > 0 and tokens_a[-1] == tokens_b[-1])
    nickname = nickname_score(name_a, name_b)
    
    # Alternative name comparisons
    alt_a_a = rec_a["alt_a_clean"]
    alt_a_b = rec_b["alt_a_clean"]
    alt_b_a = rec_a["alt_b_clean"]
    alt_b_b = rec_b["alt_b_clean"]
    
    jw_alt_a = distance.JaroWinkler.normalized_similarity(alt_a_a, alt_a_b)
    jw_alt_b = distance.JaroWinkler.normalized_similarity(alt_b_a, alt_b_b)
    jw_alt_both = distance.JaroWinkler.normalized_similarity(
        alt_a_a + " " + alt_b_a, alt_a_b + " " + alt_b_b
    )
    
    # District comparison
    district_a = rec_a["district_clean"]
    district_b = rec_b["district_clean"]
    district_exact = int(district_a == district_b and district_a != "")
    district_fuzzy = fuzz.token_set_ratio(district_a, district_b) / 100.0
    
    # Phone comparison
    phone_a = rec_a["phone_clean"]
    phone_b = rec_b["phone_clean"]
    phone_exact = int(phone_a != "" and phone_a == phone_b)
    
    last4_a = phone_a[-4:] if len(phone_a) >= 4 else ""
    last4_b = phone_b[-4:] if len(phone_b) >= 4 else ""
    phone_partial = int(last4_a != "" and last4_a == last4_b)
    phone_both_missing = int(phone_a == "" and phone_b == "")
    
    # Embedding similarity
    # For Series, use the index (RecordID), for dict use the 'RecordID' key
    rec_a_id = rec_a.get('RecordID') if isinstance(rec_a, dict) else rec_a.name
    rec_b_id = rec_b.get('RecordID') if isinstance(rec_b, dict) else rec_b.name
    
    emb_a = id_to_emb.get(rec_a_id)
    emb_b = id_to_emb.get(rec_b_id)
    cos_embed = float(util.cos_sim(emb_a, emb_b).cpu().numpy()[0][0]) if emb_a is not None and emb_b is not None else 0.0
    
    return {
        "jw_name": jw_name, "lev_name": lev_name, "token_sim": token_sim,
        "starts_with": starts_with, "ends_with": ends_with, "nickname": nickname,
        "jw_alt_a": jw_alt_a, "jw_alt_b": jw_alt_b, "jw_alt_both": jw_alt_both,
        "cos_embed": cos_embed, "district_exact": district_exact, "district_fuzzy": district_fuzzy,
        "phone_exact": phone_exact, "phone_partial": phone_partial, "phone_both_missing": phone_both_missing,
    }

# ============================================================================
# STAGE 3: BLOCKING & CANDIDATE GENERATION (EARE-AWARE)
# ============================================================================

def should_compare(rid_a, rid_b, row_a, row_b, district_threshold=DISTRICT_BLOCKING_THRESHOLD):
    """
    Decide whether two records should be compared.

    EARE-ready changes:
    - Do NOT compare records of different EntityType.
    - Require at least 2 blocking conditions to be true.
    """
    if rid_a == rid_b:
        return False

    # ---------- HARD BLOCK: entity type mismatch ----------
    if row_a.get("EntityType") != row_b.get("EntityType"):
        return False

    conditions_met = 0

    # Condition 1: District match
    d_a = row_a["district_clean"]
    d_b = row_b["district_clean"]
    if d_a and d_b:
        district_score = fuzz.token_set_ratio(d_a, d_b) / 100.0
        if district_score >= district_threshold:
            conditions_met += 1

    # Condition 2: Phone last4 match
    p_a = row_a["phone_clean"]
    p_b = row_b["phone_clean"]
    last4_a = p_a[-4:] if len(p_a) >= 4 else ""
    last4_b = p_b[-4:] if len(p_b) >= 4 else ""
    if last4_a and last4_a == last4_b:
        conditions_met += 1

    # Condition 3: Meaningful name token overlap (excluding generic words)
    common_words = {'health', 'centre', 'center', 'hospital', 'district',
                    'rural', 'clinic', 'post', 'community', 'area'}
    tokens_a = set(row_a["name_clean"].split()) - common_words
    tokens_b = set(row_b["name_clean"].split()) - common_words

    if len(tokens_a & tokens_b) > 0:
        conditions_met += 1

    # Condition 4: Facility-type consistency (only for facilities)
    if row_a.get("EntityType") == "facility" and row_b.get("EntityType") == "facility":
        type_a = extract_facility_type(row_a["Name"])
        type_b = extract_facility_type(row_b["Name"])
        if type_a != 'unknown' and type_b != 'unknown' and type_a != type_b:
            return False

    return conditions_met >= 2

def generate_candidate_pairs(unified_df):
    """
    Generate candidate pairs using blocking.

    EARE-ready:
    - Only compare records with the same EntityType.
    - Currently still focuses on NGO vs Gov vs WhatsApp, but can be extended.
    """
    u_idx = unified_df.set_index("RecordID")

    ngo_ids = unified_df[unified_df["SourceSystem"] == "NGO"]["RecordID"].tolist()
    gov_ids = unified_df[unified_df["SourceSystem"] == "Gov"]["RecordID"].tolist()
    wa_ids  = unified_df[unified_df["SourceSystem"] == "WhatsApp"]["RecordID"].tolist()

    candidate_pairs = []

    # NGO vs Gov
    for rid_ng in ngo_ids:
        row_ng = u_idx.loc[rid_ng]
        for rid_g in gov_ids:
            row_g = u_idx.loc[rid_g]
            if should_compare(rid_ng, rid_g, row_ng, row_g):
                candidate_pairs.append((rid_ng, rid_g))

    # NGO vs WhatsApp
    for rid_ng in ngo_ids:
        row_ng = u_idx.loc[rid_ng]
        for rid_w in wa_ids:
            row_w = u_idx.loc[rid_w]
            if should_compare(rid_ng, rid_w, row_ng, row_w):
                candidate_pairs.append((rid_ng, rid_w))

    # Gov vs WhatsApp
    for rid_g in gov_ids:
        row_g = u_idx.loc[rid_g]
        for rid_w in wa_ids:
            row_w = u_idx.loc[rid_w]
            if should_compare(rid_g, rid_w, row_g, row_w):
                candidate_pairs.append((rid_g, rid_w))

    return list(set(candidate_pairs))

# ============================================================================
# STAGE 4: MODEL TRAINING
# ============================================================================

def train_model(similarity_features_df):
    """Train logistic regression model"""
    X = similarity_features_df[FEATURE_COLS].fillna(0)
    y = similarity_features_df['SameEntity']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced', C=1.0)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    return model, scaler, metrics

# ============================================================================
# STAGE 5: CLUSTER BUILDING (EARE-AWARE)
# ============================================================================

def build_clusters(matched_pairs_df, unified_df):
    """
    Build entity clusters using graph connected components.

    EARE-ready:
    - Clusters are still graphs, but each row now carries EntityType and SourceSystem.
    - Includes EntityID field for app.py compatibility (EntityID = ClusterID)
    """
    # Build graph from matched pairs
    G = nx.Graph()
    for _, row in matched_pairs_df.iterrows():
        G.add_edge(row["record_A"], row["record_B"])

    u_idx = unified_df.set_index("RecordID")
    cluster_rows = []

    # Connected components = clusters
    connected_components = list(nx.connected_components(G))

    for i, comp in enumerate(connected_components, start=1):
        cluster_id = f"Cluster_{i}"
        for rid in comp:
            row = u_idx.loc[rid]
            cluster_rows.append({
                "ClusterID": cluster_id,
                "EntityID": cluster_id,  # Added for app.py compatibility
                "RecordID": rid,
                "EntityType": row.get("EntityType", "unknown"),
                "SourceSystem": row.get("SourceSystem", ""),
                "SourceName": row.get("Name", ""),  # Backward compat field name
                "SourceDistrict": row.get("District", ""),  # Backward compat field name
                "SourcePhone": row.get("Phone", ""),  # Backward compat field name
                "Name": row.get("Name", ""),
                "AltName": row.get("AltName", ""),
                "District": row.get("District", ""),
                "Phone": row.get("Phone", ""),
                "Attributes_JSON": row.get("Attributes_JSON", "{}"),
            })

    # Add singletons: records that never matched anything
    matched_ids = set(matched_pairs_df["record_A"]) | set(matched_pairs_df["record_B"])
    all_ids = set(unified_df["RecordID"])
    singleton_ids = all_ids - matched_ids

    for rid in singleton_ids:
        row = u_idx.loc[rid]
        singleton_id = f"Singleton_{rid}"
        cluster_rows.append({
            "ClusterID": singleton_id,
            "EntityID": singleton_id,  # Added for app.py compatibility
            "RecordID": rid,
            "EntityType": row.get("EntityType", "unknown"),
            "SourceSystem": row.get("SourceSystem", ""),
            "SourceName": row.get("Name", ""),  # Backward compat field name
            "SourceDistrict": row.get("District", ""),  # Backward compat field name
            "SourcePhone": row.get("Phone", ""),  # Backward compat field name
            "Name": row.get("Name", ""),
            "AltName": row.get("AltName", ""),
            "District": row.get("District", ""),
            "Phone": row.get("Phone", ""),
            "Attributes_JSON": row.get("Attributes_JSON", "{}"),
        })

    clusters_df = pd.DataFrame(cluster_rows)
    return clusters_df

# ============================================================================
# STAGE 6: CANONICAL ENTITIES TABLE (EARE MULTI-ENTITY)
# ============================================================================

def build_canonical_entities_table(clusters_df):
    """
    Build a canonical_entities table from clusters.

    Each row = one canonical entity (EntityID).

    Columns:
        EntityID            : ClusterID or Singleton ID
        EntityType          : 'facility', 'person', 'district', ...
        CanonicalName       : primary name (preferring Gov if available)
        PrimaryDistrict     : best-choice district (preferring Gov, or most common)
        CanonicalPhones     : deduped phone list (string, ' | '-joined)
        Aliases             : all distinct names (incl AltName) ' | '-joined
        SourcesRepresented  : +-joined distinct SourceSystem values
        RecordCount         : number of raw records in the cluster
        SourceRecordIDs     : ' | '-joined RecordIDs
        Attributes_JSON     : merged attributes from Attributes_JSON field
    """
    import json
    canonical_rows = []

    if clusters_df is None or len(clusters_df) == 0:
        return pd.DataFrame()

    # Ensure string types
    clusters_df = clusters_df.copy()
    clusters_df["EntityType"] = clusters_df["EntityType"].fillna("unknown")

    for (cluster_id, entity_type), group in clusters_df.groupby(["ClusterID", "EntityType"]):
        # Prioritise Gov records for canonical name, if present
        gov_records = group[group["SourceSystem"] == "Gov"]
        if len(gov_records) > 0:
            primary = gov_records.iloc[0]
        else:
            primary = group.iloc[0]

        # Names / aliases
        name_values = set(group["Name"].dropna().tolist()) | set(group["AltName"].dropna().tolist())
        name_values = {n for n in name_values if n}  # remove empty
        aliases_list = sorted(list(name_values))

        canonical_name = primary.get("Name", "") or (aliases_list[0] if aliases_list else "")

        # District
        district_values = group["District"].dropna()
        if len(gov_records) > 0 and pd.notna(primary.get("District", None)):
            primary_district = primary["District"]
        elif len(district_values) > 0:
            primary_district = district_values.mode().iloc[0]
        else:
            primary_district = ""

        # Phones
        phone_values = group["Phone"].dropna().astype(str).tolist()
        phone_values = sorted(list(set([p for p in phone_values if p])))
        canonical_phones = " | ".join(phone_values)

        # Sources
        sources = sorted(list(group["SourceSystem"].dropna().unique()))
        sources_repr = " + ".join(sources)

        # Record IDs
        record_ids = sorted(list(group["RecordID"].astype(str).tolist()))
        record_ids_str = " | ".join(record_ids)

        # Merge attributes JSON (very basic union)
        merged_attrs = {}
        for attrs_json in group["Attributes_JSON"].dropna().tolist():
            try:
                d = json.loads(attrs_json)
                if isinstance(d, dict):
                    for k, v in d.items():
                        # Keep first non-null value per key
                        if k not in merged_attrs or merged_attrs[k] in [None, "", "null"]:
                            merged_attrs[k] = v
            except Exception:
                continue

        canonical_rows.append({
            "EntityID": cluster_id,          # cluster = canonical entity
            "EntityType": entity_type,
            "CanonicalName": canonical_name,
            "PrimaryDistrict": primary_district,
            "CanonicalPhones": canonical_phones,
            "Aliases": " | ".join(aliases_list),
            "SourcesRepresented": sources_repr,
            "RecordCount": len(group),
            "SourceRecordIDs": record_ids_str,
            "Attributes_JSON": json.dumps(merged_attrs, default=str),
        })

    canonical_df = pd.DataFrame(canonical_rows)
    return canonical_df

# ============================================================================
# MAIN PIPELINE (EARE-READY)
# ============================================================================

def run_vero_pipeline(gov_df=None, ngo_df=None, whatsapp_df=None, ground_truth_df=None, 
                     use_pretrained=False, pretrained_model=None, pretrained_scaler=None,
                     high_threshold=0.90, medium_threshold=0.75, district_threshold=0.75,
                     extra_entity_sources=None):
    """
    VERO Pipeline - EARE Multi-Entity Standard v4.0
    
    Implements:
    E = Entities (multi-type support via EntityType)
    A = Attributes (dynamic via Attributes_JSON)
    R = Relationships (scaffolded, placeholder)
    E = Events (scaffolded, placeholder)
    
    Backward compatible with facility-only usage.
    """
    
    print("="*70)
    print("VERO PIPELINE - EARE Multi-Entity Standard v4.0")
    print("="*70)
    
    # Stage 1: Raw Ingest & Normalization (EARE-ready)
    print("\n[1/6] Raw ingest & semantic normalization (EARE multi-entity)...")
    unified = prepare_unified_dataset(
        gov_df=gov_df,
        ngo_df=ngo_df,
        whatsapp_df=whatsapp_df,
        extra_entity_sources=extra_entity_sources  # NEW: multi-entity support
    )
    print(f"✅ Unified: {len(unified)} records")
    
    # Show entity type breakdown
    if "EntityType" in unified.columns:
        entity_counts = unified["EntityType"].value_counts()
        print("   Entity breakdown:")
        for ent_type, count in entity_counts.items():
            print(f"   - {ent_type}: {count}")
    
    # Stage 2: Embeddings
    print("\n[2/6] Computing embeddings...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(unified["name_clean"].tolist(), convert_to_tensor=True)
    id_to_emb = {rid: emb for rid, emb in zip(unified["RecordID"], embeddings)}
    print(f"✅ Embeddings computed")
    
    # Stage 3: Model Training or Loading
    if use_pretrained and pretrained_model and pretrained_scaler:
        print("\n[3/6] Using pre-trained model...")
        model = pretrained_model
        scaler = pretrained_scaler
        metrics = {"note": "Pre-trained model"}
    elif ground_truth_df is not None:
        print("\n[3/6] Training model from ground truth...")
        u_idx = unified.set_index("RecordID")
        
        features = []
        for _, row in ground_truth_df.iterrows():
            rec_a_id = str(row['Record_ID A'])
            rec_b_id = str(row['Record_ID B'])
            same_entity = 1 if str(row['Same Entity']).lower() == 'yes' else 0
            
            if rec_a_id in u_idx.index and rec_b_id in u_idx.index:
                rec_a = u_idx.loc[rec_a_id]
                rec_b = u_idx.loc[rec_b_id]
                
                feats = compute_pair_features(rec_a, rec_b, id_to_emb)
                feats['SameEntity'] = same_entity
                features.append(feats)
        
        similarity_features = pd.DataFrame(features)
        model, scaler, metrics = train_model(similarity_features)
        print(f"✅ Model trained - ROC-AUC: {metrics['roc_auc']:.3f}")
    else:
        raise ValueError("Must provide ground_truth_df or use_pretrained=True")
    
    # Stage 4: Candidate Generation (EARE entity-type aware)
    print("\n[4/6] Generating candidate pairs with blocking (entity-type aware)...")
    candidate_pairs = generate_candidate_pairs(unified)
    print(f"✅ Generated {len(candidate_pairs)} candidates")
    
    # Stage 5: Prediction
    print("\n[5/6] Computing features and predicting...")
    u_idx = unified.set_index("RecordID")
    
    rows = []
    for rid_a, rid_b in candidate_pairs:
        row_a = u_idx.loc[rid_a]
        row_b = u_idx.loc[rid_b]
        feats = compute_pair_features(row_a, row_b, id_to_emb)
        feats["record_A"] = rid_a
        feats["record_B"] = rid_b
        feats["source_A"] = row_a["SourceSystem"]
        feats["source_B"] = row_b["SourceSystem"]
        feats["name_A"] = row_a["Name"]
        feats["name_B"] = row_b["Name"]
        rows.append(feats)
    
    cand_features = pd.DataFrame(rows)
    
    X_cand = cand_features[FEATURE_COLS].values
    X_cand_scaled = scaler.transform(X_cand)
    probs = model.predict_proba(X_cand_scaled)[:, 1]
    cand_features["match_prob"] = probs
    
    # Two-tier threshold
    high_conf = cand_features[cand_features["match_prob"] >= high_threshold]
    medium_conf = cand_features[
        (cand_features["match_prob"] >= medium_threshold) &
        (cand_features["match_prob"] < high_threshold) &
        (cand_features["jw_name"] >= NAME_SIMILARITY_THRESHOLD)
    ]
    matched_pairs = pd.concat([high_conf, medium_conf], ignore_index=True)
    matched_pairs = matched_pairs.sort_values("match_prob", ascending=False)
    
    print(f"✅ Found {len(matched_pairs)} strong matches")
    print(f"   High confidence (>={high_threshold}): {len(high_conf)}")
    print(f"   Medium confidence ({medium_threshold}-{high_threshold}): {len(medium_conf)}")
    
    # Stage 6: Build EARE Tables
    print("\n[6/6] Building EARE standard tables...")
    
    # Build clusters (EARE-aware)
    clusters = build_clusters(matched_pairs, unified)
    
    # Build canonical entities (EARE multi-entity)
    canonical_entities = build_canonical_entities_table(clusters)
    
    print(f"✅ Created {len(canonical_entities)} canonical entities")
    
    # Show entity type breakdown
    if "EntityType" in canonical_entities.columns:
        entity_counts = canonical_entities["EntityType"].value_counts()
        print("   Canonical entity breakdown:")
        for ent_type, count in entity_counts.items():
            print(f"   - {ent_type}: {count}")
    
    print(f"✅ Schema Version: {SCHEMA_VERSION}")
    
    print("\n" + "="*70)
    print("VERO PIPELINE COMPLETE")
    print("="*70)
    
    return {
        "unified": unified,
        "clusters": clusters,
        "entity_clusters": clusters,  # Backward compatibility with app.py
        "matched_pairs": matched_pairs,
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        # NEW: EARE tables
        "canonical_entities": canonical_entities,
        "entity_relationships": pd.DataFrame(),  # Placeholder for future R
        "entity_events": pd.DataFrame(),         # Placeholder for future E
        "schema_version": SCHEMA_VERSION
    }
