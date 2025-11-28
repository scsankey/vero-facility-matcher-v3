"""
vero_engine.py
VERO Entity Resolution Engine - Dynamic Identity & Context Standard v3.0

Core Principles:
1. Stable EntityID + EntityType = backbone
2. Golden tables per type = cleaned identity
3. CanonicalEntities + ExtraAttributes = future-proof schema
4. Clusters = full raw lineage
5. Events = evolving context & multimodal meaning
6. LLM only reads canonical + curated events
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
# STAGE 1: RAW INGEST & NORMALIZATION
# ============================================================================

def prepare_unified_dataset(gov_df, ngo_df, whatsapp_df):
    """
    Stage 1: Raw Ingest & Semantic Normalization
    Maps diverse inputs to standard shape while preserving raw data
    """
    timestamp = datetime.now().isoformat()
    
    # Government Registry
    gov_prepared = pd.DataFrame({
        'RecordID': gov_df['RecordID'].astype(str),
        'Name': gov_df['OfficialFacilityName'],
        'AltName': gov_df.get('AltName', None),
        'District': gov_df['District'],
        'Phone': None,
        'EntityType': 'facility',
        'SourceSystem': 'Gov',
        'SourceMeta': gov_df.apply(lambda r: json.dumps({
            'original_row': r.to_dict(),
            'ingested_at': timestamp
        }, default=str), axis=1),
        'SchemaVersion': SCHEMA_VERSION
    })
    
    # NGO Dataset
    ngo_prepared = pd.DataFrame({
        'RecordID': ngo_df['RecordID'].astype(str),
        'Name': ngo_df['FacilityName'],
        'AltName': None,
        'District': ngo_df['District'],
        'Phone': ngo_df.get('Phone', None),
        'EntityType': 'facility',
        'SourceSystem': 'NGO',
        'SourceMeta': ngo_df.apply(lambda r: json.dumps({
            'original_row': r.to_dict(),
            'ingested_at': timestamp
        }, default=str), axis=1),
        'SchemaVersion': SCHEMA_VERSION
    })
    
    # WhatsApp Dataset
    whatsapp_prepared = pd.DataFrame({
        'RecordID': whatsapp_df['RecordID'].astype(str),
        'Name': whatsapp_df['RelatedFacility'],
        'AltName': whatsapp_df.get('LocationNickname', None),
        'District': whatsapp_df['DistrictNote'],
        'Phone': whatsapp_df.get('Phone', None),
        'EntityType': 'facility',
        'SourceSystem': 'WhatsApp',
        'SourceMeta': whatsapp_df.apply(lambda r: json.dumps({
            'original_row': r.to_dict(),
            'ingested_at': timestamp
        }, default=str), axis=1),
        'SchemaVersion': SCHEMA_VERSION
    })
    
    # Combine all sources
    unified = pd.concat([gov_prepared, ngo_prepared, whatsapp_prepared], ignore_index=True)
    
    # Normalized fields for matching
    unified["name_clean"] = (
        unified["Name"].fillna("") + " " + unified["AltName"].fillna("")
    ).str.strip().str.lower()
    unified["alt_a_clean"] = unified["Name"].fillna("").str.lower()
    unified["alt_b_clean"] = unified["AltName"].fillna("").str.lower()
    unified["district_clean"] = unified["District"].fillna("").str.lower().str.strip()
    unified["phone_clean"] = unified["Phone"].fillna("").astype(str).str.replace(r"\D", "", regex=True)
    
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
# STAGE 3: BLOCKING & CANDIDATE GENERATION
# ============================================================================

def should_compare(rid_a, rid_b, row_a, row_b, district_threshold=DISTRICT_BLOCKING_THRESHOLD):
    """Blocking logic - requires at least 2 matching conditions"""
    if rid_a == rid_b:
        return False
    
    conditions_met = 0
    
    # District match
    d_a = row_a["district_clean"]
    d_b = row_b["district_clean"]
    if d_a and d_b:
        district_score = fuzz.token_set_ratio(d_a, d_b) / 100.0
        if district_score >= district_threshold:
            conditions_met += 1
    
    # Phone last4 match
    p_a = row_a["phone_clean"]
    p_b = row_b["phone_clean"]
    last4_a = p_a[-4:] if len(p_a) >= 4 else ""
    last4_b = p_b[-4:] if len(p_b) >= 4 else ""
    if last4_a and last4_a == last4_b:
        conditions_met += 1
    
    # Meaningful name token overlap
    common_words = {'health', 'centre', 'center', 'hospital', 'district', 
                   'rural', 'clinic', 'post', 'community', 'area'}
    tokens_a = set(row_a["name_clean"].split()) - common_words
    tokens_b = set(row_b["name_clean"].split()) - common_words
    
    if len(tokens_a & tokens_b) > 0:
        conditions_met += 1
    
    # Facility type must match (hard block)
    type_a = extract_facility_type(row_a["Name"])
    type_b = extract_facility_type(row_b["Name"])
    if type_a != 'unknown' and type_b != 'unknown':
        if type_a != type_b:
            return False
    
    return conditions_met >= 2

def generate_candidate_pairs(unified_df):
    """Generate candidate pairs using blocking"""
    u_idx = unified_df.set_index("RecordID")
    
    ngo_ids = unified_df[unified_df["SourceSystem"] == "NGO"]["RecordID"].tolist()
    gov_ids = unified_df[unified_df["SourceSystem"] == "Gov"]["RecordID"].tolist()
    wa_ids = unified_df[unified_df["SourceSystem"] == "WhatsApp"]["RecordID"].tolist()
    
    candidate_pairs = []
    
    for rid_ng in ngo_ids:
        row_ng = u_idx.loc[rid_ng]
        for rid_g in gov_ids:
            row_g = u_idx.loc[rid_g]
            if should_compare(rid_ng, rid_g, row_ng, row_g):
                candidate_pairs.append((rid_ng, rid_g))
    
    for rid_ng in ngo_ids:
        row_ng = u_idx.loc[rid_ng]
        for rid_w in wa_ids:
            row_w = u_idx.loc[rid_w]
            if should_compare(rid_ng, rid_w, row_ng, row_w):
                candidate_pairs.append((rid_ng, rid_w))
    
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
# STAGE 5: CLUSTER BUILDING (Full Raw Lineage)
# ============================================================================

def build_entity_clusters(matched_pairs_df, unified_df):
    """
    Build entity_clusters table preserving full raw lineage
    ADIP Principle: Clusters = full raw lineage
    """
    G = nx.Graph()
    for _, row in matched_pairs_df.iterrows():
        G.add_edge(row["record_A"], row["record_B"])
    
    clusters = list(nx.connected_components(G))
    
    u_idx = unified_df.set_index("RecordID")
    cluster_rows = []
    entity_id_counter = 1
    
    # Create EntityIDs for clusters
    for i, comp in enumerate(clusters, start=1):
        entity_id = f"Facility_{entity_id_counter}"
        entity_id_counter += 1
        
        for rid in comp:
            row = u_idx.loc[rid]
            cluster_rows.append({
                "EntityID": entity_id,
                "ClusterID": f"Cluster_{i}",
                "RecordID": rid,
                "SourceSystem": row["SourceSystem"],
                "SourceName": row["Name"],
                "SourceDistrict": row["District"],
                "SourcePhone": row["Phone"],
                "SourceMeta": row["SourceMeta"],
                "EntityType": row["EntityType"],
                "SchemaVersion": SCHEMA_VERSION
            })
    
    # Handle singletons
    all_matched_ids = set(matched_pairs_df["record_A"]) | set(matched_pairs_df["record_B"])
    singleton_ids = set(unified_df["RecordID"]) - all_matched_ids
    
    for rid in singleton_ids:
        row = u_idx.loc[rid]
        entity_id = f"Facility_{entity_id_counter}"
        entity_id_counter += 1
        
        cluster_rows.append({
            "EntityID": entity_id,
            "ClusterID": f"Singleton_{rid}",
            "RecordID": rid,
            "SourceSystem": row["SourceSystem"],
            "SourceName": row["Name"],
            "SourceDistrict": row["District"],
            "SourcePhone": row["Phone"],
            "SourceMeta": row["SourceMeta"],
            "EntityType": row["EntityType"],
            "SchemaVersion": SCHEMA_VERSION
        })
    
    return pd.DataFrame(cluster_rows)

# ============================================================================
# STAGE 6: GOLDEN TABLES (Cleaned Identity per Type)
# ============================================================================

def build_golden_tables(entity_clusters_df: pd.DataFrame) -> dict:
    """
    Build golden tables per entity type
    ADIP Principle: Golden tables per type = cleaned identity
    """
    if entity_clusters_df is None or len(entity_clusters_df) == 0:
        return {}

    entity_clusters_df = entity_clusters_df.copy()
    golden_tables = {}
    timestamp = datetime.now().isoformat()

    for entity_type, group in entity_clusters_df.groupby("EntityType"):
        rows = []
        
        for entity_id, cluster_rows in group.groupby("EntityID"):
            # Canonical name: most frequent
            canonical_name = None
            if "SourceName" in cluster_rows.columns:
                non_null = cluster_rows["SourceName"].dropna().astype(str)
                if len(non_null) > 0:
                    canonical_name = non_null.value_counts().idxmax()

            # Aliases: all unique names
            alias_values = cluster_rows["SourceName"].dropna().astype(str).unique().tolist()
            aliases_str = "; ".join(sorted(set(alias_values)))

            # Primary district: most frequent
            primary_district = None
            if "SourceDistrict" in cluster_rows.columns:
                non_null_dist = cluster_rows["SourceDistrict"].dropna().astype(str)
                if len(non_null_dist) > 0:
                    primary_district = non_null_dist.value_counts().idxmax()

            # Sources represented
            sources = cluster_rows["SourceSystem"].dropna().unique().tolist()
            sources_str = ", ".join(sorted(sources))
            primary_source = sources[0] if sources else None

            # Extra attributes (future-proofing)
            extra_attrs = {
                "phones": cluster_rows["SourcePhone"].dropna().unique().tolist(),
                "all_districts": cluster_rows["SourceDistrict"].dropna().unique().tolist(),
            }

            rows.append({
                "EntityID": entity_id,
                "EntityType": entity_type,
                "CanonicalName": canonical_name,
                "Aliases": aliases_str,
                "PrimaryDistrict": primary_district,
                "PrimarySource": primary_source,
                "SourcesRepresented": sources_str,
                "RecordCount": len(cluster_rows),
                "ExtraAttributes": json.dumps(extra_attrs),
                "Status": "active",
                "DataQualityScore": min(1.0, len(cluster_rows) * 0.2),  # Simple heuristic
                "SchemaVersion": SCHEMA_VERSION,
                "CreatedAt": timestamp,
                "UpdatedAt": timestamp
            })

        golden_df = pd.DataFrame(rows)

        if entity_type == "facility":
            key = "golden_facilities"
        elif entity_type == "person":
            key = "golden_persons"
        elif entity_type == "district":
            key = "golden_districts"
        else:
            key = f"golden_{entity_type}s"

        golden_tables[key] = golden_df

    return golden_tables

# ============================================================================
# STAGE 7: CANONICAL ENTITIES (Unified View)
# ============================================================================

def build_canonical_entities_table(golden_tables: dict) -> pd.DataFrame:
    """
    Build canonical_entities table from all golden tables
    ADIP Principle: CanonicalEntities + ExtraAttributes = future-proof schema
    """
    dfs = []
    for key, df in golden_tables.items():
        if df is None or len(df) == 0:
            continue
        
        tmp = df.copy()
        required_cols = [
            "EntityID", "EntityType", "CanonicalName", "Aliases",
            "PrimaryDistrict", "PrimarySource", "SourcesRepresented",
            "RecordCount", "ExtraAttributes", "Status", "DataQualityScore",
            "SchemaVersion", "CreatedAt", "UpdatedAt"
        ]
        
        for col in required_cols:
            if col not in tmp.columns:
                tmp[col] = None
        
        dfs.append(tmp[required_cols])

    if not dfs:
        return pd.DataFrame(columns=[
            "EntityID", "EntityType", "CanonicalName", "Aliases",
            "PrimaryDistrict", "PrimarySource", "SourcesRepresented",
            "RecordCount", "ExtraAttributes", "Status", "DataQualityScore",
            "SchemaVersion", "CreatedAt", "UpdatedAt"
        ])

    return pd.concat(dfs, ignore_index=True)

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_vero_pipeline(gov_df, ngo_df, whatsapp_df, ground_truth_df=None, 
                     use_pretrained=False, pretrained_model=None, pretrained_scaler=None,
                     high_threshold=0.90, medium_threshold=0.75, district_threshold=0.75):
    """
    VERO Pipeline - ADIP Dynamic Identity & Context Standard v1.0
    
    Implements:
    1. Stable EntityID + EntityType backbone
    2. Golden tables per type
    3. CanonicalEntities with ExtraAttributes
    4. Full cluster lineage
    5. Future-proof for events & multimodal
    """
    
    print("="*70)
    print("VERO PIPELINE - ADIP Standard v1.0")
    print("="*70)
    
    # Stage 1: Raw Ingest & Normalization
    print("\n[1/6] Raw ingest & semantic normalization...")
    unified = prepare_unified_dataset(gov_df, ngo_df, whatsapp_df)
    print(f"✅ Unified: {len(unified)} records")
    
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
    
    # Stage 4: Candidate Generation
    print("\n[4/6] Generating candidate pairs with blocking...")
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
    
    # Two-tier threshold (use passed parameters)
    high_conf = cand_features[cand_features["match_prob"] >= high_threshold]
    medium_conf = cand_features[
        (cand_features["match_prob"] >= medium_threshold) &
        (cand_features["match_prob"] < high_threshold) &
        (cand_features["jw_name"] >= 0.85)  # Name similarity threshold
    ]
    matched_pairs = pd.concat([high_conf, medium_conf], ignore_index=True)
    matched_pairs = matched_pairs.sort_values("match_prob", ascending=False)
    
    print(f"✅ Found {len(matched_pairs)} strong matches")
    print(f"   High confidence (>={high_threshold}): {len(high_conf)}")
    print(f"   Medium confidence ({medium_threshold}-{high_threshold}): {len(medium_conf)}")
    
    # Stage 6: Build ADIP Standard Tables
    print("\n[6/6] Building ADIP standard tables...")
    entity_clusters = build_entity_clusters(matched_pairs, unified)
    golden_tables = build_golden_tables(entity_clusters)
    canonical_entities = build_canonical_entities_table(golden_tables)
    
    print(f"✅ Created {len(canonical_entities)} canonical entities")
    print(f"✅ Schema Version: {SCHEMA_VERSION}")
    
    print("\n" + "="*70)
    print("VERO PIPELINE COMPLETE")
    print("="*70)
    
    return {
        "unified": unified,
        "entity_clusters": entity_clusters,
        "clusters": entity_clusters,  # Backward compatibility
        "golden_facilities": golden_tables.get("golden_facilities", pd.DataFrame()),
        "golden_persons": golden_tables.get("golden_persons", pd.DataFrame()),
        "golden_districts": golden_tables.get("golden_districts", pd.DataFrame()),
        "canonical_entities": canonical_entities,
        "matched_pairs": matched_pairs,
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "schema_version": SCHEMA_VERSION
    }
