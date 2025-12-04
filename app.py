"""
app.py
VERO.v3 - Streaming-Ready Entity Resolution Platform
Phase 1: FLEXIBLE COLUMN MAPPING - Works with ANY column names!
Upload → Map Columns → Batch Ingest → Rebuild Canonical → LLM Chat → Download
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

# Import modules
from vero_engine import run_vero_pipeline
from llm_backend import answer_entity_question, get_llm_status
from storage import init_storage, add_batch, load_all_raw, get_ingestion_history, get_storage_stats, clear_storage
from transformations import reshape_raw_to_logical_sources, enrich_with_ingest_metadata

st.set_page_config(
    page_title="VERO - Entity Resolution",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .mapping-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #b0d4f1;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_excel_file(uploaded_file, sheet_name):
    """Load specific sheet from Excel file"""
    try:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        return df
    except Exception as e:
        st.error(f"Error loading sheet '{sheet_name}': {str(e)}")
        return None

def apply_column_mapping(df, mapping):
    """
    Apply column mapping to dataframe.
    
    mapping: dict like {'RecordID': 'user_id_col', 'Name': 'user_name_col', ...}
    Returns: DataFrame with renamed columns
    """
    if df is None:
        return None
    
    df_mapped = df.copy()
    
    # Rename columns based on mapping
    reverse_mapping = {v: k for k, v in mapping.items() if v != "-- Skip --"}
    df_mapped = df_mapped.rename(columns=reverse_mapping)
    
    # Add missing required columns as None/empty
    required_cols = list(mapping.keys())
    for col in required_cols:
        if col not in df_mapped.columns:
            df_mapped[col] = None
    
    return df_mapped

def column_mapper_ui(df, source_name, default_mapping=None, source_key=None):
    """
    Streamlit UI component for mapping columns.
    
    Returns: dict of {standard_col: user_selected_col}
    """
    if df is None:
        return {}
    
    st.markdown(f'<div class="mapping-box">', unsafe_allow_html=True)
    
    # Check if we have saved mapping for this source
    saved_mapping = st.session_state.saved_mappings.get(source_key) if source_key else None
    
    # Header with saved mapping indicator
    col_header, col_action = st.columns([3, 1])
    with col_header:
        st.subheader(f"🔗 Map Columns: {source_name}")
    with col_action:
        if saved_mapping:
            st.caption("✅ Previous mapping available")
    
    # Option to use saved mapping
    use_saved = False
    if saved_mapping:
        col_use, col_clear = st.columns([3, 1])
        with col_use:
            use_saved = st.checkbox(
                "📋 Use previous mapping",
                value=st.session_state.use_saved_mapping.get(source_key, True),
                key=f"use_saved_{source_key}",
                help="Auto-fill with last used mapping"
            )
        with col_clear:
            if st.button("🗑️ Clear", key=f"clear_{source_key}", help="Clear saved mapping"):
                st.session_state.saved_mappings[source_key] = None
                st.rerun()
    
    # Define required fields and optional fields
    if source_name == "Government":
        required_fields = {
            'RecordID': 'Unique ID for each record',
            'OfficialFacilityName': 'Official facility name',
            'District': 'District/region location'
        }
        optional_fields = {
            'AltName': 'Alternative name',
            'Phone': 'Phone number',
            'GPS_Lat': 'GPS Latitude',
            'GPS_Lon': 'GPS Longitude'
        }
    elif source_name == "NGO":
        required_fields = {
            'RecordID': 'Unique ID for each record',
            'FacilityName': 'Facility name',
            'District': 'District/region location'
        }
        optional_fields = {
            'Phone': 'Phone number',
            'FarmerName': 'Person/farmer name',
            'Gender': 'Gender'
        }
    elif source_name == "WhatsApp":
        required_fields = {
            'RecordID': 'Unique ID for each record',
            'RelatedFacility': 'Related facility name',
            'DistrictNote': 'District/location note'
        }
        optional_fields = {
            'Phone': 'Phone number',
            'ContactName': 'Contact person name',
            'LocationNickname': 'Location nickname'
        }
    else:
        required_fields = {
            'RecordID': 'Unique ID',
            'Name': 'Entity name',
            'District': 'Location'
        }
        optional_fields = {}
    
    available_cols = ["-- Skip --"] + list(df.columns)
    mapping = {}
    
    # Determine which default mapping to use
    effective_mapping = saved_mapping if (use_saved and saved_mapping) else default_mapping
    
    # Required fields
    st.markdown("**Required Fields:**")
    cols_required = st.columns(len(required_fields))
    
    for idx, (field, description) in enumerate(required_fields.items()):
        with cols_required[idx]:
            # Try to find matching column
            default_idx = 0
            
            # Priority 1: Saved/default mapping
            if effective_mapping and field in effective_mapping:
                try:
                    default_idx = available_cols.index(effective_mapping[field])
                except ValueError:
                    default_idx = 0
            # Priority 2: Exact column name match
            elif field in df.columns:
                default_idx = available_cols.index(field)
            # Priority 3: Smart matching
            else:
                for col in df.columns:
                    if field.lower() in col.lower() or col.lower() in field.lower():
                        default_idx = available_cols.index(col)
                        break
            
            selected = st.selectbox(
                f"{field}",
                available_cols,
                index=default_idx,
                help=description,
                key=f"{source_name}_{field}_{source_key}"
            )
            mapping[field] = selected
    
    # Optional fields in expander
    if optional_fields:
        with st.expander("➕ Optional Fields (Click to Map)"):
            for field, description in optional_fields.items():
                # Try to find matching column
                default_idx = 0
                
                if effective_mapping and field in effective_mapping:
                    try:
                        default_idx = available_cols.index(effective_mapping[field])
                    except ValueError:
                        default_idx = 0
                elif field in df.columns:
                    default_idx = available_cols.index(field)
                else:
                    # Smart matching
                    for col in df.columns:
                        if field.lower() in col.lower() or col.lower() in field.lower():
                            default_idx = available_cols.index(col)
                            break
                
                selected = st.selectbox(
                    f"{field}",
                    available_cols,
                    index=default_idx,
                    help=description,
                    key=f"{source_name}_{field}_{source_key}"
                )
                if selected != "-- Skip --":
                    mapping[field] = selected
    
    # Save this mapping for future use
    if source_key and mapping:
        st.session_state.saved_mappings[source_key] = mapping.copy()
    
    # Show preview
    with st.expander("👁️ Preview Mapped Data"):
        mapped_df = apply_column_mapping(df, mapping)
        display_cols = [col for col in mapping.keys() if col in mapped_df.columns]
        st.dataframe(mapped_df[display_cols].head(3), use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return mapping

def validate_mapping(mapping, required_fields):
    """Validate that all required fields are mapped"""
    missing = []
    for field in required_fields:
        if field not in mapping or mapping[field] == "-- Skip --":
            missing.append(field)
    return missing

def to_excel(dataframes_dict):
    """Convert multiple dataframes to Excel with multiple sheets"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in dataframes_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("⚙️ Settings")
    
    st.markdown("---")
    
    # LLM Configuration
    st.subheader("🤖 LLM Backend")
    llm_status = get_llm_status()
    st.info(f"**Mode:** {llm_status['mode']}")
    
    if llm_status['mode'] == 'hf_inference':
        if llm_status['hf_token_set']:
            st.success("✅ HuggingFace token configured")
        else:
            st.warning("⚠️ HF_TOKEN not set")
        st.caption(f"Model: {llm_status['hf_model']}")
    elif llm_status['mode'] == 'local':
        st.info(f"Local: {llm_status['local_url']}")
    else:
        st.info("Mock mode (demo)")
    
    with st.expander("Configure LLM"):
        st.markdown("""
        **Set environment variables:**
        ```bash
        export VERO_LLM_MODE=hf_inference
        export HF_TOKEN=your_token_here
        ```
        """)
    
    st.markdown("---")
    
    st.subheader("Matching Thresholds")
    high_threshold = st.slider(
        "High Confidence", 0.7, 1.0, 0.90, 0.05,
        help="Matches above this are auto-accepted"
    )
    
    medium_threshold = st.slider(
        "Medium Confidence", 0.6, 0.9, 0.75, 0.05,
        help="Medium confidence requires strong name match"
    )
    
    st.markdown("---")
    
    st.subheader("Blocking Settings")
    district_threshold = st.slider(
        "District Match", 0.5, 1.0, 0.75, 0.05,
        help="Higher = stricter district matching"
    )
    
    st.markdown("---")
    
    # Storage Management
    st.subheader("📦 Storage")
    stats = get_storage_stats()
    st.metric("Total Batches", stats["total_batches"])
    st.metric("Total Records", stats["total_records"])
    
    if st.button("🗑️ Clear All Storage", help="Delete all ingested batches"):
        if st.session_state.get("confirm_clear"):
            clear_storage()
            st.session_state.results = None
            st.session_state.confirm_clear = False
            st.success("Storage cleared!")
            st.rerun()
        else:
            st.session_state.confirm_clear = True
            st.warning("Click again to confirm")

# ============================================================================
# MAIN HEADER
# ============================================================================

st.markdown('<div class="main-header">🏥 VERO Entity Resolution</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Streaming-Ready Canonical Identity Fabric | Phase 1: Flexible Column Mapping</div>', unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'confirm_clear' not in st.session_state:
    st.session_state.confirm_clear = False
if 'column_mappings' not in st.session_state:
    st.session_state.column_mappings = {}
if 'saved_mappings' not in st.session_state:
    # Persistent mappings across sessions (per source type)
    st.session_state.saved_mappings = {
        'gov': None,
        'ngo': None,
        'wa': None
    }
if 'use_saved_mapping' not in st.session_state:
    st.session_state.use_saved_mapping = {
        'gov': True,
        'ngo': True,
        'wa': True
    }

# Initialize storage
init_storage()

# ============================================================================
# STEP 1: FILE UPLOAD
# ============================================================================

st.header("📤 Step 1: Upload Your Data")
st.caption("✨ NEW: Works with ANY column names! Map your columns to VERO's expected fields.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Option A: Single Excel File")
    excel_file = st.file_uploader(
        "Upload Excel with multiple sheets",
        type=['xlsx', 'xls'],
        help="Can contain any sheet names - you'll select which to use"
    )
    
    if excel_file:
        try:
            excel_sheets = pd.ExcelFile(excel_file).sheet_names
            st.success(f"✅ Found {len(excel_sheets)} sheets: {', '.join(excel_sheets)}")
            
            gov_sheet = st.selectbox("Government Registry Sheet", excel_sheets, 
                                    index=excel_sheets.index('Government registry') if 'Government registry' in excel_sheets else 0)
            ngo_sheet = st.selectbox("NGO Dataset Sheet", excel_sheets,
                                    index=excel_sheets.index('NGO Dataset') if 'NGO Dataset' in excel_sheets else 0)
            wa_sheet = st.selectbox("WhatsApp Dataset Sheet", excel_sheets,
                                   index=excel_sheets.index('WhatsApp Dataset') if 'WhatsApp Dataset' in excel_sheets else 0)
            
            has_ground_truth = st.checkbox("Include Ground Truth for Training")
            gt_sheet = None
            if has_ground_truth:
                gt_sheet = st.selectbox("Ground Truth Sheet", excel_sheets,
                                       index=excel_sheets.index('Sankey GTP') if 'Sankey GTP' in excel_sheets else 0)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

with col2:
    st.subheader("Option B: Separate CSV Files")
    gov_csv = st.file_uploader("Government CSV", type=['csv'], key="gov")
    ngo_csv = st.file_uploader("NGO CSV", type=['csv'], key="ngo")
    wa_csv = st.file_uploader("WhatsApp CSV", type=['csv'], key="wa")
    gt_csv = st.file_uploader("Ground Truth CSV (optional)", type=['csv'], key="gt")

# ============================================================================
# STEP 2: COLUMN MAPPING (NEW!)
# ============================================================================

if excel_file or (gov_csv and ngo_csv and wa_csv):
    st.markdown("---")
    st.header("🔗 Step 2: Map Your Columns (NEW!)")
    st.info("🎯 VERO can now work with ANY column names! Just map your columns to the expected fields below.")
    
    # Load data
    if excel_file:
        gov_df = load_excel_file(excel_file, gov_sheet)
        ngo_df = load_excel_file(excel_file, ngo_sheet)
        wa_df = load_excel_file(excel_file, wa_sheet)
        gt_df = load_excel_file(excel_file, gt_sheet) if has_ground_truth and gt_sheet else None
    else:
        gov_df = pd.read_csv(gov_csv) if gov_csv else None
        ngo_df = pd.read_csv(ngo_csv) if ngo_csv else None
        wa_df = pd.read_csv(wa_csv) if wa_csv else None
        gt_df = pd.read_csv(gt_csv) if gt_csv else None
    
    # Column mapping UI
    if gov_df is not None:
        gov_mapping = column_mapper_ui(
            gov_df, 
            "Government",
            st.session_state.column_mappings.get('gov', None),
            source_key='gov'
        )
        st.session_state.column_mappings['gov'] = gov_mapping
    
    if ngo_df is not None:
        ngo_mapping = column_mapper_ui(
            ngo_df, 
            "NGO",
            st.session_state.column_mappings.get('ngo', None),
            source_key='ngo'
        )
        st.session_state.column_mappings['ngo'] = ngo_mapping
    
    if wa_df is not None:
        wa_mapping = column_mapper_ui(
            wa_df, 
            "WhatsApp",
            st.session_state.column_mappings.get('wa', None),
            source_key='wa'
        )
        st.session_state.column_mappings['wa'] = wa_mapping
    
    # Validate mappings
    st.markdown("---")
    st.subheader("✅ Validation")
    
    col_val1, col_val2, col_val3 = st.columns(3)
    
    with col_val1:
        gov_missing = validate_mapping(gov_mapping, ['RecordID', 'OfficialFacilityName', 'District'])
        if not gov_missing:
            st.success("✅ Government: All required fields mapped")
        else:
            st.error(f"❌ Government missing: {', '.join(gov_missing)}")
    
    with col_val2:
        ngo_missing = validate_mapping(ngo_mapping, ['RecordID', 'FacilityName', 'District'])
        if not ngo_missing:
            st.success("✅ NGO: All required fields mapped")
        else:
            st.error(f"❌ NGO missing: {', '.join(ngo_missing)}")
    
    with col_val3:
        wa_missing = validate_mapping(wa_mapping, ['RecordID', 'RelatedFacility', 'DistrictNote'])
        if not wa_missing:
            st.success("✅ WhatsApp: All required fields mapped")
        else:
            st.error(f"❌ WhatsApp missing: {', '.join(wa_missing)}")
    
    all_valid = not gov_missing and not ngo_missing and not wa_missing
    
    # ============================================================================
    # STEP 3: BATCH LABELING & INGESTION HISTORY
    # ============================================================================
    
    if all_valid:
        st.markdown("---")
        st.header("📊 Step 3: Batch Configuration & History")
        
        # Apply mappings
        gov_df_mapped = apply_column_mapping(gov_df, gov_mapping)
        ngo_df_mapped = apply_column_mapping(ngo_df, ngo_mapping)
        wa_df_mapped = apply_column_mapping(wa_df, wa_mapping)
        
        # Batch label input
        batch_label = st.text_input(
            "Batch label (for audit trail)",
            value="Initial_Load",
            help="Short name for this data drop, e.g., 'Morehouse_Malawi_Oct2025'"
        )
        
        # Show ingestion history
        with st.expander("📜 Ingestion History (Streaming Audit Trail)", expanded=True):
            history = get_ingestion_history()
            if len(history) == 0:
                st.caption("No ingested batches yet. This will show all historical uploads.")
            else:
                st.dataframe(history, use_container_width=True)
                st.caption(f"Total historical batches: {len(history)}")
        
        # Preview mapped data
        with st.expander("👁️ Preview Mapped Data", expanded=False):
            t1, t2, t3 = st.tabs(["Government", "NGO", "WhatsApp"])
            
            with t1:
                st.dataframe(gov_df_mapped.head(5), use_container_width=True)
                st.caption(f"Mapped batch: {len(gov_df_mapped)} records")
            
            with t2:
                st.dataframe(ngo_df_mapped.head(5), use_container_width=True)
                st.caption(f"Mapped batch: {len(ngo_df_mapped)} records")
            
            with t3:
                st.dataframe(wa_df_mapped.head(5), use_container_width=True)
                st.caption(f"Mapped batch: {len(wa_df_mapped)} records")
        
        # ============================================================================
        # STEP 4: INGEST & REBUILD CANONICAL REGISTRY
        # ============================================================================
        
        st.markdown("---")
        st.header("🚀 Step 4: Ingest & Rebuild Canonical Registry")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Batch: Government", len(gov_df_mapped))
        with col2:
            st.metric("Current Batch: NGO", len(ngo_df_mapped))
        with col3:
            st.metric("Current Batch: WhatsApp", len(wa_df_mapped))
        
        if st.button("📥 Ingest & Rebuild Canonical Registry", type="primary", use_container_width=True):
            with st.spinner("Ingesting batch and rebuilding canonical registry..."):
                try:
                    # 1) Add this batch into storage
                    add_batch("Gov", gov_df_mapped, batch_label=batch_label + "_GOV")
                    add_batch("NGO", ngo_df_mapped, batch_label=batch_label + "_NGO")
                    add_batch("WhatsApp", wa_df_mapped, batch_label=batch_label + "_WA")
                    
                    st.info("✅ Batch ingested into storage")
                    
                    # 2) Load ALL raw records (historical + new)
                    all_raw = load_all_raw()
                    st.info(f"📦 Loaded {len(all_raw)} total records from storage")
                    
                    # 3) Convert raw_records into logical sources
                    gov_all_df, ngo_all_df, wa_all_df = reshape_raw_to_logical_sources(all_raw)
                    st.info(f"🔄 Reshaped into Gov: {len(gov_all_df)}, NGO: {len(ngo_all_df)}, WA: {len(wa_all_df)}")
                    
                    # 4) Call pipeline on ALL data so far
                    results = run_vero_pipeline(
                        gov_df=gov_all_df,
                        ngo_df=ngo_all_df,
                        whatsapp_df=wa_all_df,
                        ground_truth_df=gt_df,
                        high_threshold=high_threshold,
                        medium_threshold=medium_threshold,
                        district_threshold=district_threshold
                    )
                    
                    # 5) Enrich entity_clusters with batch metadata
                    if "entity_clusters" in results and len(results["entity_clusters"]) > 0:
                        results["entity_clusters"] = enrich_with_ingest_metadata(
                            results["entity_clusters"],
                            all_raw
                        )
                    
                    st.session_state.results = results
                    st.success("✅ Ingestion complete and canonical registry rebuilt!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error during ingestion: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# ============================================================================
# STEP 5: DISPLAY RESULTS (Same as before)
# ============================================================================

if st.session_state.results:
    results = st.session_state.results
    
    st.markdown("---")
    st.header("📈 Step 5: Results & Insights")
    
    # Extract data
    canonical = results.get("canonical_entities", pd.DataFrame())
    matched = results.get("matched_pairs", pd.DataFrame())
    entity_clusters = results.get("entity_clusters", pd.DataFrame())
    unified = results.get("unified", pd.DataFrame())
    events = results.get("entity_events", pd.DataFrame())
    metrics = results.get("metrics", {})

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Canonical Entities", len(canonical))
    with col2:
        st.metric("Matched Pairs", len(matched))
    with col3:
        st.metric("Clusters", entity_clusters["EntityID"].nunique() if len(entity_clusters) > 0 else 0)
    with col4:
        st.metric("Model ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "🎯 Matched Pairs",
        "🧩 Canonical Entities",
        "📁 Download",
        "💬 LLM Chat",
        "🧪 Simulations & APIs"
    ])
    
    # ----------------------------------------------------------------------
    # TAB 1: OVERVIEW
    # ----------------------------------------------------------------------
    with tab1:
        st.subheader("Matching Overview")
        
        if len(matched) > 0 and "match_prob" in matched.columns:
            fig = px.histogram(
                matched, x='match_prob', nbins=20,
                title="Match Probability Distribution",
                labels={'match_prob': 'Probability', 'count': 'Pairs'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if len(entity_clusters) > 0 and "SourceSystem" in entity_clusters.columns:
            source_dist = entity_clusters["SourceSystem"].value_counts()
            fig = px.pie(
                values=source_dist.values,
                names=source_dist.index,
                title="Records by Source"
            )
            st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------------------------
    # TAB 2: MATCHED PAIRS
    # ----------------------------------------------------------------------
    with tab2:
        st.subheader("High-Confidence Matched Pairs")
        if len(matched) > 0:
            cols = ['record_A', 'record_B', 'name_A', 'name_B', 'source_A', 'source_B', 'match_prob']
            display_cols = [c for c in cols if c in matched.columns]
            st.dataframe(matched[display_cols].head(200), use_container_width=True)
        else:
            st.info("No matched pairs found")

    # ----------------------------------------------------------------------
    # TAB 3: CANONICAL ENTITIES
    # ----------------------------------------------------------------------
    with tab3:
        st.subheader("🧩 Canonical Entities (LLM-Ready Identity Table)")
        st.caption("One row per real-world entity, deduplicated across all ingested batches")
        
        if len(canonical) > 0:
            st.dataframe(canonical, use_container_width=True)
            st.caption(f"Total canonical entities: {len(canonical)}")
            
            if "EntityType" in canonical.columns:
                type_counts = canonical["EntityType"].value_counts()
                st.markdown("**Entity Types:**")
                for entity_type, count in type_counts.items():
                    st.write(f"- {entity_type}: {count}")
        else:
            st.info("No canonical entities. Run matching first.")

    # ----------------------------------------------------------------------
    # TAB 4: DOWNLOAD
    # ----------------------------------------------------------------------
    with tab4:
        st.subheader("📥 Download Results")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### Excel (All Sheets)")
            if st.button("Generate Excel", use_container_width=True):
                excel_data = to_excel({
                    'Canonical Entities': canonical,
                    'Matched Pairs': matched,
                    'Entity Clusters': entity_clusters,
                    'Unified Dataset': unified,
                })
                st.download_button(
                    "⬇️ Download Excel",
                    excel_data,
                    "vero_results.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        with col_right:
            st.markdown("### Individual CSVs")
            
            if len(canonical) > 0:
                st.download_button(
                    "Canonical Entities",
                    canonical.to_csv(index=False),
                    "canonical_entities.csv",
                    use_container_width=True
                )
            
            if len(matched) > 0:
                st.download_button(
                    "Matched Pairs",
                    matched.to_csv(index=False),
                    "matched_pairs.csv",
                    use_container_width=True
                )
            
            if len(entity_clusters) > 0:
                st.download_button(
                    "Entity Clusters",
                    entity_clusters.to_csv(index=False),
                    "entity_clusters.csv",
                    use_container_width=True
                )

    # ----------------------------------------------------------------------
    # TAB 5: LLM CHAT
    # ----------------------------------------------------------------------
    with tab5:
        st.subheader("💬 LLM Chat on Canonical Entities")
        
        llm_status = get_llm_status()
        if llm_status['mode'] == 'mock':
            st.info("🎭 Running in **mock mode** (demo). Set HF_TOKEN for real LLM.")
        elif llm_status['mode'] == 'hf_inference':
            if llm_status['hf_token_set']:
                st.success(f"🤗 Connected to **{llm_status['hf_model']}**")
            else:
                st.warning("⚠️ HF_TOKEN not set - using mock mode")

        if len(canonical) == 0:
            st.info("No canonical entities. Run matching first.")
        else:
            # Entity selector
            entity_options = []
            for _, row in canonical.iterrows():
                label = f"{row['EntityID']} | {row['CanonicalName']} ({row['EntityType']}, {row.get('PrimaryDistrict', 'N/A')})"
                entity_options.append(label)
            
            selected_label = st.selectbox(
                "Select an entity to query:",
                options=entity_options,
                help="Choose the canonical entity you want to ask about"
            )
            
            selected_entity_id = selected_label.split(" | ")[0]
            selected = canonical[canonical["EntityID"] == selected_entity_id].iloc[0]
            
            st.markdown(f"**Selected:** `{selected['CanonicalName']}` ({selected['EntityType']})")
            st.caption(
                f"ID: {selected['EntityID']} | District: {selected.get('PrimaryDistrict', 'N/A')} | "
                f"Sources: {selected.get('SourcesRepresented', 'N/A')}"
            )
            
            # Show context preview
            if len(entity_clusters) > 0:
                entity_cluster_rows = entity_clusters[entity_clusters["EntityID"] == selected_entity_id]
                
                with st.expander(f"📋 View {len(entity_cluster_rows)} source records (with batch audit)"):
                    show_cols = [c for c in ["RecordID", "SourceSystem", "SourceName", "SourceDistrict", "BatchLabel"] 
                                if c in entity_cluster_rows.columns]
                    st.dataframe(entity_cluster_rows[show_cols] if show_cols else entity_cluster_rows, 
                               use_container_width=True)
            
            user_question = st.text_area(
                "Your question:",
                placeholder="e.g., Summarize this facility's data quality, What sources contributed to this entity?, etc.",
                height=100
            )
            
            if st.button("🧠 Ask LLM", type="primary", disabled=not user_question.strip()):
                with st.spinner("Querying LLM backend..."):
                    answer = answer_entity_question(
                        entity_id=selected_entity_id,
                        question=user_question,
                        canonical_df=canonical,
                        entity_clusters_df=entity_clusters,
                        events_df=events
                    )
                
                st.markdown("### 🧠 LLM Answer")
                st.markdown(answer)

    # ----------------------------------------------------------------------
    # TAB 6: SIMULATIONS & APIS
    # ----------------------------------------------------------------------
    with tab6:
        st.subheader("🧪 Simulations & External APIs")
        st.info("Placeholder for future integrations: system dynamics, RWA oracles, external APIs")

        col_sim, col_api = st.columns(2)

        with col_sim:
            st.markdown("### 📊 Simulations")
            st.button("Run Scenario Simulation", disabled=True)
            st.button("Stress Test: Facility Overload", disabled=True)
            st.button("Forecast Stockouts", disabled=True)

        with col_api:
            st.markdown("### 🌐 External APIs")
            st.button("Sync External Registry", disabled=True)
            st.button("Pull RWA Oracle Data", disabled=True)
            st.button("Push to Partner System", disabled=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>VERO - Entity Resolution Platform v4.0.4 + Phase 1 Column Mapping</strong></p>
    <p>Streaming-Ready Canonical Identity Fabric | Works with ANY column names!</p>
    <p>© 2025</p>
</div>
""", unsafe_allow_html=True)
