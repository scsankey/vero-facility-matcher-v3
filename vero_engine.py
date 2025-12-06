"""
app.py
VERO v4.0.4 - Complete Entity Resolution Platform
✅ Phase 1: Flexible Column Mapping
✅ Mapping Memory (Auto-fill saved mappings)
✅ Multi-Entity Support (Facilities + Persons)
✅ Enhanced Entity Display
✅ Smart Auto-Mapping with HITL

Upload → Map Columns → Multi-Entity → Batch Ingest → Rebuild Canonical → LLM Chat → Download
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
import json

# Import modules
from vero_engine import run_vero_pipeline
from llm_backend import answer_entity_question, get_llm_status
from storage import init_storage, add_batch, load_all_raw, get_ingestion_history, get_storage_stats, clear_storage
from transformations import reshape_raw_to_logical_sources, enrich_with_ingest_metadata
from smart_mapper import smart_auto_map, get_mapping_summary, SmartColumnMapper

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
    .entity-badge {
        display: inline-block;
        padding: 0.3rem 0.6rem;
        border-radius: 0.3rem;
        font-size: 0.9rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    .badge-facility {
        background-color: #e3f2fd;
        color: #1976d2;
    }
    .badge-person {
        background-color: #f3e5f5;
        color: #7b1fa2;
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

def column_mapper_ui(df, source_name, field_config, source_key=None):
    """
    Smart column mapping UI with auto-detection and HITL editing
    
    Features:
    - Auto-maps columns with high confidence
    - Shows confidence scores
    - Allows editing of mappings
    - Visual indicators for match quality
    """
    if df is None:
        return {}
    
    st.markdown(f'<div class="mapping-box">', unsafe_allow_html=True)
    
    # Check if we have saved mapping for this source
    saved_mapping = st.session_state.saved_mappings.get(source_key) if source_key else None
    
    # Header
    st.subheader(f"🔗 Map Columns: {source_name}")
    
    required_fields = field_config.get('required', {})
    optional_fields = field_config.get('optional', {})
    
    # Determine which mapping to use as starting point
    if saved_mapping:
        # Use saved mapping
        st.info("📋 Using previously saved mapping (you can edit below)")
        initial_mapping = saved_mapping
        confidence_scores = {k: 100 for k in saved_mapping.keys()}  # Saved = 100% confidence
        match_methods = {k: "saved_mapping" for k in saved_mapping.keys()}
    else:
        # Smart auto-map
        with st.spinner("🤖 Auto-detecting column mappings..."):
            initial_mapping, confidence_scores, match_methods = smart_auto_map(
                df, 
                required_fields, 
                optional_fields,
                confidence_threshold=70
            )
        
        # Show auto-mapping summary
        summary = get_mapping_summary(initial_mapping, confidence_scores, match_methods)
        
        if summary['mapping_rate'] >= 80:
            st.success(f"✅ Auto-mapped {summary['auto_mapped']}/{summary['total_fields']} columns ({summary['mapping_rate']:.0f}%)")
        elif summary['mapping_rate'] >= 50:
            st.warning(f"⚠️ Auto-mapped {summary['auto_mapped']}/{summary['total_fields']} columns ({summary['mapping_rate']:.0f}%) - Please review")
        else:
            st.info(f"ℹ️ Auto-mapped {summary['auto_mapped']}/{summary['total_fields']} columns ({summary['mapping_rate']:.0f}%) - Manual mapping needed")
    
    # HITL Preview & Edit Section
    st.markdown("---")
    st.markdown("### 🎯 Review & Edit Mappings")
    st.caption("Auto-detected mappings shown below. Edit if needed.")
    
    available_cols = ["-- Skip --"] + list(df.columns)
    mapping = {}
    
    # Create editable mapping table
    col1, col2, col3, col4 = st.columns([2, 3, 2, 1])
    
    with col1:
        st.markdown("**Standard Field**")
    with col2:
        st.markdown("**Your Column**")
    with col3:
        st.markdown("**Confidence**")
    with col4:
        st.markdown("**Required**")
    
    st.markdown("---")
    
    # Required fields first
    st.markdown("**📌 Required Fields:**")
    for field, description in required_fields.items():
        col1, col2, col3, col4 = st.columns([2, 3, 2, 1])
        
        with col1:
            st.markdown(f"`{field}`")
            st.caption(description)
        
        with col2:
            # Get initial value
            initial_value = initial_mapping.get(field, "-- Skip --")
            try:
                default_idx = available_cols.index(initial_value)
            except ValueError:
                default_idx = 0
            
            # Editable dropdown
            selected = st.selectbox(
                f"Map {field}",
                available_cols,
                index=default_idx,
                key=f"smart_map_{source_key}_{field}",
                label_visibility="collapsed"
            )
            mapping[field] = selected
        
        with col3:
            # Show confidence score
            score = confidence_scores.get(field, 0)
            method = match_methods.get(field, "unknown")
            
            if score >= 90:
                st.success(f"✅ {score:.0f}%")
            elif score >= 70:
                st.warning(f"⚠️ {score:.0f}%")
            elif selected != "-- Skip --":
                st.info(f"ℹ️ Manual")
            else:
                st.error(f"❌ {score:.0f}%")
            
            # Show match method as tooltip
            if method == "exact_match":
                st.caption("🎯 Exact")
            elif method == "synonym_match":
                st.caption("📚 Synonym")
            elif "fuzzy" in method:
                st.caption("🔍 Fuzzy")
            elif method == "saved_mapping":
                st.caption("💾 Saved")
        
        with col4:
            st.markdown("✅")
    
    # Optional fields in expander
    if optional_fields:
        with st.expander("➕ Optional Fields (Click to review/edit)"):
            for field, description in optional_fields.items():
                col1, col2, col3 = st.columns([2, 3, 2])
                
                with col1:
                    st.markdown(f"`{field}`")
                    st.caption(description)
                
                with col2:
                    initial_value = initial_mapping.get(field, "-- Skip --")
                    try:
                        default_idx = available_cols.index(initial_value)
                    except ValueError:
                        default_idx = 0
                    
                    selected = st.selectbox(
                        f"Map {field}",
                        available_cols,
                        index=default_idx,
                        key=f"smart_map_{source_key}_{field}_opt",
                        label_visibility="collapsed"
                    )
                    if selected != "-- Skip --":
                        mapping[field] = selected
                
                with col3:
                    score = confidence_scores.get(field, 0)
                    if selected != "-- Skip --":
                        if score >= 90:
                            st.success(f"✅ {score:.0f}%")
                        elif score >= 70:
                            st.warning(f"⚠️ {score:.0f}%")
                        else:
                            st.info(f"ℹ️ Manual")
    
    # Save this mapping for future use
    if source_key and mapping:
        st.session_state.saved_mappings[source_key] = mapping.copy()
    
    # Show preview of mapped data
    st.markdown("---")
    with st.expander("👁️ Preview Mapped Data"):
        mapped_df = apply_column_mapping(df, mapping)
        display_cols = [col for col in mapping.keys() if col in mapped_df.columns]
        
        if display_cols:
            st.dataframe(mapped_df[display_cols].head(3), use_container_width=True)
            
            # Show mapping summary
            st.caption(f"**Mapped:** {len([v for v in mapping.values() if v != '-- Skip --'])} columns | "
                      f"**Skipped:** {len([v for v in mapping.values() if v == '-- Skip --'])} columns")
        else:
            st.warning("No columns mapped yet")
    
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

st.markdown('<div class="main-header">🏥 VERO Entity Resolution v4.0.4</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">🎯 Complete Platform: Flexible Columns + Multi-Entity + Smart Auto-Mapping</div>', unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'confirm_clear' not in st.session_state:
    st.session_state.confirm_clear = False
if 'column_mappings' not in st.session_state:
    st.session_state.column_mappings = {}
if 'saved_mappings' not in st.session_state:
    st.session_state.saved_mappings = {
        'gov': None,
        'ngo': None,
        'wa': None,
        'person_ngo': None,
        'person_wa': None
    }
if 'use_saved_mapping' not in st.session_state:
    st.session_state.use_saved_mapping = {
        'gov': True,
        'ngo': True,
        'wa': True,
        'person_ngo': True,
        'person_wa': True
    }
if 'enable_multi_entity' not in st.session_state:
    st.session_state.enable_multi_entity = False

# Initialize storage
init_storage()

# ============================================================================
# STEP 1: FILE UPLOAD
# ============================================================================

st.header("📤 Step 1: Upload Your Data")
st.caption("✨ Works with ANY column names! ✨ Supports Multi-Entity (Facilities + Persons)")

# Multi-entity toggle
enable_multi_entity = st.checkbox(
    "🔥 Enable Multi-Entity Matching (Facilities + Persons/Farmers)",
    value=st.session_state.enable_multi_entity,
    help="Check this to match both facilities AND persons (farmers, contacts, etc.)"
)
st.session_state.enable_multi_entity = enable_multi_entity

if enable_multi_entity:
    st.info("🎯 Multi-Entity Mode: Will match facilities AND persons across your data sources")

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
# STEP 2: COLUMN MAPPING
# ============================================================================

if excel_file or (gov_csv and ngo_csv and wa_csv):
    st.markdown("---")
    st.header("🔗 Step 2: Map Your Columns")
    
    with st.expander("📋 Click to Map Columns", expanded=True):
        st.info("🎯 VERO can work with ANY column names! Map your columns to expected fields below.")
        
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
        
        # Ground truth column mapping (if provided)
        gt_mapping = None
        if gt_df is not None:
            st.markdown("---")
            st.subheader("🎯 Ground Truth Mapping")
            st.info("💡 Map your ground truth columns - RecordIDs will be automatically matched to sources")
            
            gt_mapping = column_mapper_ui(
                gt_df,
                "Ground Truth",
                {
                    'required': {
                        'record_A': 'First record ID (e.g., G1, N5, W12)',
                        'record_B': 'Second record ID (e.g., N8, G3, W15)',
                        'label': 'Match label (yes/1 = match, no/0 = not match)'
                    },
                    'optional': {}
                },
                source_key='ground_truth'
            )
            
            # Apply ground truth mapping
            gt_df_mapped = apply_column_mapping(gt_df, gt_mapping)
            
            # CRITICAL FIX: Convert ground truth RecordIDs to match the prefixed format
            # The unified dataset uses: facility_Gov_G1, facility_NGO_N5, etc.
            # But ground truth has: G1, N5, etc.
            # We need to add prefixes based on the RecordID pattern
            
            def add_record_prefix(record_id):
                """Add entity_type_source prefix to match unified dataset format"""
                record_id = str(record_id).strip()
                
                # Already has prefix? Return as-is
                if '_' in record_id and any(record_id.startswith(f"{t}_") for t in ['facility', 'person', 'farm']):
                    return record_id
                
                # Detect source from RecordID pattern
                # G1-G999 = Gov, N1-N999 = NGO, W1-W999 = WhatsApp
                if record_id.startswith('G'):
                    return f"facility_Gov_{record_id}"
                elif record_id.startswith('N'):
                    return f"facility_NGO_{record_id}"
                elif record_id.startswith('W'):
                    return f"facility_WhatsApp_{record_id}"
                else:
                    # Unknown pattern, try to find in unified dataset
                    return record_id
            
            # Add prefixes to ground truth RecordIDs
            gt_df_mapped['record_A'] = gt_df_mapped['record_A'].apply(add_record_prefix)
            gt_df_mapped['record_B'] = gt_df_mapped['record_B'].apply(add_record_prefix)
            
            # Standardize label column to 1/0 integers
            def standardize_label(val):
                """Convert label to 1 (match) or 0 (no match)"""
                val_str = str(val).lower().strip()
                if val_str in ['yes', '1', 'true', 'match', 'same']:
                    return 1
                elif val_str in ['no', '0', 'false', 'different', 'not']:
                    return 0
                else:
                    # Try to convert to int
                    try:
                        return int(float(val))
                    except:
                        return 0
            
            gt_df_mapped['label'] = gt_df_mapped['label'].apply(standardize_label)
            
            # Show preview of mapped ground truth
            with st.expander("👁️ Preview Mapped Ground Truth"):
                preview_df = gt_df_mapped[['record_A', 'record_B', 'label']].head(5).copy()
                st.dataframe(preview_df, use_container_width=True)
                st.caption(f"✅ Mapped {len(gt_df_mapped)} ground truth pairs")
                st.caption(f"   Matches: {(gt_df_mapped['label'] == 1).sum()} | Non-matches: {(gt_df_mapped['label'] == 0).sum()}")
            
            # Use mapped version
            gt_df = gt_df_mapped
        
        # Column mapping UI for facilities
        st.subheader("🏢 Facility Data Mapping")
        
        if gov_df is not None:
            gov_mapping = column_mapper_ui(
                gov_df, 
                "Government",
                {
                    'required': {
                        'RecordID': 'Unique ID for each record',
                        'OfficialFacilityName': 'Official facility name',
                        'District': 'District/region location'
                    },
                    'optional': {
                        'AltName': 'Alternative name',
                        'Phone': 'Phone number',
                        'GPS_Lat': 'GPS Latitude',
                        'GPS_Lon': 'GPS Longitude'
                    }
                },
                source_key='gov'
            )
            st.session_state.column_mappings['gov'] = gov_mapping
        
        if ngo_df is not None:
            ngo_mapping = column_mapper_ui(
                ngo_df, 
                "NGO",
                {
                    'required': {
                        'RecordID': 'Unique ID for each record',
                        'FacilityName': 'Facility name',
                        'District': 'District/region location'
                    },
                    'optional': {
                        'Phone': 'Phone number',
                        'FarmerName': 'Person/farmer name (for multi-entity)',
                        'Gender': 'Gender (for persons)'
                    }
                },
                source_key='ngo'
            )
            st.session_state.column_mappings['ngo'] = ngo_mapping
        
        if wa_df is not None:
            wa_mapping = column_mapper_ui(
                wa_df, 
                "WhatsApp",
                {
                    'required': {
                        'RecordID': 'Unique ID for each record',
                        'RelatedFacility': 'Related facility name',
                        'DistrictNote': 'District/location note'
                    },
                    'optional': {
                        'Phone': 'Phone number',
                        'ContactName': 'Contact person name (for multi-entity)',
                        'LocationNickname': 'Location nickname'
                    }
                },
                source_key='wa'
            )
            st.session_state.column_mappings['wa'] = wa_mapping
        
        # Multi-entity person mapping
        person_ngo_mapping = None
        person_wa_mapping = None
        
        if enable_multi_entity:
            st.markdown("---")
            st.subheader("👤 Person Data Mapping (Multi-Entity)")
            st.info("Map person/farmer fields from your data sources")
            
            # Check if NGO has person fields
            if ngo_df is not None and ('FarmerName' in ngo_df.columns or any('name' in col.lower() for col in ngo_df.columns if col not in ['FacilityName'])):
                person_ngo_mapping = column_mapper_ui(
                    ngo_df,
                    "NGO Persons",
                    {
                        'required': {
                            'RecordID': 'Unique ID',
                            'PersonName': 'Person/Farmer name',
                            'District': 'District/location'
                        },
                        'optional': {
                            'Phone': 'Phone number',
                            'Gender': 'Gender',
                            'Role': 'Role/occupation'
                        }
                    },
                    source_key='person_ngo'
                )
            
            # Check if WhatsApp has person fields
            if wa_df is not None and ('ContactName' in wa_df.columns or any('contact' in col.lower() or 'person' in col.lower() for col in wa_df.columns)):
                person_wa_mapping = column_mapper_ui(
                    wa_df,
                    "WhatsApp Persons",
                    {
                        'required': {
                            'RecordID': 'Unique ID',
                            'PersonName': 'Contact/Person name',
                            'District': 'District/location'
                        },
                        'optional': {
                            'Phone': 'Phone number',
                            'Gender': 'Gender'
                        }
                    },
                    source_key='person_wa'
                )
        
        # Validate mappings
        st.markdown("---")
        st.subheader("✅ Validation")
        
        validation_cols = st.columns(3 if not enable_multi_entity else 5)
        
        with validation_cols[0]:
            gov_missing = validate_mapping(gov_mapping, ['RecordID', 'OfficialFacilityName', 'District'])
            if not gov_missing:
                st.success("✅ Government: Ready")
            else:
                st.error(f"❌ Gov missing: {', '.join(gov_missing)}")
        
        with validation_cols[1]:
            ngo_missing = validate_mapping(ngo_mapping, ['RecordID', 'FacilityName', 'District'])
            if not ngo_missing:
                st.success("✅ NGO: Ready")
            else:
                st.error(f"❌ NGO missing: {', '.join(ngo_missing)}")
        
        with validation_cols[2]:
            wa_missing = validate_mapping(wa_mapping, ['RecordID', 'RelatedFacility', 'DistrictNote'])
            if not wa_missing:
                st.success("✅ WhatsApp: Ready")
            else:
                st.error(f"❌ WA missing: {', '.join(wa_missing)}")
        
        person_valid = True
        if enable_multi_entity:
            with validation_cols[3]:
                if person_ngo_mapping:
                    person_ngo_missing = validate_mapping(person_ngo_mapping, ['RecordID', 'PersonName', 'District'])
                    if not person_ngo_missing:
                        st.success("✅ NGO Persons: Ready")
                    else:
                        st.warning(f"⚠️ NGO Persons: {', '.join(person_ngo_missing)}")
                        person_valid = person_valid and False
                else:
                    st.info("ℹ️ NGO Persons: Skipped")
            
            with validation_cols[4]:
                if person_wa_mapping:
                    person_wa_missing = validate_mapping(person_wa_mapping, ['RecordID', 'PersonName', 'District'])
                    if not person_wa_missing:
                        st.success("✅ WA Persons: Ready")
                    else:
                        st.warning(f"⚠️ WA Persons: {', '.join(person_wa_missing)}")
                        person_valid = person_valid and False
                else:
                    st.info("ℹ️ WA Persons: Skipped")
        
        all_valid = not gov_missing and not ngo_missing and not wa_missing
    # END OF STEP 2 EXPANDER
    
    # ============================================================================
    # STEP 3: BATCH CONFIGURATION
    # ============================================================================
    
    if all_valid:
        st.markdown("---")
        st.header("📊 Step 3: Batch Configuration & History")
        
        with st.expander("📜 Batch Configuration (Click to expand/collapse)", expanded=True):
            # Apply mappings
            gov_df_mapped = apply_column_mapping(gov_df, gov_mapping)
            ngo_df_mapped = apply_column_mapping(ngo_df, ngo_mapping)
            wa_df_mapped = apply_column_mapping(wa_df, wa_mapping)
            
            # Prepare extra entity sources for multi-entity
            extra_entity_sources = []
            
            if enable_multi_entity and person_ngo_mapping:
                person_ngo_df = apply_column_mapping(ngo_df, person_ngo_mapping)
                extra_entity_sources.append({
                    "df": person_ngo_df,
                    "entity_type": "person",
                    "source_system": "NGO_Persons",
                    "name_col": "PersonName",
                    "district_col": "District",
                    "phone_col": "Phone" if "Phone" in person_ngo_mapping else None,
                    "record_id_col": "RecordID",
                    "extra_attribute_cols": ["Gender", "Role"] if "Gender" in person_ngo_mapping else []
                })
            
            if enable_multi_entity and person_wa_mapping:
                person_wa_df = apply_column_mapping(wa_df, person_wa_mapping)
                extra_entity_sources.append({
                    "df": person_wa_df,
                    "entity_type": "person",
                    "source_system": "WhatsApp_Persons",
                    "name_col": "PersonName",
                    "district_col": "District",
                    "phone_col": "Phone" if "Phone" in person_wa_mapping else None,
                    "record_id_col": "RecordID",
                    "extra_attribute_cols": ["Gender"] if "Gender" in person_wa_mapping else []
                })
            
            # Batch label input
            batch_label = st.text_input(
                "Batch label (for audit trail)",
                value="Initial_Load",
                help="Short name for this data drop, e.g., 'Morehouse_Malawi_Oct2025'"
            )
            
            # Show ingestion history
            with st.expander("📜 Ingestion History (Streaming Audit Trail)", expanded=False):
                history = get_ingestion_history()
                if len(history) == 0:
                    st.caption("No ingested batches yet. This will show all historical uploads.")
                else:
                    st.dataframe(history, use_container_width=True)
                    st.caption(f"Total historical batches: {len(history)}")
            
            # Preview mapped data
            with st.expander("👁️ Preview Mapped Data", expanded=False):
                tabs = ["Government", "NGO", "WhatsApp"]
                if enable_multi_entity:
                    if person_ngo_mapping:
                        tabs.append("NGO Persons")
                    if person_wa_mapping:
                        tabs.append("WhatsApp Persons")
                
                tab_objects = st.tabs(tabs)
                
                with tab_objects[0]:
                    st.dataframe(gov_df_mapped.head(5), use_container_width=True)
                    st.caption(f"Mapped batch: {len(gov_df_mapped)} records")
                
                with tab_objects[1]:
                    st.dataframe(ngo_df_mapped.head(5), use_container_width=True)
                    st.caption(f"Mapped batch: {len(ngo_df_mapped)} records")
                
                with tab_objects[2]:
                    st.dataframe(wa_df_mapped.head(5), use_container_width=True)
                    st.caption(f"Mapped batch: {len(wa_df_mapped)} records")
                
                tab_idx = 3
                if enable_multi_entity and person_ngo_mapping:
                    with tab_objects[tab_idx]:
                        st.dataframe(person_ngo_df.head(5), use_container_width=True)
                        st.caption(f"Person records: {len(person_ngo_df)}")
                    tab_idx += 1
                
                if enable_multi_entity and person_wa_mapping:
                    with tab_objects[tab_idx]:
                        st.dataframe(person_wa_df.head(5), use_container_width=True)
                        st.caption(f"Person records: {len(person_wa_df)}")
        # END OF STEP 3 EXPANDER
        
        # ============================================================================
        # STEP 4: INGEST & REBUILD
        # ============================================================================
        
        st.markdown("---")
        st.header("🚀 Step 4: Ingest & Rebuild Canonical Registry")
        
        metrics_cols = st.columns(3 if not enable_multi_entity else 5)
        with metrics_cols[0]:
            st.metric("Gov Facilities", len(gov_df_mapped))
        with metrics_cols[1]:
            st.metric("NGO Facilities", len(ngo_df_mapped))
        with metrics_cols[2]:
            st.metric("WhatsApp Facilities", len(wa_df_mapped))
        
        if enable_multi_entity:
            with metrics_cols[3]:
                st.metric("NGO Persons", len(person_ngo_df) if person_ngo_mapping else 0)
            with metrics_cols[4]:
                st.metric("WhatsApp Persons", len(person_wa_df) if person_wa_mapping else 0)
        
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
                    
                    # 4) Call pipeline with multi-entity support
                    # Determine if we should use pretrained model
                    # use_pretrained=True when NO ground truth (uses saved model)
                    # use_pretrained=False when ground truth provided (trains new model)
                    use_pretrained_flag = gt_df is None or len(gt_df) == 0
                    
                    # DIAGNOSTIC LOGGING
                    st.info("🔍 **Pipeline Configuration:**")
                    st.write(f"- Ground truth provided: {gt_df is not None}")
                    if gt_df is not None:
                        st.write(f"- Ground truth rows: {len(gt_df)}")
                        st.write(f"- Ground truth columns: {list(gt_df.columns)}")
                    st.write(f"- use_pretrained flag: {use_pretrained_flag}")
                    st.write(f"- Mode: {'Pretrained Model' if use_pretrained_flag else 'Train from Ground Truth'}")
                    
                    if gt_df is not None and len(gt_df) > 0:
                        st.success(f"✅ Ground truth loaded: {len(gt_df)} labeled pairs")
                        st.info("🎯 Pipeline will train NEW model from ground truth")
                    else:
                        st.info("🎯 Pipeline will use PRETRAINED model (no ground truth)")
                    
                    if enable_multi_entity and len(extra_entity_sources) > 0:
                        st.info(f"🎯 Multi-Entity Mode: Processing {len(extra_entity_sources)} person data sources")
                        results = run_vero_pipeline(
                            gov_df=gov_all_df,
                            ngo_df=ngo_all_df,
                            whatsapp_df=wa_all_df,
                            ground_truth_df=gt_df,
                            extra_entity_sources=extra_entity_sources,
                            high_threshold=high_threshold,
                            medium_threshold=medium_threshold,
                            district_threshold=district_threshold,
                            use_pretrained=use_pretrained_flag
                        )
                    else:
                        results = run_vero_pipeline(
                            gov_df=gov_all_df,
                            ngo_df=ngo_all_df,
                            whatsapp_df=wa_all_df,
                            ground_truth_df=gt_df,
                            high_threshold=high_threshold,
                            medium_threshold=medium_threshold,
                            district_threshold=district_threshold,
                            use_pretrained=use_pretrained_flag
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
# STEP 5: DISPLAY RESULTS (Enhanced Multi-Entity)
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

    # Enhanced summary metrics with entity type breakdown
    if "EntityType" in canonical.columns:
        entity_type_counts = canonical["EntityType"].value_counts()
        
        metrics_row = st.columns([2, 2, 2, 2, 2])
        with metrics_row[0]:
            st.metric("Total Entities", len(canonical))
        with metrics_row[1]:
            facilities = entity_type_counts.get('facility', 0)
            st.metric("🏢 Facilities", facilities)
        with metrics_row[2]:
            persons = entity_type_counts.get('person', 0)
            st.metric("👤 Persons", persons)
        with metrics_row[3]:
            st.metric("Matched Pairs", len(matched))
        with metrics_row[4]:
            st.metric("Model ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
    else:
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
    # TAB 1: OVERVIEW (Enhanced)
    # ----------------------------------------------------------------------
    with tab1:
        st.subheader("Matching Overview")
        
        # Entity type breakdown
        if "EntityType" in canonical.columns:
            st.markdown("### Entity Type Distribution")
            entity_type_counts = canonical["EntityType"].value_counts()
            
            col_chart, col_table = st.columns([2, 1])
            with col_chart:
                fig = px.pie(
                    values=entity_type_counts.values,
                    names=entity_type_counts.index,
                    title="Canonical Entities by Type",
                    color_discrete_map={'facility': '#1976d2', 'person': '#7b1fa2'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_table:
                st.markdown("**Breakdown:**")
                for entity_type, count in entity_type_counts.items():
                    badge_class = "badge-facility" if entity_type == "facility" else "badge-person"
                    st.markdown(f'<span class="entity-badge {badge_class}">{entity_type.title()}: {count}</span>', unsafe_allow_html=True)
                
                # Duplicate resolution stats
                if len(entity_clusters) > 0:
                    total_records = len(entity_clusters)
                    total_entities = len(canonical)
                    resolution_rate = ((total_records - total_entities) / total_records) * 100
                    st.markdown(f"**Resolution Rate:** {resolution_rate:.1f}%")
                    st.caption(f"{total_records - total_entities} duplicates resolved")
        
        if len(matched) > 0 and "match_prob" in matched.columns:
            st.markdown("### Match Probability Distribution")
            fig = px.histogram(
                matched, x='match_prob', nbins=20,
                title="Match Confidence Distribution",
                labels={'match_prob': 'Probability', 'count': 'Pairs'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if len(entity_clusters) > 0 and "SourceSystem" in entity_clusters.columns:
            st.markdown("### Records by Source System")
            source_dist = entity_clusters["SourceSystem"].value_counts()
            # Fix for plotly duplicate index error
            source_df = pd.DataFrame({
                'Source': source_dist.index,
                'Count': source_dist.values
            })
            fig = px.bar(
                source_df, 
                x='Source', 
                y='Count',
                title="Distribution by Source",
                labels={'Source': 'Source System', 'Count': 'Record Count'}
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
    # TAB 3: CANONICAL ENTITIES (Enhanced Multi-Entity Display)
    # ----------------------------------------------------------------------
    with tab3:
        st.subheader("🧩 Canonical Entities (LLM-Ready Identity Table)")
        st.caption("One row per real-world entity, deduplicated across all ingested batches")
        
        if len(canonical) > 0:
            # Entity type filter
            if "EntityType" in canonical.columns:
                entity_types = ["All"] + list(canonical["EntityType"].unique())
                selected_type = st.selectbox("Filter by Entity Type:", entity_types)
                
                if selected_type != "All":
                    filtered_canonical = canonical[canonical["EntityType"] == selected_type]
                else:
                    filtered_canonical = canonical
            else:
                filtered_canonical = canonical
            
            # Display facilities
            if "EntityType" not in canonical.columns or selected_type in ["All", "facility"]:
                facilities = filtered_canonical[filtered_canonical["EntityType"] == "facility"] if "EntityType" in filtered_canonical.columns else filtered_canonical
                if len(facilities) > 0:
                    st.markdown("### 🏢 Facility Entities")
                    st.dataframe(facilities, use_container_width=True)
                    st.caption(f"Total facilities: {len(facilities)}")
            
            # Display persons
            if "EntityType" in canonical.columns and selected_type in ["All", "person"]:
                persons = filtered_canonical[filtered_canonical["EntityType"] == "person"]
                if len(persons) > 0:
                    st.markdown("### 👤 Person Entities")
                    
                    # Create person-specific display
                    person_display = []
                    for _, person in persons.iterrows():
                        try:
                            attrs = json.loads(person['Attributes_JSON']) if 'Attributes_JSON' in person else {}
                        except:
                            attrs = {}
                        
                        person_display.append({
                            'EntityID': person['EntityID'],
                            'Name': person['CanonicalName'],
                            'District': person.get('PrimaryDistrict', 'N/A'),
                            'Phone': person.get('CanonicalPhones', 'N/A'),
                            'Gender': attrs.get('Gender', 'N/A'),
                            'Records': person['RecordCount'],
                            'Sources': person['SourcesRepresented']
                        })
                    
                    st.dataframe(pd.DataFrame(person_display), use_container_width=True)
                    st.caption(f"Total persons: {len(persons)}")
                    
                    # Highlight resolved duplicates
                    duplicates = [p for p in person_display if p['Records'] > 1]
                    if duplicates:
                        st.success(f"✅ Resolved {len(duplicates)} person duplicates across sources!")
                        with st.expander("View Resolved Person Duplicates"):
                            for person in duplicates:
                                st.write(f"- **{person['Name']}**: {person['Records']} records from {person['Sources']}")
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
                entity_type = row.get('EntityType', 'entity')
                icon = "🏢" if entity_type == "facility" else "👤"
                label = f"{icon} {row['EntityID']} | {row['CanonicalName']} ({entity_type}, {row.get('PrimaryDistrict', 'N/A')})"
                entity_options.append(label)
            
            selected_label = st.selectbox(
                "Select an entity to query:",
                options=entity_options,
                help="Choose the canonical entity you want to ask about"
            )
            
            selected_entity_id = selected_label.split(" | ")[0].split(" ")[1]
            selected = canonical[canonical["EntityID"] == selected_entity_id].iloc[0]
            
            entity_type = selected.get('EntityType', 'entity')
            icon = "🏢" if entity_type == "facility" else "👤"
            st.markdown(f"**Selected:** {icon} `{selected['CanonicalName']}` ({entity_type})")
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
                placeholder="e.g., Summarize this entity's data quality, What sources contributed?, etc.",
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
    <p><strong>VERO - Entity Resolution Platform v4.0.4 Ultimate</strong></p>
    <p>✅ Flexible Column Mapping | ✅ Mapping Memory | ✅ Multi-Entity Support | ✅ Smart Auto-Mapping</p>
    <p>Streaming-Ready Canonical Identity Fabric | 65.7% Person Resolution Rate</p>
    <p>© 2025</p>
</div>
""", unsafe_allow_html=True)
