"""
app.py
VERO - Streaming-Ready Entity Resolution Platform
Upload → Batch Ingest → Rebuild Canonical → LLM Chat → Download
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

def validate_dataframe(df, required_cols, dataset_name):
    """Validate that dataframe has required columns"""
    if df is None:
        return False
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"{dataset_name} is missing columns: {', '.join(missing_cols)}")
        return False
    return True

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
st.markdown('<div class="sub-header">Streaming-Ready Canonical Identity Fabric</div>', unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'confirm_clear' not in st.session_state:
    st.session_state.confirm_clear = False

# Initialize storage
init_storage()

# ============================================================================
# STEP 1: FILE UPLOAD
# ============================================================================

st.header("📤 Step 1: Upload Your Data (Batch Ingestion)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Option A: Single Excel File")
    excel_file = st.file_uploader(
        "Upload Excel with multiple sheets",
        type=['xlsx', 'xls'],
        help="Should contain: Government registry, NGO Dataset, WhatsApp Dataset"
    )
    
    if excel_file:
        try:
            excel_sheets = pd.ExcelFile(excel_file).sheet_names
            st.success(f"✅ Found {len(excel_sheets)} sheets")
            
            gov_sheet = st.selectbox("Government Registry", excel_sheets, 
                                    index=excel_sheets.index('Government registry') if 'Government registry' in excel_sheets else 0)
            ngo_sheet = st.selectbox("NGO Dataset", excel_sheets,
                                    index=excel_sheets.index('NGO Dataset') if 'NGO Dataset' in excel_sheets else 0)
            wa_sheet = st.selectbox("WhatsApp Dataset", excel_sheets,
                                   index=excel_sheets.index('WhatsApp Dataset') if 'WhatsApp Dataset' in excel_sheets else 0)
            
            has_ground_truth = st.checkbox("Include Ground Truth for Training")
            gt_sheet = None
            if has_ground_truth:
                gt_sheet = st.selectbox("Ground Truth", excel_sheets,
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
# STEP 2: BATCH LABELING & INGESTION HISTORY
# ============================================================================

if excel_file or (gov_csv and ngo_csv and wa_csv):
    st.markdown("---")
    st.header("📊 Step 2: Batch Configuration & History")
    
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
    
    # Validate
    valid_gov = validate_dataframe(gov_df, ['RecordID', 'OfficialFacilityName', 'District'], "Government")
    valid_ngo = validate_dataframe(ngo_df, ['RecordID', 'FacilityName', 'District'], "NGO")
    valid_wa = validate_dataframe(wa_df, ['RecordID', 'RelatedFacility', 'DistrictNote'], "WhatsApp")
    
    if valid_gov and valid_ngo and valid_wa:
        # Preview current batch
        with st.expander("👁️ Preview Current Batch", expanded=False):
            t1, t2, t3 = st.tabs(["Government", "NGO", "WhatsApp"])
            
            with t1:
                st.dataframe(gov_df.head(5), use_container_width=True)
                st.caption(f"Current batch: {len(gov_df)} records")
            
            with t2:
                st.dataframe(ngo_df.head(5), use_container_width=True)
                st.caption(f"Current batch: {len(ngo_df)} records")
            
            with t3:
                st.dataframe(wa_df.head(5), use_container_width=True)
                st.caption(f"Current batch: {len(wa_df)} records")
        
        # ============================================================================
        # STEP 3: INGEST & REBUILD CANONICAL REGISTRY
        # ============================================================================
        
        st.markdown("---")
        st.header("🚀 Step 3: Ingest & Rebuild Canonical Registry")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Batch: Government", len(gov_df))
        with col2:
            st.metric("Current Batch: NGO", len(ngo_df))
        with col3:
            st.metric("Current Batch: WhatsApp", len(wa_df))
        
        if st.button("📥 Ingest & Rebuild Canonical Registry", type="primary", use_container_width=True):
            with st.spinner("Ingesting batch and rebuilding canonical registry..."):
                try:
                    # 1) Add this batch into storage
                    add_batch("Gov", gov_df, batch_label=batch_label + "_GOV")
                    add_batch("NGO", ngo_df, batch_label=batch_label + "_NGO")
                    add_batch("WhatsApp", wa_df, batch_label=batch_label + "_WA")
                    
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
# STEP 4: DISPLAY RESULTS
# ============================================================================

if st.session_state.results:
    results = st.session_state.results
    
    st.markdown("---")
    st.header("📈 Step 4: Results & Insights")
    
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
    <p><strong>VERO - Entity Resolution Platform</strong></p>
    <p>Streaming-Ready Canonical Identity Fabric | ADIP v1.0</p>
    <p>© 2025</p>
</div>
""", unsafe_allow_html=True)
