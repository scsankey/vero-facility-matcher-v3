"""
test_app_real.py - Streamlit app to test VERO with REAL data
"""
import streamlit as st
import pandas as pd
from vero_engine import run_vero_pipeline
from test_data_real import get_real_data, get_real_multi_entity_data
import json

st.set_page_config(page_title="VERO Real Data Test", page_icon="🌾", layout="wide")

st.title("🌾 VERO Engine - Real Agricultural Cooperative Data")
st.markdown("---")

# Test selection
test_type = st.sidebar.selectbox(
    "Select Test",
    ["Test 1: Facility Matching (Agricultural Coops)", 
     "Test 2: Multi-Entity (Facilities + Farmers)",
     "Test 3: View Raw Data"]
)

# ============================================================================
# TEST 1: Real Facility Matching
# ============================================================================

if test_type == "Test 1: Facility Matching (Agricultural Coops)":
    st.header("Test 1: Agricultural Cooperative Matching")
    st.info("Testing with real data: 15 Gov + 20 NGO + 15 WhatsApp records")
    
    if st.button("🚀 Run Test 1", type="primary"):
        with st.spinner("Running pipeline on real data..."):
            try:
                # Get real data
                gov_df, ngo_df, whatsapp_df, ground_truth = get_real_data()
                
                # Show input data
                with st.expander("📊 Input Data"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.subheader("Gov Registry")
                        st.dataframe(gov_df[['RecordID', 'OfficialFacilityName', 'District']].head(10), 
                                   use_container_width=True)
                    with col2:
                        st.subheader("NGO Dataset")
                        st.dataframe(ngo_df[['RecordID', 'FacilityName', 'District']].head(10), 
                                   use_container_width=True)
                    with col3:
                        st.subheader("WhatsApp Data")
                        st.dataframe(whatsapp_df[['RecordID', 'RelatedFacility', 'DistrictNote']].head(10), 
                                   use_container_width=True)
                
                # Show ground truth stats
                st.info(f"Ground Truth: {len(ground_truth)} pairs ({(ground_truth['Same Entity']=='Yes').sum()} positive, {(ground_truth['Same Entity']=='No').sum()} negative)")
                
                # Run pipeline
                results = run_vero_pipeline(
                    gov_df=gov_df,
                    ngo_df=ngo_df,
                    whatsapp_df=whatsapp_df,
                    ground_truth_df=ground_truth
                )
                
                # Display results
                st.success("✅ Pipeline completed successfully!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Canonical Entities", len(results['canonical_entities']))
                with col2:
                    st.metric("Matched Pairs", len(results['matched_pairs']))
                with col3:
                    st.metric("Model ROC-AUC", f"{results['metrics']['roc_auc']:.3f}")
                with col4:
                    st.metric("Accuracy", f"{results['metrics']['accuracy']:.3f}")
                
                # Canonical entities
                st.subheader("📋 Canonical Entities (Deduplicated)")
                canonical = results['canonical_entities']
                st.dataframe(
                    canonical[['EntityID', 'CanonicalName', 'PrimaryDistrict', 
                              'RecordCount', 'SourcesRepresented', 'Aliases']],
                    use_container_width=True
                )
                
                # Show some examples of resolved duplicates
                st.subheader("🔗 Example Resolved Duplicates")
                multi_source = canonical[canonical['RecordCount'] > 1].head(5)
                for _, entity in multi_source.iterrows():
                    with st.expander(f"{entity['CanonicalName']} ({entity['RecordCount']} records)"):
                        st.write(f"**Sources:** {entity['SourcesRepresented']}")
                        st.write(f"**District:** {entity['PrimaryDistrict']}")
                        st.write(f"**All names:** {entity['Aliases']}")
                        st.write(f"**Record IDs:** {entity['SourceRecordIDs']}")
                
                # Matched pairs (top 10)
                st.subheader("🔗 Top Matched Pairs")
                if len(results['matched_pairs']) > 0:
                    top_matches = results['matched_pairs'].head(10)
                    st.dataframe(
                        top_matches[['record_A', 'record_B', 'name_A', 'name_B', 
                                    'match_prob', 'source_A', 'source_B']],
                        use_container_width=True
                    )
                else:
                    st.warning("No matches found")
                
                # Verification checks
                st.subheader("✅ Quality Checks")
                checks = []
                
                # Duplicate rate
                total_records = len(gov_df) + len(ngo_df) + len(whatsapp_df)
                canonical_count = len(canonical)
                duplicate_rate = (total_records - canonical_count) / total_records * 100
                checks.append(("📊", f"Duplicate rate: {duplicate_rate:.1f}% ({total_records - canonical_count} duplicates found)"))
                
                # Cross-source matches
                cross_source = canonical[canonical['SourcesRepresented'].str.contains('\+')]
                checks.append(("🔗", f"Cross-source matches: {len(cross_source)} entities in multiple sources"))
                
                # Model performance
                if results['metrics']['roc_auc'] >= 0.85:
                    checks.append(("✅", f"Excellent model performance (ROC-AUC: {results['metrics']['roc_auc']:.3f})"))
                elif results['metrics']['roc_auc'] >= 0.75:
                    checks.append(("⚠️", f"Good model performance (ROC-AUC: {results['metrics']['roc_auc']:.3f})"))
                else:
                    checks.append(("❌", f"Model needs improvement (ROC-AUC: {results['metrics']['roc_auc']:.3f})"))
                
                for icon, msg in checks:
                    st.markdown(f"{icon} {msg}")
                
            except Exception as e:
                st.error(f"❌ Test failed: {str(e)}")
                st.exception(e)

# ============================================================================
# TEST 2: Multi-Entity (Facilities + Farmers)
# ============================================================================

elif test_type == "Test 2: Multi-Entity (Facilities + Farmers)":
    st.header("Test 2: Multi-Entity (Facilities + Farmers)")
    st.info("Testing EARE with agricultural cooperatives AND farmer persons")
    
    if st.button("🚀 Run Test 2", type="primary"):
        with st.spinner("Running multi-entity pipeline..."):
            try:
                # Get multi-entity data
                gov_df, ngo_df, whatsapp_df, ground_truth, extra_entity_sources = get_real_multi_entity_data()
                
                # Show input data
                with st.expander("📊 Input Data"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Facility Data (Gov)")
                        st.dataframe(gov_df[['RecordID', 'OfficialFacilityName', 'District']].head(8), 
                                   use_container_width=True)
                    with col2:
                        st.subheader("Farmer Data (NGO)")
                        st.dataframe(extra_entity_sources[0]['df'].head(8), use_container_width=True)
                
                # Run pipeline
                results = run_vero_pipeline(
                    gov_df=gov_df,
                    ngo_df=ngo_df,
                    whatsapp_df=whatsapp_df,
                    ground_truth_df=ground_truth,
                    extra_entity_sources=extra_entity_sources
                )
                
                # Display results
                st.success("✅ Multi-entity pipeline completed!")
                
                canonical = results['canonical_entities']
                
                # Overall metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Canonical Entities", len(canonical))
                with col2:
                    facilities = canonical[canonical['EntityType'] == 'facility']
                    st.metric("Facilities (Coops)", len(facilities))
                with col3:
                    persons = canonical[canonical['EntityType'] == 'person']
                    st.metric("Persons (Farmers)", len(persons))
                
                # Entity breakdown
                st.subheader("📊 Entity Type Breakdown")
                entity_counts = canonical['EntityType'].value_counts()
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.dataframe(entity_counts.reset_index().rename(
                        columns={'index': 'Entity Type', 'EntityType': 'Count'}
                    ))
                with col2:
                    st.bar_chart(entity_counts)
                
                # Show facilities
                st.subheader("🌾 Facility Entities (Agricultural Coops)")
                facilities = canonical[canonical['EntityType'] == 'facility']
                st.dataframe(
                    facilities[['EntityID', 'CanonicalName', 'PrimaryDistrict', 'RecordCount', 'SourcesRepresented']],
                    use_container_width=True
                )
                
                # Show persons (farmers)
                st.subheader("👤 Person Entities (Farmers)")
                persons = canonical[canonical['EntityType'] == 'person']
                if len(persons) > 0:
                    person_display = []
                    for _, person in persons.iterrows():
                        attrs = json.loads(person['Attributes_JSON'])
                        person_display.append({
                            'EntityID': person['EntityID'],
                            'Name': person['CanonicalName'],
                            'District': person['PrimaryDistrict'],
                            'Phone': person.get('CanonicalPhones', 'N/A'),
                            'Gender': attrs.get('Gender', 'N/A'),
                            'Records': person['RecordCount'],
                            'Sources': person['SourcesRepresented']
                        })
                    st.dataframe(pd.DataFrame(person_display), use_container_width=True)
                    
                    # Highlight resolved farmer duplicates
                    farmer_dupes = [p for p in person_display if p['Records'] > 1]
                    if farmer_dupes:
                        st.success(f"✅ Resolved {len(farmer_dupes)} farmer duplicates across sources!")
                        for farmer in farmer_dupes:
                            st.write(f"- **{farmer['Name']}**: {farmer['Records']} records from {farmer['Sources']}")
                else:
                    st.warning("No person entities found")
                
                # Type safety verification
                st.subheader("🔒 Type Safety Check")
                clusters = results['entity_clusters']
                type_safety_ok = True
                for entity_id in canonical['EntityID'].unique():
                    cluster_records = clusters[clusters['EntityID'] == entity_id]
                    entity_types = cluster_records['EntityType'].unique()
                    if len(entity_types) > 1:
                        st.error(f"❌ Type mixing in {entity_id}: {entity_types}")
                        type_safety_ok = False
                
                if type_safety_ok:
                    st.success("✅ Type safety maintained - no cross-type matches!")
                
            except Exception as e:
                st.error(f"❌ Test failed: {str(e)}")
                st.exception(e)

# ============================================================================
# TEST 3: View Raw Data
# ============================================================================

else:  # Test 3: View Raw Data
    st.header("Test 3: View Raw Data")
    st.info("Preview real data from your Excel file")
    
    gov_df, ngo_df, whatsapp_df, ground_truth = get_real_data()
    
    tab1, tab2, tab3, tab4 = st.tabs(["Gov Registry", "NGO Dataset", "WhatsApp Data", "Ground Truth"])
    
    with tab1:
        st.subheader("Government Registry (Agricultural Cooperatives)")
        st.dataframe(gov_df, use_container_width=True)
        st.caption(f"Total records: {len(gov_df)}")
    
    with tab2:
        st.subheader("NGO Dataset (Farmers & Facilities)")
        st.dataframe(ngo_df, use_container_width=True)
        st.caption(f"Total records: {len(ngo_df)}")
    
    with tab3:
        st.subheader("WhatsApp Data (Informal Reports)")
        st.dataframe(whatsapp_df, use_container_width=True)
        st.caption(f"Total records: {len(whatsapp_df)}")
    
    with tab4:
        st.subheader("Ground Truth Labels")
        st.dataframe(ground_truth, use_container_width=True)
        yes_count = (ground_truth['Same Entity'] == 'Yes').sum()
        no_count = (ground_truth['Same Entity'] == 'No').sum()
        st.caption(f"Total pairs: {len(ground_truth)} ({yes_count} positive, {no_count} negative)")

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.caption("VERO Engine v4.0.3 | Real Agricultural Cooperative Data | EARE Framework")
