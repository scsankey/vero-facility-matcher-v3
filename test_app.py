"""
test_app.py - Streamlit app to test VERO engine v4.0.3
"""
import streamlit as st
import pandas as pd
from vero_engine import run_vero_pipeline
from test_data import get_test_data, get_multi_entity_test_data
import json

st.set_page_config(page_title="VERO Engine Test", page_icon="🧪", layout="wide")

st.title("🧪 VERO Engine v4.0.3 - Test Suite")
st.markdown("---")

# Test selection
test_type = st.sidebar.selectbox(
    "Select Test",
    ["Test 1: Basic Facility Matching", 
     "Test 2: Multi-Entity (Facility + Person)",
     "Test 3: View Sample Data Only"]
)

# ============================================================================
# TEST 1: Basic Facility Matching
# ============================================================================

if test_type == "Test 1: Basic Facility Matching":
    st.header("Test 1: Basic Facility Matching")
    st.info("Testing backward compatibility with facility-only data")
    
    if st.button("🚀 Run Test 1", type="primary"):
        with st.spinner("Running pipeline..."):
            try:
                # Get test data
                gov_df, ngo_df, whatsapp_df, ground_truth = get_test_data()
                
                # Show input data
                with st.expander("📊 Input Data"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.subheader("Gov Data")
                        st.dataframe(gov_df, use_container_width=True)
                    with col2:
                        st.subheader("NGO Data")
                        st.dataframe(ngo_df, use_container_width=True)
                    with col3:
                        st.subheader("WhatsApp Data")
                        st.dataframe(whatsapp_df, use_container_width=True)
                
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
                st.subheader("📋 Canonical Entities")
                canonical = results['canonical_entities']
                st.dataframe(
                    canonical[['EntityID', 'EntityType', 'CanonicalName', 'PrimaryDistrict', 
                              'RecordCount', 'SourcesRepresented']],
                    use_container_width=True
                )
                
                # Matched pairs
                st.subheader("🔗 Matched Pairs")
                if len(results['matched_pairs']) > 0:
                    st.dataframe(
                        results['matched_pairs'][['record_A', 'record_B', 'name_A', 'name_B', 
                                                  'match_prob', 'source_A', 'source_B']],
                        use_container_width=True
                    )
                else:
                    st.warning("No matches found")
                
                # Verification checks
                st.subheader("✅ Verification Checks")
                checks = []
                
                # Check 1: All required fields present
                required_fields = ['EntityID', 'EntityType', 'CanonicalName', 'PrimaryDistrict']
                missing_fields = [f for f in required_fields if f not in canonical.columns]
                if not missing_fields:
                    checks.append(("✅", "All required fields present"))
                else:
                    checks.append(("❌", f"Missing fields: {missing_fields}"))
                
                # Check 2: Entity type consistency
                if canonical['EntityType'].nunique() == 1 and canonical['EntityType'].iloc[0] == 'facility':
                    checks.append(("✅", "All entities are facilities (correct)"))
                else:
                    checks.append(("⚠️", "Mixed entity types found"))
                
                # Check 3: No empty canonical names
                if canonical['CanonicalName'].notna().all():
                    checks.append(("✅", "All canonical names populated"))
                else:
                    checks.append(("❌", "Some canonical names are empty"))
                
                # Check 4: ROC-AUC quality
                if results['metrics']['roc_auc'] >= 0.80:
                    checks.append(("✅", f"Good model performance (ROC-AUC: {results['metrics']['roc_auc']:.3f})"))
                else:
                    checks.append(("⚠️", f"Model performance could be better (ROC-AUC: {results['metrics']['roc_auc']:.3f})"))
                
                for icon, msg in checks:
                    st.markdown(f"{icon} {msg}")
                
            except Exception as e:
                st.error(f"❌ Test failed: {str(e)}")
                st.exception(e)

# ============================================================================
# TEST 2: Multi-Entity (EARE)
# ============================================================================

elif test_type == "Test 2: Multi-Entity (Facility + Person)":
    st.header("Test 2: Multi-Entity Support (EARE)")
    st.info("Testing EARE framework with facilities AND persons")
    
    if st.button("🚀 Run Test 2", type="primary"):
        with st.spinner("Running multi-entity pipeline..."):
            try:
                # Get test data
                gov_df, ngo_df, whatsapp_df, ground_truth, extra_entity_sources = get_multi_entity_test_data()
                
                # Show input data
                with st.expander("📊 Input Data"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Facility Data")
                        st.caption("Gov")
                        st.dataframe(gov_df, use_container_width=True)
                        st.caption("NGO")
                        st.dataframe(ngo_df, use_container_width=True)
                    with col2:
                        st.subheader("Person Data (HR)")
                        st.dataframe(extra_entity_sources[0]['df'], use_container_width=True)
                
                # Run pipeline with multi-entity
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
                    st.metric("Facilities", len(facilities))
                with col3:
                    persons = canonical[canonical['EntityType'] == 'person']
                    st.metric("Persons", len(persons))
                
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
                st.subheader("🏥 Facility Entities")
                facilities = canonical[canonical['EntityType'] == 'facility']
                st.dataframe(
                    facilities[['EntityID', 'CanonicalName', 'PrimaryDistrict', 'RecordCount', 'SourcesRepresented']],
                    use_container_width=True
                )
                
                # Show persons with attributes
                st.subheader("👤 Person Entities")
                persons = canonical[canonical['EntityType'] == 'person']
                if len(persons) > 0:
                    person_display = []
                    for _, person in persons.iterrows():
                        attrs = json.loads(person['Attributes_JSON'])
                        person_display.append({
                            'EntityID': person['EntityID'],
                            'Name': person['CanonicalName'],
                            'District': person['PrimaryDistrict'],
                            'Gender': attrs.get('Gender', 'N/A'),
                            'Occupation': attrs.get('Occupation', 'N/A'),
                            'Records': person['RecordCount']
                        })
                    st.dataframe(pd.DataFrame(person_display), use_container_width=True)
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
# TEST 3: View Sample Data
# ============================================================================

else:  # Test 3: View Sample Data Only
    st.header("Test 3: View Sample Data")
    st.info("Preview test data without running the pipeline")
    
    gov_df, ngo_df, whatsapp_df, ground_truth = get_test_data()
    
    tab1, tab2, tab3, tab4 = st.tabs(["Gov Data", "NGO Data", "WhatsApp Data", "Ground Truth"])
    
    with tab1:
        st.subheader("Government Facility Data")
        st.dataframe(gov_df, use_container_width=True)
        st.caption(f"Total records: {len(gov_df)}")
    
    with tab2:
        st.subheader("NGO Facility Data")
        st.dataframe(ngo_df, use_container_width=True)
        st.caption(f"Total records: {len(ngo_df)}")
    
    with tab3:
        st.subheader("WhatsApp Data")
        st.dataframe(whatsapp_df, use_container_width=True)
        st.caption(f"Total records: {len(whatsapp_df)}")
    
    with tab4:
        st.subheader("Ground Truth Labels")
        st.dataframe(ground_truth, use_container_width=True)
        st.caption(f"Total labeled pairs: {len(ground_truth)}")
    
    # Multi-entity preview
    st.markdown("---")
    st.subheader("Multi-Entity Test Data")
    if st.checkbox("Show person data"):
        _, _, _, _, extra_entity_sources = get_multi_entity_test_data()
        st.dataframe(extra_entity_sources[0]['df'], use_container_width=True)

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.caption("VERO Engine v4.0.3 Test Suite | Testing EARE multi-entity framework")
