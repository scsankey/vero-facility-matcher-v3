"""
test_data.py - Sample test data for VERO engine
"""
import pandas as pd

def get_test_data():
    """Returns sample Gov, NGO, WhatsApp, and ground truth data"""
    
    # Government facility data
    gov_df = pd.DataFrame({
        'RecordID': ['G1', 'G2', 'G3', 'G4', 'G5'],
        'OfficialFacilityName': [
            'Lilongwe District Hospital',
            'Zomba Health Centre',
            'Mzuzu Clinic',
            'Blantyre Central Hospital',
            'Karonga Rural Health Centre'
        ],
        'District': ['Lilongwe', 'Zomba', 'Mzuzu', 'Blantyre', 'Karonga'],
        'AltName': [None, 'Zomba HC', None, 'Blantyre Central', 'Karonga RHC']
    })
    
    # NGO facility data (with duplicates)
    ngo_df = pd.DataFrame({
        'RecordID': ['N1', 'N2', 'N3', 'N4'],
        'FacilityName': [
            'Lilongwe DH',
            'Zomba Health Center',
            'Mzuzu Community Clinic',
            'Blantyre Central Referral Hospital'
        ],
        'District': ['Lilongwe', 'Zomba', 'Mzuzu', 'Blantyre'],
        'Phone': ['265999123456', '265888234567', '265777345678', '265666456789']
    })
    
    # WhatsApp data (informal mentions)
    whatsapp_df = pd.DataFrame({
        'RecordID': ['W1', 'W2', 'W3'],
        'RelatedFacility': [
            'Lilongwe Hospital',
            'Zomba HC',
            'Blantyre Central'
        ],
        'DistrictNote': ['Lilongwe', 'Zomba', 'Blantyre'],
        'Phone': ['265999123456', '265888234567', '265666456789'],
        'LocationNickname': ['LWI DH', 'ZBA HC', 'BT Central']
    })
    
    # Ground truth (labeled matches)
    ground_truth = pd.DataFrame({
        'Record_ID A': ['G1', 'G2', 'G3', 'G4', 'N1', 'N2'],
        'Record_ID B': ['N1', 'N2', 'N3', 'N4', 'W1', 'W2'],
        'Same Entity': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes']
    })
    
    return gov_df, ngo_df, whatsapp_df, ground_truth


def get_multi_entity_test_data():
    """Returns test data with facilities AND persons"""
    
    # Get facility data
    gov_df, ngo_df, whatsapp_df, ground_truth = get_test_data()
    
    # Add person data (healthcare workers)
    hr_persons_df = pd.DataFrame({
        'EmployeeID': ['HR001', 'HR002', 'HR003', 'HR004'],
        'FullName': [
            'Dr. John Banda',
            'Nurse Mary Phiri',
            'Dr. John M. Banda',
            'Nurse Mary C. Phiri'
        ],
        'District': ['Lilongwe', 'Zomba', 'Lilongwe', 'Zomba'],
        'Phone': ['265111222333', '265444555666', '265111222333', '265444555666'],
        'Gender': ['Male', 'Female', 'Male', 'Female'],
        'Occupation': ['Doctor', 'Nurse', 'Doctor', 'Nurse']
    })
    
    # Configure extra entity sources for persons
    extra_entity_sources = [{
        "df": hr_persons_df,
        "entity_type": "person",
        "source_system": "HR",
        "name_col": "FullName",
        "district_col": "District",
        "phone_col": "Phone",
        "record_id_col": "EmployeeID",
        "extra_attribute_cols": ["Gender", "Occupation"]
    }]
    
    return gov_df, ngo_df, whatsapp_df, ground_truth, extra_entity_sources
