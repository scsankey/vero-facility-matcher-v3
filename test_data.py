"""
test_data_real.py - Real data from Algorithims_Deterministic.xlsx
"""
import pandas as pd

def get_real_data():
    """
    Returns the REAL Gov, NGO, WhatsApp, and ground truth data
    from your actual Excel file.
    
    NOTE: This data is about agricultural cooperatives/farm centers,
    not health facilities!
    """
    
    # Government registry data
    gov_df = pd.DataFrame({
        'RecordID': ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15'],
        'FacilityID': ['MW-CHKW-001', 'MW-CHKW-002', 'MW-CHKW-003', 'MW-CHKW-004', 'MW-CHKW-005', 
                       'MW-CHKW-006', 'MW-CHKW-007', 'MW-CHKW-008', 'MW-CHKW-009', 'MW-CHKW-010',
                       'MW-CHKW-011', 'MW-CHKW-012', 'MW-CHKW-013', 'MW-CHKW-014', 'MW-CHKW-015'],
        'OfficialFacilityName': [
            'Mbale District Coorporation',
            'Mbhale Farm Centre',
            'Mbale Rural Farm Centre',
            'Rakai District Coorporation',
            'Bangula Farm Post',
            'Nsanje District Coorporation',
            'Queen Elizabeth Farmers Centre',
            'Tororo District Coop',
            'Masenjere District Farm Centre',
            'Area 25 Farmers Hub',
            'Mbale District Coorporation',  # Duplicate of G1
            'Kanjedza District Farm',
            'Shire Valley Cooperative',
            'Mbale Rural Farm Centre',  # Duplicate of G3
            'Balaka District RFC'
        ],
        'AltName': [
            'Mbale RFC', 'Mbhale Farm Center', 'Mbale RFC', 'Rakai RFC', None,
            'Nsanje DC', None, 'Tororo Dist Coop', 'Masenjere RFC', None,
            'Mbale District', 'Kanjedza Farm', 'Shire Coop', 'Mbale Rural', 'Balaka RFC'
        ],
        'District': [
            'Mbale', 'Mbale', 'Mbale', 'Rakai', 'Rakai',
            'Nsanje', 'Chikwawa', 'Tororo Rural', 'Masenjere', 'Tororo Urban',
            'Mbale', 'Kanjedza', 'Shire', 'Mbale', 'Balaka'
        ]
    })
    
    # NGO dataset
    ngo_df = pd.DataFrame({
        'RecordID': ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10',
                     'N11', 'N12', 'N13', 'N14', 'N15', 'N16', 'N17', 'N18', 'N19', 'N20'],
        'FarmerID': list(range(1, 21)),
        'FarmerName': [
            'Mphatso Banda', 'Mfatsso B.', 'John Phiri', 'Grace Tembo', 'Sarah Mwale',
            'James Kachingwe', 'Mary Phiri', 'Patrick Banda', 'Alice Nkhoma', 'David Chirwa',
            'Grace T.', 'Mphatso B', 'John P.', 'Sarah M.', 'James K.',
            'Mary P.', 'Patrick B.', 'Alice N.', 'David C.', 'Grace Tembo'
        ],
        'Gender': ['F', 'F', 'M', 'F', 'F', 'M', 'F', 'M', 'F', 'M',
                   'F', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
        'Phone': [
            '265882001001', None, '889002300', '991234600', '888900123',
            '999111222', '888777666', '777666555', '666555444', '555444333',
            '991234600', '265882001001', '889002300', '888900123', '999111222',
            '888777666', '777666555', '666555444', '555444333', '991234600'
        ],
        'FacilityName': [
            'Mbale Coop',
            'Mbhale Coorporation Center',
            'Rakai District Coop',
            'Rakai Dist Corp',
            'Area 25 Farmers Centre',
            'Bangula Post',
            'Queen E Central',
            'Tororo Coop',
            'Masenjere Farm Centre',
            'Area 25 Hub',
            'Kanjedza Coop',
            'Mbale District Cooperative',
            'Shire Valley',
            'Rakai RFC',
            'Mbhale FC',
            'Mbhale Center',
            'Tororo Area 25',
            'Bangula Farm',
            'QECC',
            'Masenjere DC'
        ],
        'District': [
            'Mbale', 'Mbhale', 'Rakai', 'Rakai', 'Tororo Urban',
            'Rakai', 'Chikwawa', 'Tororo Rural', 'Masenjere', 'Tororo',
            'Kanjedza', 'Mbale', 'Shire', 'Rakai', 'Mbale',
            'Mbhale', 'Tororo', 'Bangula', 'Chikwawa', 'Masenjere'
        ]
    })
    
    # WhatsApp dataset
    whatsapp_df = pd.DataFrame({
        'RecordID': ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10',
                     'W11', 'W12', 'W13', 'W14', 'W15'],
        'ChatID': list(range(1, 16)),
        'ContactName': [
            'Mphatso Banda', 'Mfatsso B', 'Grace T Area25', 'Grace T', 'Peter K',
            'Mary P', 'John P', 'Alice N', 'Sarah M', 'James K',
            'Patrick B', 'David C', 'Mphatso B.', 'Grace Tembo', 'Mary Phiri'
        ],
        'Phone': [
            '265882001001', '882001000', '991234600', '991234600', '888900456',
            '888777666', '889002300', '666555444', '888900123', '999111222',
            '777666555', '555444333', '265882001001', '991234600', '888777666'
        ],
        'LocationNickname': [
            'Kanjedza Mbale RFC', 'Kanjedza near Mbhale HC', 'Area 25 Tororo', 'Area 25',
            'Bangula', 'Queen E', 'Rakai', 'Masenjere', 'Area 25 Farmers', 'Kanjedza',
            'Tororo', 'Masenjere DC', 'Mbale District', 'Area 25 Hub', 'Mbhale Center'
        ],
        'RelatedFacility': [
            'Mbale RFC',
            'Mbhale Farmers Center',
            'Area 25 Farmers Centre',
            'Tororo Area 25 HC',
            'Queen Elizabeth Central Coorporation',
            'QE Farmers Centre',
            'Rakai District Coop',
            'Masenjere Farm Centre',
            'Area 25 Farmers Hub',
            'Kanjedza District Farm',
            'Tororo District Coop',
            'Masenjere RFC',
            'Mbale District Coop',
            'Area 25 Hub',
            'Mbhale Farm Centre'
        ],
        'DistrictNote': [
            'Mbale', 'Mbhale', 'TRR', 'Tororo Rural', 'MK',
            'Chikwawa', 'Rakai', 'Masenjere', 'Tororo', 'Kanjedza',
            'Tororo', 'Masenjere', 'Mbale', 'Tororo', 'Mbhale'
        ]
    })
    
    # Ground truth (with both Yes and No examples)
    ground_truth = pd.DataFrame({
        'Record_ID A': [
            # Positive examples (facilities that match)
            'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10',
            'G1', 'G3', 'N1', 'N2', 'N3', 'N5', 'N6', 'N7', 'N8', 'N9',
            'N11', 'N12', 'N13', 'N14', 'N15', 'N16', 'N17', 'N18', 'N19', 'N20',
            # Negative examples (facilities that don't match)
            'G1', 'G1', 'G1', 'G2', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7',
            'N1', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9',
            'W1', 'W2', 'W3', 'W4', 'W5'
        ],
        'Record_ID B': [
            # Positive examples
            'N1', 'N2', 'N3', 'N4', 'N6', 'G13', 'N7', 'N8', 'N9', 'N10',
            'W1', 'W7', 'W1', 'W2', 'W7', 'W3', 'W5', 'W6', 'W11', 'W8',
            'W10', 'W13', 'W7', 'W7', 'W2', 'W2', 'W11', 'W5', 'W6', 'W8',
            # Negative examples
            'N3', 'N4', 'N6', 'N3', 'N5', 'N5', 'N6', 'N7', 'N8', 'N9',
            'N5', 'N6', 'N7', 'N8', 'N9', 'N10', 'N11', 'N12', 'N13', 'N14',
            'W5', 'W6', 'W7', 'W8', 'W9'
        ],
        'Same Entity': [
            # Positive
            'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes',
            'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes',
            'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes',
            # Negative
            'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No',
            'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No',
            'No', 'No', 'No', 'No', 'No'
        ]
    })
    
    return gov_df, ngo_df, whatsapp_df, ground_truth


def get_real_multi_entity_data():
    """
    Returns real data WITH person entities extracted from farmer names.
    
    This demonstrates EARE multi-entity: facilities + persons
    """
    
    # Get facility data
    gov_df, ngo_df, whatsapp_df, ground_truth = get_real_data()
    
    # Extract person data from NGO dataset
    # (Farmers are mentioned in the NGO data)
    ngo_full = pd.DataFrame({
        'RecordID': ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10',
                     'N11', 'N12', 'N13', 'N14', 'N15', 'N16', 'N17', 'N18', 'N19', 'N20'],
        'FarmerID': list(range(1, 21)),
        'FarmerName': [
            'Mphatso Banda', 'Mfatsso B.', 'John Phiri', 'Grace Tembo', 'Sarah Mwale',
            'James Kachingwe', 'Mary Phiri', 'Patrick Banda', 'Alice Nkhoma', 'David Chirwa',
            'Grace T.', 'Mphatso B', 'John P.', 'Sarah M.', 'James K.',
            'Mary P.', 'Patrick B.', 'Alice N.', 'David C.', 'Grace Tembo'
        ],
        'Gender': ['F', 'F', 'M', 'F', 'F', 'M', 'F', 'M', 'F', 'M',
                   'F', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
        'Phone': [
            '265882001001', None, '889002300', '991234600', '888900123',
            '999111222', '888777666', '777666555', '666555444', '555444333',
            '991234600', '265882001001', '889002300', '888900123', '999111222',
            '888777666', '777666555', '666555444', '555444333', '991234600'
        ],
        'District': [
            'Mbale', 'Mbhale', 'Rakai', 'Rakai', 'Tororo Urban',
            'Rakai', 'Chikwawa', 'Tororo Rural', 'Masenjere', 'Tororo',
            'Kanjedza', 'Mbale', 'Shire', 'Rakai', 'Mbale',
            'Mbhale', 'Tororo', 'Bangula', 'Chikwawa', 'Masenjere'
        ]
    })
    
    # Create person entities from farmer data
    persons_ngo_df = ngo_full[['FarmerID', 'FarmerName', 'Gender', 'Phone', 'District']].copy()
    persons_ngo_df['RecordID'] = 'NP' + persons_ngo_df['FarmerID'].astype(str)
    persons_ngo_df = persons_ngo_df[['RecordID', 'FarmerName', 'Gender', 'Phone', 'District']]
    
    # Extract persons from WhatsApp data
    persons_wa_df = pd.DataFrame({
        'RecordID': ['WP1', 'WP2', 'WP3', 'WP4', 'WP5', 'WP6', 'WP7', 'WP8', 'WP9', 'WP10',
                     'WP11', 'WP12', 'WP13', 'WP14', 'WP15'],
        'ContactName': [
            'Mphatso Banda', 'Mfatsso B', 'Grace T Area25', 'Grace T', 'Peter K',
            'Mary P', 'John P', 'Alice N', 'Sarah M', 'James K',
            'Patrick B', 'David C', 'Mphatso B.', 'Grace Tembo', 'Mary Phiri'
        ],
        'Phone': [
            '265882001001', '882001000', '991234600', '991234600', '888900456',
            '888777666', '889002300', '666555444', '888900123', '999111222',
            '777666555', '555444333', '265882001001', '991234600', '888777666'
        ],
        'District': [
            'Mbale', 'Mbhale', 'TRR', 'Tororo Rural', 'MK',
            'Chikwawa', 'Rakai', 'Masenjere', 'Tororo', 'Kanjedza',
            'Tororo', 'Masenjere', 'Mbale', 'Tororo', 'Mbhale'
        ]
    })
    
    # Configure extra entity sources
    extra_entity_sources = [
        # Persons from NGO data
        {
            "df": persons_ngo_df,
            "entity_type": "person",
            "source_system": "NGO_Farmers",
            "name_col": "FarmerName",
            "district_col": "District",
            "phone_col": "Phone",
            "record_id_col": "RecordID",
            "extra_attribute_cols": ["Gender"]
        },
        # Persons from WhatsApp data
        {
            "df": persons_wa_df,
            "entity_type": "person",
            "source_system": "WhatsApp_Contacts",
            "name_col": "ContactName",
            "district_col": "District",
            "phone_col": "Phone",
            "record_id_col": "RecordID",
            "extra_attribute_cols": []
        }
    ]
    
    return gov_df, ngo_df, whatsapp_df, ground_truth, extra_entity_sources
