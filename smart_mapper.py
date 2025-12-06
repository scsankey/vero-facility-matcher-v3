"""
smart_mapper.py
Smart Column Mapping with Fuzzy Matching and Synonym Detection
Human-in-the-Loop (HITL) Preview with Editing Capability
"""

from rapidfuzz import fuzz
import pandas as pd

class SmartColumnMapper:
    """
    Smart column mapper that uses fuzzy matching and synonym detection
    to automatically map columns with high confidence.
    """
    
    # Synonym dictionary for common column name variations
    COLUMN_SYNONYMS = {
        'RecordID': ['id', 'record_id', 'recordid', 'unique_id', 'uid', 'identifier', 
                     'record_number', 'no', 'number', '#'],
        
        'OfficialFacilityName': ['facility_name', 'facilityname', 'name', 'facility', 
                                 'official_name', 'officialname', 'site_name', 'sitename'],
        
        'FacilityName': ['facility_name', 'facilityname', 'name', 'facility', 
                        'site_name', 'sitename', 'location_name'],
        
        'RelatedFacility': ['facility', 'related_facility', 'relatedfacility', 
                           'facility_name', 'name', 'site'],
        
        'District': ['district', 'region', 'area', 'zone', 'location', 'province',
                    'county', 'municipality', 'district_name', 'districtname'],
        
        'DistrictNote': ['district', 'district_note', 'districtnote', 'location', 
                        'area', 'region_note', 'location_note'],
        
        'Phone': ['phone', 'telephone', 'tel', 'mobile', 'contact', 'phone_number',
                 'phonenumber', 'contact_number', 'cellphone', 'cell'],
        
        'AltName': ['alt_name', 'altname', 'alternate_name', 'alternatename', 
                   'alias', 'other_name', 'nickname'],
        
        'LocationNickname': ['nickname', 'location_nickname', 'locnickname', 
                            'local_name', 'informal_name'],
        
        'GPS_Lat': ['lat', 'latitude', 'gps_lat', 'gpslat', 'y', 'coord_y'],
        
        'GPS_Lon': ['lon', 'long', 'longitude', 'gps_lon', 'gpslon', 'x', 'coord_x'],
        
        'PersonName': ['person_name', 'personname', 'name', 'full_name', 'fullname',
                      'farmer_name', 'farmername', 'contact_name', 'contactname'],
        
        'FarmerName': ['farmer_name', 'farmername', 'farmer', 'grower_name', 
                      'producer_name', 'name'],
        
        'ContactName': ['contact_name', 'contactname', 'contact', 'name', 
                       'person_name', 'contact_person'],
        
        'Gender': ['gender', 'sex', 'm/f', 'male/female'],
        
        'Role': ['role', 'position', 'title', 'job_title', 'occupation']
    }
    
    def __init__(self, confidence_threshold=70):
        """
        Initialize smart mapper
        
        Args:
            confidence_threshold: Minimum confidence score (0-100) for auto-mapping
        """
        self.confidence_threshold = confidence_threshold
        
    def normalize_column_name(self, col_name):
        """Normalize column name for comparison"""
        if pd.isna(col_name):
            return ""
        return str(col_name).lower().strip().replace(' ', '_').replace('-', '_')
    
    def calculate_match_score(self, user_col, standard_col):
        """
        Calculate match score between user column and standard column
        
        Returns:
            score (0-100): Confidence score
            method (str): How the match was found
        """
        user_normalized = self.normalize_column_name(user_col)
        standard_normalized = self.normalize_column_name(standard_col)
        
        # 1. Exact match (after normalization)
        if user_normalized == standard_normalized:
            return 100, "exact_match"
        
        # 2. Check synonyms
        if standard_col in self.COLUMN_SYNONYMS:
            synonyms = [self.normalize_column_name(s) for s in self.COLUMN_SYNONYMS[standard_col]]
            if user_normalized in synonyms:
                return 95, "synonym_match"
        
        # 3. Fuzzy string matching
        # Use multiple algorithms and take the best score
        ratio_score = fuzz.ratio(user_normalized, standard_normalized)
        partial_score = fuzz.partial_ratio(user_normalized, standard_normalized)
        token_sort_score = fuzz.token_sort_ratio(user_normalized, standard_normalized)
        
        best_score = max(ratio_score, partial_score, token_sort_score)
        
        if best_score >= 90:
            return best_score, "high_fuzzy_match"
        elif best_score >= 70:
            return best_score, "medium_fuzzy_match"
        else:
            return best_score, "low_fuzzy_match"
    
    def auto_map_columns(self, df, required_fields, optional_fields=None):
        """
        Automatically map DataFrame columns to standard fields
        
        Args:
            df: DataFrame with user columns
            required_fields: dict of {standard_col: description}
            optional_fields: dict of {standard_col: description}
        
        Returns:
            mapping: dict of {standard_col: user_col}
            confidence: dict of {standard_col: confidence_score}
            method: dict of {standard_col: match_method}
        """
        if df is None:
            return {}, {}, {}
        
        optional_fields = optional_fields or {}
        all_fields = {**required_fields, **optional_fields}
        
        mapping = {}
        confidence = {}
        method = {}
        
        user_columns = list(df.columns)
        
        for standard_col in all_fields.keys():
            best_match = None
            best_score = 0
            best_method = None
            
            # Find best matching user column
            for user_col in user_columns:
                score, match_method = self.calculate_match_score(user_col, standard_col)
                
                if score > best_score:
                    best_score = score
                    best_match = user_col
                    best_method = match_method
            
            # Only auto-map if confidence is high enough
            if best_score >= self.confidence_threshold and best_match:
                mapping[standard_col] = best_match
                confidence[standard_col] = best_score
                method[standard_col] = best_method
            else:
                # Don't map if confidence too low
                mapping[standard_col] = "-- Skip --"
                confidence[standard_col] = best_score if best_match else 0
                method[standard_col] = "no_match"
        
        return mapping, confidence, method
    
    def get_mapping_quality(self, mapping, required_fields):
        """
        Assess overall quality of mapping
        
        Returns:
            quality (str): "excellent", "good", "needs_review", "poor"
            missing_required (list): List of required fields not mapped
        """
        missing_required = []
        
        for field in required_fields:
            if field not in mapping or mapping[field] == "-- Skip --":
                missing_required.append(field)
        
        if len(missing_required) == 0:
            return "excellent", missing_required
        elif len(missing_required) <= 1:
            return "good", missing_required
        elif len(missing_required) <= 2:
            return "needs_review", missing_required
        else:
            return "poor", missing_required


# Convenience function for use in app.py
def smart_auto_map(df, required_fields, optional_fields=None, confidence_threshold=70):
    """
    Convenience function to auto-map columns
    
    Usage:
        mapping, confidence, method = smart_auto_map(
            df, 
            {'RecordID': 'Unique ID', 'Name': 'Facility name'},
            {'Phone': 'Phone number'}
        )
    """
    mapper = SmartColumnMapper(confidence_threshold=confidence_threshold)
    return mapper.auto_map_columns(df, required_fields, optional_fields)


def get_mapping_summary(mapping, confidence, method):
    """
    Get human-readable summary of mapping
    
    Returns:
        summary (dict): Summary statistics
    """
    total = len(mapping)
    auto_mapped = sum(1 for v in mapping.values() if v != "-- Skip --")
    
    high_confidence = sum(1 for score in confidence.values() if score >= 90)
    medium_confidence = sum(1 for score in confidence.values() if 70 <= score < 90)
    low_confidence = sum(1 for score in confidence.values() if score < 70)
    
    exact_matches = sum(1 for m in method.values() if m == "exact_match")
    synonym_matches = sum(1 for m in method.values() if m == "synonym_match")
    
    return {
        'total_fields': total,
        'auto_mapped': auto_mapped,
        'unmapped': total - auto_mapped,
        'high_confidence': high_confidence,
        'medium_confidence': medium_confidence,
        'low_confidence': low_confidence,
        'exact_matches': exact_matches,
        'synonym_matches': synonym_matches,
        'mapping_rate': (auto_mapped / total * 100) if total > 0 else 0
    }
