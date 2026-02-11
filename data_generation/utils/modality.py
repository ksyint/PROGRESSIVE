def infer_modality(image_path: str) -> str:
    path_lower = image_path.lower()
    
    if 'xray' in path_lower or 'radiograph' in path_lower:
        return 'xray'
    elif 'ct' in path_lower:
        return 'ct'
    elif 'mri' in path_lower or 'mr' in path_lower:
        return 'mri'
    elif 'ultrasound' in path_lower or 'us' in path_lower:
        return 'ultrasound'
    else:
        return 'unknown'

def get_modality_from_metadata(metadata: dict) -> str:
    if 'modality' in metadata:
        return metadata['modality'].lower()
    elif 'Modality' in metadata:
        return metadata['Modality'].lower()
    else:
        return 'unknown'
