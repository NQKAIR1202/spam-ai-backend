import re

def process_and_extract(text: str):
    """
    Pre-processes raw email text and extracts numerical features for DS1 SVM model.
    The order of features MUST match the training phase: [url, dangerous, common, other, html]
    """
    # Convert to lowercase for consistent matching
    text = text.lower()

    # 1. Handle URLs
    url_pattern = r'http[s]?://\S+|www\.\S+'
    has_url = 1 if re.search(url_pattern, text) else 0
    clean_text = re.sub(url_pattern, ' http_url_token ', text) 
    
    # 2. Handle Executable Files (Dangerous)
    exe_pattern = r'\.(exe|bat|msi|sh|vbs|jar)'
    has_exe = 1 if re.search(exe_pattern, clean_text) else 0
    clean_text = re.sub(exe_pattern, ' file_exe_token ', clean_text)
    
    # 3. Handle Common Files
    doc_pattern = r'\.(pdf|docx|doc|txt|zip|png|jpg|xlsx|pptx)'
    has_doc = 1 if re.search(doc_pattern, clean_text) else 0
    clean_text = re.sub(doc_pattern, ' file_common_token ', clean_text)

    # 4. Handle HTML Tags
    html_pattern = r'<[^>]+>'
    has_html = 1 if re.search(html_pattern, text) else 0
    clean_text = re.sub(html_pattern, ' ', clean_text)

    # Final cleanup: remove special characters and extra spaces
    clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
    clean_text = " ".join(clean_text.split())

    # Features: order must be exact as defined in prepare_data(df) during training
    # [has_url, has_dangerous_file, has_common_file, has_other_file, has_html]
    features = [has_url, has_exe, has_doc, 0, has_html]
    
    return clean_text, features