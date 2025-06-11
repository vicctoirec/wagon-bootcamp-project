import re

def clean_title(title):
    return title.split(" - ")[0].strip()

def clean_text(text):
    if not isinstance(text, str):
        return None
    text = text.replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([,.!:;?])', r'\1', text)
    return text.strip()
