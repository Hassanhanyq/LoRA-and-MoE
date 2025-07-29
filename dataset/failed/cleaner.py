import re

def remove_abstract_and_intro(text: str) -> str:
    """
    Removes abstract and introduction from text.
    Tries to return the body starting from section 2.
    """
    lower_text = text.lower()

    
    abstract_match = re.search(r"(abstract)\b", lower_text)
    if not abstract_match:
        return text

    after_abstract = text[abstract_match.end():]

    
    intro_match = re.search(r"\n\s*1\.?\s*introduction\b", after_abstract.lower())
    if not intro_match:
        return after_abstract

    after_intro = after_abstract[intro_match.end():]

    
    section_2_match = re.search(r"\n\s*2\.?\s+[A-Z][^\n]{3,}", after_intro)
    if not section_2_match:
        return after_intro

    return after_intro[section_2_match.start():].strip()
