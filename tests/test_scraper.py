import pytest
import pandas as pd
from scripts.scrape_data import combine_text, _extract_name, _extract_title

def test_combine_text():
    paragraphs = ["First paragraph.", "Second paragraph."]
    assert combine_text(paragraphs) == "First paragraph. Second paragraph."
    
def test_extract_name():
    assert _extract_name("John Mulaney: Kid Gorgeous at Radio City") == "John Mulaney"
    assert _extract_name("Dave Chappelle’s Equanimity") == "Dave Chappelle’s"
    assert _extract_name("Ricky Gervais: Humanity") == "Ricky Gervais"
    assert _extract_name("Bo Burnham (2016)") == "Bo Burnham"

def test_extract_title():
    # Tag | Names | Year
    row1 = pd.Series({
        "CleanTag": "John Mulaney: Kid Gorgeous at Radio City",
        "Names": "John Mulaney",
        "Year": "2018"
    })
    assert _extract_title(row1) == "Kid Gorgeous at Radio City"
    
    row2 = pd.Series({
        "CleanTag": "Dave Chappelle’s Equanimity",
        "Names": "Dave Chappelle’s",
        "Year": "2017"
    })
    assert _extract_title(row2) == "Equanimity"
