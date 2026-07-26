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
    assert _extract_title(
        "John Mulaney: Kid Gorgeous at Radio City", "John Mulaney", "2018"
    ) == "Kid Gorgeous at Radio City"
    
    assert _extract_title(
        "Dave Chappelle's Equanimity", "Dave Chappelle's", "2017"
    ) == "Equanimity"
