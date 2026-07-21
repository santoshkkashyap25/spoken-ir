import pytest
import pandas as pd
from ai.topics import avg_rating_for_topic, TOPIC_LABELS

def test_avg_rating_for_topic():
    # Create a mock corpus with some topics and ratings
    data = {
        "rating": [8.0, 9.0, pd.NA, 7.5],
        TOPIC_LABELS[0]: [0.8, 0.1, 0.2, 0.9],
        TOPIC_LABELS[1]: [0.1, 0.8, 0.7, 0.05],
    }
    # Add remaining topics as 0
    for label in TOPIC_LABELS[2:]:
        data[label] = [0.0] * 4
        
    corpus = pd.DataFrame(data)
    
    # TOPIC_LABELS[0] is dominant for index 0 and 3. Ratings are 8.0 and 7.5 (avg 7.75)
    avg_0 = avg_rating_for_topic(TOPIC_LABELS[0], corpus=corpus)
    assert avg_0 == 7.75
    
    # TOPIC_LABELS[1] is dominant for index 1 and 2. Rating for 1 is 9.0, for 2 is NA (avg 9.0)
    avg_1 = avg_rating_for_topic(TOPIC_LABELS[1], corpus=corpus)
    assert avg_1 == 9.0
    
    # Non-existent topic
    assert avg_rating_for_topic("invalid_topic", corpus=corpus) is None

def test_avg_rating_empty_corpus():
    corpus = pd.DataFrame(columns=["rating"] + TOPIC_LABELS)
    assert avg_rating_for_topic(TOPIC_LABELS[0], corpus=corpus) is None
    
def test_avg_rating_missing_rating_col():
    corpus = pd.DataFrame(columns=TOPIC_LABELS)
    # Insert one row
    corpus.loc[0] = [1.0] + [0.0] * (len(TOPIC_LABELS) - 1)
    assert avg_rating_for_topic(TOPIC_LABELS[0], corpus=corpus) is None
