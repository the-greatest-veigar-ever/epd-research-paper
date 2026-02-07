
import pandas as pd
import json

try:
    df = pd.read_parquet("ai/benchmarks/HalluLens/data/wiki_data/sample.jsonl")
    # HalluLens expects: title, document, h_score_cat, etc.
    # The HF dataset has 'title', 'text', 'url', 'wikipedia_id'
    
    # Take top 50
    df = df.head(50)
    
    records = []
    for _, row in df.iterrows():
        records.append({
            "title": row['title'],
            "document": row['text'],
            "h_score_cat": 5, # Mock score
            "pageid": row.get('id', 0),
            "revid": 0,
            "description": "Sample description",
            "categories": []
        })
        
    with open("ai/benchmarks/HalluLens/data/wiki_data/doc_goodwiki.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    print(f"Created doc_goodwiki.jsonl with {len(records)} records")
except Exception as e:
    print(f"Error: {e}")
