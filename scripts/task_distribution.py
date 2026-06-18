import pandas as pd
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import torch


VisNec_CSV_PATH = "results/dataset_visnec_scores.csv"
ORIGINAL_JSON_PATH = "llava-v1.5-instruct/llava_v1_5_mix665k.json"
OUTPUT_JSON_PATH = "llava_v1.5-7b-top15.json"

NUM_CLUSTERS = 20           
EMBEDDING_MODEL = 'all-MiniLM-L6-v2' 


TOP_RATIO = 0.15   


def extract_clean_instruction(conversations):
    if not isinstance(conversations, list): return ""
    for turn in conversations:
        if turn.get('from') == 'human':
            return turn.get('value', "").replace('<image>', '').replace('\n', ' ').strip()
    return ""

def main():
    print(f">>> [1/6] loading the original file")
    
    df_scores = pd.read_csv(VisNec_CSV_PATH, low_memory=False)
    
    if 'row_idx' not in df_scores.columns:
        df_scores['row_idx'] = df_scores.index
    
    df_scores = df_scores[['row_idx', 'visnec']]

    df_meta = pd.read_json(ORIGINAL_JSON_PATH)
    df_meta['row_idx'] = df_meta.index 

    
    df_merged = pd.merge(
        df_scores, 
        df_meta[['row_idx', 'id', 'image', 'conversations']], 
        on='row_idx', 
        how='inner'
    )
    tqdm.pandas(desc="extract prompt")
    df_merged['instruction'] = df_merged['conversations'].progress_apply(extract_clean_instruction)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = SentenceTransformer(EMBEDDING_MODEL, device=device)
    
    embeddings = encoder.encode(
        df_merged['instruction'].tolist(),
        batch_size=1024,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    kmeans = MiniBatchKMeans(n_clusters=NUM_CLUSTERS, random_state=42, batch_size=2048, n_init='auto')
    df_merged['cluster_id'] = kmeans.fit_predict(embeddings)

  
    
    final_dfs = []

    grouped = df_merged.groupby('cluster_id')
    
    for cid, group in grouped:
        total_in_cluster = len(group)
        
        
        n_keep = int(total_in_cluster * TOP_RATIO)
        
    
        if n_keep == 0 and total_in_cluster > 0:
            n_keep = 1
        subset = group.nlargest(n_keep, 'visnec')
        final_dfs.append(subset)

    df_selected = pd.concat(final_dfs)
    
  
    df_selected = df_selected.sort_values('row_idx')
    
    print(f"  finished! selected samples: {len(df_selected)} (original:: {len(df_selected)/len(df_merged):.2%})")



    
    raw_list = df_selected.to_dict(orient='records')
    cleaned_list = []
    text_only_count = 0
    
    for item in tqdm(raw_list, desc="Exporting"):
        clean_item = {}
        clean_item['id'] = str(item['id']).strip()
        
        if 'image' in item:
            val = item['image']
            if val and str(val).lower() != 'nan' and str(val).strip() != "":
                clean_item['image'] = str(val).strip()
            else:
                text_only_count += 1
        
        clean_item['conversations'] = item['conversations']
        cleaned_list.append(clean_item)

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(cleaned_list, f, indent=2, ensure_ascii=False)

    print(f"\n✅ task finished")
    print(f"   - outputfile: {OUTPUT_JSON_PATH}")
    print(f"   - final selected amount: {len(cleaned_list)}")

if __name__ == "__main__":
    main()
