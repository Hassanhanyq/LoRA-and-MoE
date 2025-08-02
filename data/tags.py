import json


file_tag_map = {
    "QAcv.json": "cs.CV",
    "QAcl.json": "cs.CL",
    "QAro.json": "cs.RO",
    "QAlg.json": "cs.LG"
}

for file_name, tag in file_tag_map.items():
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    
    for entry in data:
        entry["tag"] = tag

    
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
