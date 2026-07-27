import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion.config import settings

outputs_dir = os.path.join(os.path.dirname(__file__), '..', 'E Books Updated_md', 'outputs')
records = []
for fname in os.listdir(outputs_dir):
    if not fname.endswith('_questions.jsonl'):
        continue
    with open(os.path.join(outputs_dir, fname)) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

holdout_set = set(settings.holdout_folder_list)
print('Holdout folders from config:', holdout_set)
print()

# True holdout test records (what the comparison SHOULD have used)
test_recs = [
    r for r in records
    if r.get('project_id') in holdout_set
    and r.get('empirical_label') not in ('escalation_required',)
]
print(f'True holdout test records: {len(test_recs)}')
print()

# All graph_dense records and where they come from
gd_all = [r for r in records if r.get('empirical_label') == 'graph_dense']
print(f'All graph_dense records across ALL files: {len(gd_all)}')
for r in gd_all[:5]:
    pid = r.get('project_id')
    ih = r.get('is_holdout')
    q = r.get('question', '')[:80]
    print(f'  project_id={pid} | is_holdout={ih} | q={q}')

print()

# What the comparison script ACTUALLY used as test set (fallback)
if not test_recs:
    fallback = [
        r for r in records
        if r.get('empirical_label') not in ('escalation_required',)
    ]
    print(f'FALLBACK used: {len(fallback)} records (the FULL training set!)')
    gd_in_fallback = [r for r in fallback if r.get('empirical_label') == 'graph_dense']
    print(f'graph_dense in fallback (training data): {len(gd_in_fallback)}')
    print()
    print('CONCLUSION: The model was tested on its own training data.')
    print('100% recall is data leakage, not generalization.')
