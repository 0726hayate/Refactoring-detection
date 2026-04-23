#!/bin/bash
# Upload the 3 large data files to a HuggingFace Dataset repo.
#
# Prereqs:
#   - HuggingFace account + write-scope access token (see README)
#   - Dataset repo created at https://huggingface.co/new-dataset
#   - `pip install -U huggingface_hub` (already installed on this server)
#
# Run:
#   export HF_TOKEN="hf_xxxxx"           # your write-scope token
#   export HF_DATASET_REPO="amrarhs/t6-refactoring-detection-data"   # owner/repo
#   bash upload_to_huggingface.sh
#
# Uploads each file with progress; files appear in the dataset's "Files" tab.
# Total ~6.2 GB; expect 10-30 min on a typical research-server uplink.
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ] || [ -z "${HF_DATASET_REPO:-}" ]; then
    echo "ERROR: set HF_TOKEN and HF_DATASET_REPO env vars first."
    echo "  export HF_TOKEN='hf_xxxxx'"
    echo "  export HF_DATASET_REPO='amrarhs/t6-refactoring-detection-data'"
    exit 1
fi

# Files to upload (path on this server → name in the HF dataset)
declare -A FILES=(
    ["/home/25fxvd/summer2025/0807/dspy/langchain_pipeline/commit_level_train_k15_headline.json"]="commit_level_train_k15_headline.json"
    ["/home/25fxvd/summer2025/0807/dspy/langchain_pipeline/commit_level_test_k15_headline.json"]="commit_level_test_k15_headline.json"
    ["/home/25fxvd/summer2025/0807/dspy/langchain_pipeline/commit_level_benchmark_k15_headline.json"]="commit_level_benchmark_k15_headline.json"
    ["/home/25fxvd/summer2025/ELEC825/evaluation_data_v3_ast/contrastive_v5_unixcoder_holdout/projected/unixcoder_java_projected.npy"]="java_retrieval/unixcoder_java_projected.npy"
    ["/home/25fxvd/summer2025/ELEC825/evaluation_data_v3_ast/embeddings/unixcoder_java_meta.json"]="java_retrieval/unixcoder_java_meta.json"
    ["/home/25fxvd/summer2025/ELEC825/evaluation_data_v3_ast/java/java_records_all.pkl"]="java_retrieval/java_records_all.pkl"
)

for src in "${!FILES[@]}"; do
    if [ ! -f "$src" ] && [ ! -L "$src" ]; then
        echo "  SKIP (missing): $src"
        continue
    fi
    dest="${FILES[$src]}"
    size=$(du -h "$src" | cut -f1)
    echo ""
    echo "Uploading $src ($size) → $HF_DATASET_REPO/$dest"
    python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='$HF_TOKEN')
api.upload_file(
    path_or_fileobj='$src',
    path_in_repo='$dest',
    repo_id='$HF_DATASET_REPO',
    repo_type='dataset',
)
print('  done')
"
done

echo ""
echo "All done. Verify at: https://huggingface.co/datasets/$HF_DATASET_REPO/tree/main"
