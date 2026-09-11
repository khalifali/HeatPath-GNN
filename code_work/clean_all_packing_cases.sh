#!/usr/bin/env bash
set -euo pipefail

# Removes only case directories named packing_seed_<digits> directly beneath
# the current directory. Input scripts, Python programs and other files remain.

project_dir=$(pwd -P)
declare -a targets=()

while IFS= read -r -d '' path; do
    name=$(basename "$path")
    parent=$(dirname "$(realpath -m "$path")")
    if [[ "$parent" == "$project_dir" && "$name" =~ ^packing_seed_[0-9]+$ ]]; then
        targets+=("$project_dir/$name")
    fi
done < <(find "$project_dir" -mindepth 1 -maxdepth 1 -type d \
         -name 'packing_seed_*' -print0)

if (( ${#targets[@]} == 0 )); then
    echo "No packing_seed_<digits> directories found in: $project_dir"
    exit 0
fi

echo "Packing directories selected for deletion:"
printf '  %s\n' "${targets[@]}"

if [[ "${1:-}" != "--yes" ]]; then
    echo
    echo "Preview only. Nothing was deleted."
    echo "Run again with --yes to delete these directories permanently:"
    echo "  bash clean_all_packing_cases.sh --yes"
    exit 0
fi

for target in "${targets[@]}"; do
    name=$(basename "$target")
    parent=$(dirname "$target")
    if [[ "$parent" != "$project_dir" || ! "$name" =~ ^packing_seed_[0-9]+$ ]]; then
        echo "Safety check failed for: $target" >&2
        exit 1
    fi
    rm -rf -- "$target"
    echo "Deleted: $target"
done

echo "All packing case directories were removed."
