for case_dir in packing_seed_*; do
    if [[ -d "$case_dir" ]]; then
        echo "Running $case_dir"
        python3 solve_packing_heat_transfer.py "$case_dir" \
          --output thermal_results_contact \
          --k-low 1.0 \
          --k-fluid 0.0 \
          --t-hot 301 \
          --t-cold 300
    fi
done
