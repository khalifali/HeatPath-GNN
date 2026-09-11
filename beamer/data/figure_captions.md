**01_conductivity**: Conductivity by allocation method. Each point is one packing; grey lines connect the same packing. Black marks show mean +/- sample SD. Random denotes each packing's random-allocation mean.

**02_allocation_gain**: Per-packing improvement relative to that packing's random mean; summary marks show mean +/- sample SD. This uses equal material-volume quotas in all methods.

**03_ml_recovery**: Held-out-packing results. Recovery = 100 (k_model - k_random)/(k_best_search - k_random). The reference is the best fixed-budget search, not a proven global optimum. Models use identical node features and packing-level splits.

**04_mechanisms**: Mechanisms across all packings. Random is the sampled allocation closest to the packing's random mean. High-high heat fraction uses absolute particle-contact heat rates. Black marks are ensemble means.

**05_contact_geometry**: Pooled contact-size distribution. Contacts within a packing are correlated; this histogram is a geometry diagnostic, not an independent-sample statistical test. Large ratios warrant checking the small-contact approximation.

**06_floor_sensitivity**: Numerical overlap-floor sensitivity for fixed allocations: mean +/- sample SD over packings. Nominal floor is 1e-8. This is not a physical stiffness sensitivity or experimental validation.