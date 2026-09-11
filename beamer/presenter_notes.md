## 1. HeatPath-GNN

Today I will explain a particle-allocation problem. We have a fixed packed bed and a limited amount of highly conductive material. The question is where to place that material to improve heat transfer. We first solve and optimize a physical thermal network, then train a graph neural network to propose allocations for another packing. The results cover thirty independent realizations of one packing specification.

## 2. Motivation: conductive material is a limited resource

The motivation is to improve transport without increasing the conductive-material budget. An effective conductivity describes the whole bed, but it hides the individual contacts through which heat moves. The design variable here is the assignment of material to particles. We keep the geometry fixed while comparing assignments, so any difference between methods is attributable to allocation within this model. Practical manufacturing of a prescribed assignment remains a separate question.

## 3. The allocation problem

This schematic illustrates the hypothesis, not a measured packing or an optimal solution. Both arrangements contain the same number of orange particles. Connecting conductive particles could reduce resistance, but the whole network, including interfaces with low-conductivity particles, determines the result. The actual problem has five hundred particles and three size classes. We enforce the budget separately within each class, which also fixes the high-conductivity solid volume.

## 4. Thirty packings with the same nominal porosity

Each realization contains the same exact size distribution and occupies the same fixed-volume cell. Nominal porosity is calculated from the sum of sphere volumes divided by cell volume. With overlapping DEM spheres, this nominal quantity is not exactly the geometrical void fraction of their union. The modulus is deliberately softened. The final packings pass the particle-count, diameter, porosity, kinetic-energy and wall-spanning checks. We vary the random initialization, not the physical specification.

## 5. Contact conductance links temperature difference to heat rate

Here i and j identify particles, while w identifies a thermal wall. G is a conductance in watts per kelvin. Multiplying it by a temperature difference gives a heat rate in watts. For example, G equal to 0.01 W/K and a difference of 2 K gives 0.02 W. Material conductivity k has different units, W/(m K), because conductance also contains the contact geometry. Every internal contact contributes opposite heat rates to its two endpoints. The balance therefore conserves heat, and the net wall heat rate gives the effective bed conductivity. Fluid-gap conduction, radiation and convection are excluded.

## 6. Thermal conductance follows contact size and material

The contact radius comes from the Hertz geometrical relation between overlap and effective radius. The thermal model then adds two half-space spreading resistances, each equal to one over four k a. For identical conductivities this gives G equals two k a. An isothermal wall has zero spreading resistance on its side, leaving four k a. The corrected prefactor is used throughout this campaign. The formulas are idealizations, especially when the contact radius is not small compared with the particle radius.

## 7. Every method uses the same conductive-material budget

The optimization is discrete: a particle is assigned either low or high conductivity. The constraints are exact within each size class, not just over the total count. Because all particles within a class share their diameter, fixing those counts also fixes the conductive-material volume. We do not alter positions, radii or contact topology when comparing allocations on a packing. This isolates the value of arranging the material intelligently.

## 8. Reference methods select which 50 particles become conductive

All three methods produce particle selections, not temperature predictions. We assign high conductivity to the selected fifty particles and low conductivity to the remaining four hundred and fifty, then solve the thermal problem. The degree and heat-throughput rules give one allocation each. Random sampling gives one hundred allocations, and its reported conductivity is their mean. Quotas are applied separately within each size class. Heat-throughput ranking sorts the all-low thermal heat-throughput feature. It does not run a shortest-path algorithm.

## 9. Swap search starts from the better reference allocation

The starting point is a complete fifty-particle selection. We compare the degree and heat-throughput selections and initialize every run with the better one; a tie favours degree. Each proposal samples a size class and then one selected and one unselected particle within it. Swapping their labels preserves the quota exactly. The code accepts an increase larger than a relative tolerance of 1e-12. All five runs have the same initial assignment but independent random proposal sequences. Their final selections supply the learning targets, and the best of their final conductivities supplies the search reference. This is not a proven global optimum.

## 10. Where the methods enter training, prediction and evaluation

There are three distinct roles, and this diagram separates them. During training, we first calculate degree and solve the all-low-conductivity thermal network to obtain temperature and heat-throughput features. Degree ranking and heat-throughput ranking turn two of those quantities into complete candidate allocations. The better allocation initializes all five swap-search runs. Their final selection frequencies provide the targets. The dashed arrow is essential: features also go directly into model fitting; they are not used only to initialize search. At prediction time we calculate the same features for the new packing, run the trained GNN, enforce the size quotas and solve the proposed thermal allocation. We do not need to execute the separate ranking allocation methods or swap search to produce that GNN allocation. For the scientific evaluation, we additionally execute those methods on the held-out packing to assess how much value the learned allocation adds. Edge features go to the GNN only; both models receive the same particle features.

## 11. Degree and heat throughput have three distinct roles

Degree is a neighbour count and heat throughput is a sum of absolute contact heat rates from the all-low thermal solve. Those are numerical features, available on both training and new packings. Degree ranking and heat-throughput ranking are allocation methods that sort those features within each size class and enforce the conductive-particle quotas. We use the resulting allocations to initialize the training-data search and as independent evaluation references. Using these features in a GNN does not force the GNN to reproduce either ranking: it can combine them with other particle information and contact messages. The MLP checks how much learning from the particle features achieves without those messages and edge features. For practical GNN prediction the features remain necessary, while running the reference allocation methods is optional benchmarking.

## 12. All-low thermal solve before material allocation

Homogeneous refers only to the material conductivity, not to the geometry or the temperature field. We set all particles to low conductivity and solve once with the imposed wall temperatures. T superscript zero is the resulting baseline temperature. The throughput score sums the magnitudes of the heat rates on particle-particle contacts incident on a node; the implementation does not add wall heat rates to this score. It is not net heat accumulation, which is zero at steady state. In a simple interior chain it counts incoming and outgoing heat, so it would equal twice the transmitted heat rate. This score ranks the simple heat-throughput reference and also becomes a normalized input to the learned models.

## 13. The GNN uses geometry and the all-low thermal solution

The baseline quantities are normalized before entering the network. Anchored means connected through the thermal graph to an imposed-temperature boundary. Contact degree counts neighbouring particles, while the separate wall flags describe boundary contacts. The GNN passes information over the actual contact graph for three layers and uses the six edge features. The MLP sees the same eight node features but has no graph message passing or edge features. Consequently the comparison measures the benefit of this graph-based model and its contact information together. Dataset-wide feature standardization uses only the training packings.

## 14. Training targets describe how often search selects each particle

The target y is the fraction of final search runs that select a particle. It can take values zero, 0.2, 0.4, 0.6, 0.8 or one. It is an empirical selection frequency from five runs, not a proof that this particle belongs to every optimal allocation. The listwise ranking term normalizes these targets within each size class and encourages larger scores for more frequently selected particles. Binary cross entropy adds the mean negative value of y log p plus one minus y times log one minus p. It therefore penalizes confident disagreement with the soft target. The logits are converted to p with a sigmoid for interpreting this loss; p is not independently thresholded to assign materials.

## 15. Scoring assigns priorities, then ranking enforces the budget

Scoring is a forward evaluation of the trained neural network. It produces numbers attached to particles. Scores need not be on the same scale for different training repeats, so the ensemble combines within-size-class ranks. We select the prescribed counts after combining the ranks. This guarantees the same material budget as the reference methods. Finally, the thermal solver evaluates the proposed allocation. The GNN does not directly predict the reported effective conductivity.

## 16. Each test packing is withheld from model fitting

The independent unit is the packing, not an individual particle. Splitting particles from the same bed between training and testing would expose almost the same structure on both sides. Here we hold out a whole packing, use another for validation and fit on the remaining twenty-eight. We repeat the process thirty times. These folds share training data, so they are not independent training experiments. This does not establish transfer to a new porosity, particle-size distribution, system size or material contrast.

## 17. Gain and recovery answer different questions

Gain normalizes the change by the random-allocation conductivity. Recovery instead divides by the available improvement found by the search. For example, a model that improves conductivity by thirty percent when search improves it by about fifty-two percent recovers roughly fifty-eight percent of the gain. The reported values are averages of per-packing ratios, so dividing displayed ensemble means will not necessarily reproduce them exactly. R-squared is not the performance measure for this discrete allocation task.

## 18. GNN improves conductivity by 30\% over random

The degree rule adds almost no average benefit. A homogeneous heat-throughput ranking is better, but both learned methods improve on it. The GNN delivers thirty percent mean gain relative to random assignment. Search still achieves substantially more, at fifty-one point eight percent. Each point is a packing, and the black summary marks show the mean and sample standard deviation. These results support learning useful allocation structure while also showing room for improvement.

## 19. Absolute conductivity remains below the search reference

This slide gives the absolute scale of the conductivity changes. The mean rises from approximately 0.228 watts per metre kelvin for random assignment to 0.297 for the GNN. Search reaches approximately 0.347. The thin connecting lines preserve pairing across methods. These absolute numbers belong to the corrected contact-only model and should not be mixed with values from the earlier manuscript, which used a different thermal prefactor and packing-generation implementation.

## 20. GNN beats MLP on 29 of 30 held-out packings

The GNN outperforms the MLP on twenty-nine of the thirty withheld packings. Its mean recovery is fifty-eight percent, with a standard deviation of nine point eight percentage points. This is descriptive evidence across our realizations, not a claim of independent folds or universal superiority. The gap to the search reference is still substantial. Both successful and less successful packings are included in this plot.

## 21. Contact-network diagnostics help interpret allocation

The first diagnostic measures what fraction of summed absolute particle-contact heat rates occurs on contacts whose endpoints are both high-conductivity particles. This is not the fraction of net wall heat because heat can traverse multiple contacts. The second diagnostic asks whether the high-conductivity subgraph alone spans the two thermal boundaries. Low-conductivity particles can still participate in useful paths, so a spanning high-only cluster is not required for heat transfer. The random visualization uses the sampled assignment closest to the random mean.

## 22. Numerical checks pass, but contact sizes need scrutiny

The conservation and mechanical acceptance checks pass. However, numerical convergence does not establish physical validity. The contact-radius ratio has a median of about 0.19 and reaches about 0.35. Those contacts are not uniformly small relative to the particles. The half-space constriction approximation and the softened mechanical generation therefore deserve further assessment. This pooled histogram is a geometry diagnostic; individual contacts from the same bed are correlated.

## 23. The numerical overlap floor has little influence

A small overlap floor regularizes contact-radius evaluation. We repeat the thermal evaluation with different floor values while keeping the material assignments fixed. The largest absolute relative change is less than one thousandth of a percent. That is reassuring about this numerical choice. It does not resolve the physical contact-size concern on the previous slide, because we did not regenerate the beds under another modulus, pressure or contact model.

## 24. Fast scoring is one part of the complete prediction procedure

Fast scoring means producing particle-priority scores with the already trained model. The quoted ensemble scoring time is the sum of forward-evaluation timings for the three model repeats. It excludes rank aggregation and exact quota selection as well as baseline solving, feature construction and final verification. Search uses nearly fifteen thousand candidate thermal solves per packing, which motivates replacing repeated search with learned priorities. However, this timing comparison is not a measurement of complete design time. Teacher-data generation and model training must also be considered when assessing amortized cost.

## 25. Conclusions and next research steps

The main contribution is a constrained allocation workflow that combines a physical contact network with graph learning. Thirty independent packings give a clearer assessment than the earlier ten-case study. The results are promising, but the physical modelling assumptions and the remaining search-performance gap should shape the next work. A practical next algorithmic test is to initialize a short swap search with the GNN selection and compare its gain per thermal solve against random initialization. Physical sensitivity and external transfer should be evaluated before making broader claims.