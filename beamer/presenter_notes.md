## 1. HeatPath--GNN

Today I will explain a particle-allocation problem. We have a fixed packed bed and a limited amount of highly conductive material. The question is where to place that material to improve heat transfer. We first solve and optimize a physical thermal network, then train a graph neural network to propose allocations for another packing. The results cover thirty independent realizations of one packing specification.

## 2. Outline

The presentation follows five questions. Why does particle allocation matter? How do we convert the DEM packing into a thermal network? How do the reference methods and swap search generate strong allocations? What exactly does the GNN learn and predict? Finally, how much performance does it recover and at what computational cost?

## 3. Motivation and research question

This section defines the design problem. The packing and the amount of conductive material remain fixed. Only the particle material labels change.

## 4. Motivation: conductive material is a limited resource

Several packed-bed applications contain a real material-budget trade-off. In catalytic reactors, conductive inert pellets such as silicon carbide can spread heat and reduce hot spots, but each inert pellet replaces active catalyst. Adsorption beds use graphite or other conductive additives to improve heat removal, but the additives reduce the amount of active sorbent. Thermal-storage beds can combine inexpensive storage material with a smaller conductive fraction, but that fraction adds cost or displaces storage capacity. These applications motivate the same design question studied here: if the conductive fraction, geometry and particle-size distribution are fixed, which particle positions should receive the conductive material? The present model isolates conduction through the solid contact network; it does not yet represent a complete reactor, adsorption unit or thermal-storage cycle.

Sources: [Kulkarni et al. (2023), *Silicon carbide in catalysis*](https://doi.org/10.1080/01614940.2022.2025670); [Fayazmanesh et al. (2017), *Effective thermal conductivity modeling of consolidated sorption materials*](https://www.sfu.ca/~mbahrami/pdf/pdf/2017/Effective%20thermal%20conductivity%20modeling%20of%20consolidated%20sorption.pdf); Ma et al. (2023), *Numerical and experimental studies of packed-bed thermal energy storage*.

## 5. The allocation problem

This schematic illustrates the hypothesis, not a measured packing or an optimal solution. Both arrangements contain the same number of orange particles. Connecting conductive particles could reduce resistance, but the whole network, including interfaces with low-conductivity particles, determines the result. The actual problem has five hundred particles and three size classes. We enforce the budget separately within each class, which also fixes the high-conductivity solid volume.

## 6. Packing and thermal model

This section explains how DEM contacts become a conservative thermal network. The thermal solver provides the particle temperatures, contact heat rates and whole-bed effective conductivity used throughout the study.

## 7. Thirty packings with the same nominal porosity

Each realization contains the same exact size distribution and occupies the same fixed-volume cell. Nominal porosity is calculated from the sum of sphere volumes divided by cell volume. With overlapping DEM spheres, this nominal quantity is not exactly the geometrical void fraction of their union. The modulus is deliberately softened. The final packings pass the particle-count, diameter, porosity, kinetic-energy and wall-spanning checks. We vary the random initialization, not the physical specification.

## 8. Contact conductance links temperature difference to heat rate

Here i and j identify particles, while w identifies a thermal wall. G is a conductance in watts per kelvin. Multiplying it by a temperature difference gives a heat rate in watts. For example, G equal to 0.01 W/K and a difference of 2 K gives 0.02 W. Material conductivity k has different units, W/(m K), because conductance also contains the contact geometry. Every internal contact contributes opposite heat rates to its two endpoints. The balance therefore conserves heat, and the net wall heat rate gives the effective bed conductivity. Fluid-gap conduction, radiation and convection are excluded.

## 9. Thermal conductance follows contact size and material

The contact radius comes from the Hertz geometrical relation between overlap and effective radius. The thermal model then adds two half-space spreading resistances, each equal to one over four k a. For identical conductivities this gives G equals two k a. An isothermal wall has zero spreading resistance on its side, leaving four k a. The corrected prefactor is used throughout this campaign. The formulas are idealizations, especially when the contact radius is not small compared with the particle radius.

## 10. Allocation methods and teacher search

This section follows the experiment in chronological order. We first impose the wall temperatures and fix the conductive-material quota. The all-low thermal solve supplies baseline temperatures and contact heat rates. We then compare allocation rules, improve the best reference with same-size swaps, and use the searched allocations as training targets.

## 11. Thermal experiment and fixed material budget

We impose 301 K at the bottom wall and 300 K at the top wall. The lateral directions are periodic, and the present model includes steady conduction through particle contacts. The table fixes the conductive budget: ten, thirty and ten particles in the three size classes. The binary variable s_i records the material assigned to particle i. A value of one means high conductivity. The set B_b contains the particles in size class b, and the equation fixes their sum to the quota q_b. The optimization changes only these material labels and seeks the allocation with the largest whole-bed effective conductivity.

## 12. All-low thermal solve before material allocation

This is the first thermal calculation for each packing. We assign the low conductivity to every particle, impose 301 K at the bottom wall and 300 K at the top wall, and solve the steady contact network. The solution gives a baseline temperature for each particle and a baseline heat rate for each contact. The heat-throughput score sums the absolute particle-contact heat rates incident on a particle. These baseline quantities are available before any high-conductivity particles are selected.

## 13. Reference methods select which 50 particles become conductive

All three methods produce particle selections, not temperature predictions. We assign high conductivity to the selected fifty particles and low conductivity to the remaining four hundred and fifty, then solve the thermal problem. The degree and heat-throughput rules give one allocation each. Random sampling gives one hundred allocations, and its reported conductivity is their mean. Quotas are applied separately within each size class. Heat-throughput ranking sorts the all-low thermal heat-throughput feature. It does not run a shortest-path algorithm.

## 14. Swap search starts from the better reference allocation

The starting point is a complete fifty-particle selection. We compare the degree and heat-throughput selections and initialize every run with the better one; a tie favours degree. Each proposal samples a size class and then one selected and one unselected particle within it. Swapping their labels preserves the quota exactly. The code accepts an increase larger than a relative tolerance of 1e-12. All five runs have the same initial assignment but independent random proposal sequences. Their final selections supply the learning targets, and the best of their final conductivities supplies the search reference. This is not a proven global optimum.

## 15. Teacher allocations for model training

This diagram now shows only model construction. For every training packing, we compute degree and the all-low thermal features. Degree ranking and heat-throughput ranking produce two valid allocations, and the better one initializes five independent swap-search runs. A particle's target is the fraction of the five final selections that contain it. Random, degree and heat-throughput remain separate benchmarks, but they are not required when the trained GNN is deployed.

## 16. Degree and heat throughput have three distinct roles

Degree is a neighbour count and heat throughput is a sum of absolute contact heat rates from the all-low thermal solve. Those are numerical features, available on both training and new packings. Degree ranking and heat-throughput ranking are allocation methods that sort those features within each size class and enforce the conductive-particle quotas. We use the resulting allocations to initialize the training-data search and as independent evaluation references. Using these features in a GNN does not force the GNN to reproduce either ranking: it can combine them with other particle information and contact messages. The MLP checks how much learning from the particle features achieves without those messages and edge features. For practical GNN prediction the features remain necessary, while running the reference allocation methods is optional benchmarking.

## 17. Graph learning and prediction

This section explains the machine-learning task. The GNN performs node-level soft classification. It gives each particle a priority score, after which ranking within each size class enforces the exact material budget.

## 18. The GNN uses geometry and the all-low thermal solution

The baseline quantities are normalized before entering the network. Anchored means connected through the thermal graph to an imposed-temperature boundary. Contact degree counts neighbouring particles, while the separate wall flags describe boundary contacts. The GNN passes information over the actual contact graph for three layers and uses the six edge features. The MLP sees the same eight node features but has no graph message passing or edge features. Consequently the comparison measures the benefit of this graph-based model and its contact information together. Dataset-wide feature standardization uses only the training packings.

## 19. The GNN performs particle classification

The model performs node-level classification, not regression of effective conductivity. Each particle begins with eight node features, and every contact carries six edge features. Three message-passing layers combine the particle's own information with its neighbourhood and create a latent vector h for that particle. A small classification head converts h into a logit and sigmoid score. A high score means that the particle resembles particles repeatedly selected by the teacher search. The latent representation itself has no prescribed physical meaning.

## 20. Soft classification targets come from the search

The target is soft because five searches may finish with different but similarly good allocations. If four of the five final allocations select particle i, its target is 0.8. Binary cross entropy trains the sigmoid output to agree with this selection frequency. The ranking term additionally preserves the relative order within a size class. The model therefore learns particle-selection priorities; it does not predict k effective.

## 21. Ranking within size classes enforces the budget

The GNN scores all five hundred particles, but we do not compare every score in one global list. We first separate the particles by diameter. Within each of the three size classes, each ensemble member supplies a ranking. We average those ranks and choose the top ten, thirty and ten particles. This avoids gaining an advantage merely by selecting more large particles and preserves the exact conductive-material volume.

## 22. Applying the trained GNN to a new packing

For an unseen packing, we first solve the all-low-conductivity thermal network and construct the same node and edge features used during training. The trained GNN then scores every particle. Ranking is performed separately within each size class and selects ten, thirty and ten particles. One final thermal solve evaluates that proposed allocation. The swap search and the simple reference allocations are unnecessary for practical prediction; we run them on held-out cases only to evaluate the GNN.

## 23. Each test packing is withheld from model fitting

The independent unit is the packing, not an individual particle. Splitting particles from the same bed between training and testing would expose almost the same structure on both sides. Here we hold out a whole packing, use another for validation and fit on the remaining twenty-eight. We repeat the process thirty times. These folds share training data, so they are not independent training experiments. This does not establish transfer to a new porosity, particle-size distribution, system size or material contrast.

## 24. Results and conclusions

This section compares all methods on complete held-out packings, reports conductivity gain and recovered search improvement, and closes with numerical limitations and future tests.

## 25. Gain and recovery answer different questions

Gain normalizes the change by the random-allocation conductivity. Recovery instead divides by the available improvement found by the search. For example, a model that improves conductivity by thirty percent when search improves it by about fifty-two percent recovers roughly fifty-eight percent of the gain. The reported values are averages of per-packing ratios, so dividing displayed ensemble means will not necessarily reproduce them exactly. R-squared is not the performance measure for this discrete allocation task.

## 26. GNN improves conductivity by 30% over random

The degree rule adds almost no average benefit. A homogeneous heat-throughput ranking is better, but both learned methods improve on it. The GNN delivers thirty percent mean gain relative to random assignment. Search still achieves substantially more, at fifty-one point eight percent. Each point is a packing, and the black summary marks show the mean and sample standard deviation. These results support learning useful allocation structure while also showing room for improvement.

## 27. Absolute conductivity remains below the search reference

This slide gives the absolute scale of the conductivity changes. The mean rises from approximately 0.228 watts per metre kelvin for random assignment to 0.297 for the GNN. Search reaches approximately 0.347. The thin connecting lines preserve pairing across methods. These absolute numbers belong to the corrected contact-only model and should not be mixed with values from the earlier manuscript, which used a different thermal prefactor and packing-generation implementation.

## 28. GNN beats MLP on 29 of 30 held-out packings

The GNN outperforms the MLP on twenty-nine of the thirty withheld packings. Its mean recovery is fifty-eight percent, with a standard deviation of nine point eight percentage points. This is descriptive evidence across our realizations, not a claim of independent folds or universal superiority. The gap to the search reference is still substantial. Both successful and less successful packings are included in this plot.

## 29. Contact-network diagnostics help interpret allocation

The first diagnostic measures what fraction of summed absolute particle-contact heat rates occurs on contacts whose endpoints are both high-conductivity particles. This is not the fraction of net wall heat because heat can traverse multiple contacts. The second diagnostic asks whether the high-conductivity subgraph alone spans the two thermal boundaries. Low-conductivity particles can still participate in useful paths, so a spanning high-only cluster is not required for heat transfer. The random visualization uses the sampled assignment closest to the random mean.

## 30. Numerical checks pass, but contact sizes need scrutiny

The conservation and mechanical acceptance checks pass. However, numerical convergence does not establish physical validity. The contact-radius ratio has a median of about 0.19 and reaches about 0.35. Those contacts are not uniformly small relative to the particles. The half-space constriction approximation and the softened mechanical generation therefore deserve further assessment. This pooled histogram is a geometry diagnostic; individual contacts from the same bed are correlated.

## 31. The numerical overlap floor has little influence

A small overlap floor regularizes contact-radius evaluation. We repeat the thermal evaluation with different floor values while keeping the material assignments fixed. The largest absolute relative change is less than one thousandth of a percent. That is reassuring about this numerical choice. It does not resolve the physical contact-size concern on the previous slide, because we did not regenerate the beds under another modulus, pressure or contact model.

## 32. Fast scoring is one part of the complete prediction procedure

Fast scoring means producing particle-priority scores with the already trained model. The quoted ensemble scoring time is the sum of forward-evaluation timings for the three model repeats. It excludes rank aggregation and exact quota selection as well as baseline solving, feature construction and final verification. Search uses nearly fifteen thousand candidate thermal solves per packing, which motivates replacing repeated search with learned priorities. However, this timing comparison is not a measurement of complete design time. Teacher-data generation and model training must also be considered when assessing amortized cost.

## 33. Conclusions and next research steps

The main contribution is a constrained allocation workflow that combines a physical contact network with graph learning. Thirty independent packings give a clearer assessment than the earlier ten-case study. The results are promising, but the physical modelling assumptions and the remaining search-performance gap should shape the next work. A practical next algorithmic test is to initialize a short swap search with the GNN selection and compare its gain per thermal solve against random initialization. Physical sensitivity and external transfer should be evaluated before making broader claims.
