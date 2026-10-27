# Federated Semi-Supervised k-DPP

This research project explores how to select one shared, diverse subset from a
public collection using private context sets held by multiple clients. The
selected items should complement what the clients already hold, without
revealing membership in their private sets.

A determinantal point process (DPP) favors subsets with less redundancy. A
k-DPP selects a fixed number of items, while the semi-supervised setting uses
existing context items to guide the selection of additional items.

The approach is still being developed. Initial experiments use DNA k-mers to
compare direct centralized sampling with a distributed baseline: clients sample
local candidates, then a server samples the final subset from their merged
selections. This baseline is context-free and non-private; it is an initial
comparison, not a solution to the private semi-supervised problem. Local
selection and aggregation are possible approaches, not requirements for the
final method.

See the [problem statement](writing/problem.tex) for the evolving formulation.
