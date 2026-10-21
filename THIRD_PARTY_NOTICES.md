# Third-party notices

GraCo is under the MIT license (`LICENSE`), except the code below that
derives from code released under the Apache License 2.0 and stays under it.
A copy of that license is in `licenses/Apache-2.0.txt`.

## algorithms/convmatch.py

Derived from amazon-science/convolution-matching (Dickens et al., 2024).

    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

Changes: the costs are written on torch sparse CSR, where the original uses
DGL message passing; the exact cost is computed for all candidate pairs at
once, where the original scores them one by one; the options
pair="binary", pair="normalized" and merge_candidates="knn+edges" are added.

## algorithms/fgc.py

Derived from GraphCoarsening/Featured-Graph-Coarsening (Kumar et al., 2023).

Changes: the metrics solver moves from numpy to torch, both solvers sit
behind one variant argument, the hard partition is returned by default,
and the Delta-Laplacian family is added.

## algorithms/local_variation/

Implements local variation (Loukas, JMLR 2019). It started from
loukasa/graph-coarsening.

    Copyright 2019 Andreas Loukas

It keeps the structure of that code, several function names and a few short
passages (in cost.py, main.py and matching.py), which stay under the Apache
License 2.0. New in GraCo: the Laplacian family (combinatorial, normalized,
self-loop normalized and the Delta-Laplacian); a closed-form cost for pairs,
computed for every edge in one vectorized pass; a numba-compiled cost for
neighbourhoods; a per-level budget that follows the current graph size; an
absorption pass when a level stalls; and the Partition output shared by
every GraCo method.

## Other methods

Every other method is implemented from its paper, and from its released
code where the paper leaves a point open or the code departs from it. Each
module names both in its docstring, with the departures.

## Optional GPL dependencies

The `leiden` extra installs igraph (GPL-2.0) and leidenalg (GPL-3.0).
GraCo imports them only inside `coarsen_leiden`.
