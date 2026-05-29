# Process Graph Design Principles

## Purpose

Define the graph abstraction for process feature planning without baking in concentrator or pyromet concepts. This document guides the future `processfeaturecompiler.graph` module. It is not part of the `rtdfeatures` API design.

## Process Graph, Not Flowsheet-Only Graph

A process graph is a directed graph whose nodes represent physical or logical process entities and whose edges represent flows of material, energy, or information.

This is broader than a flowsheet diagram. Flowsheets typically show unit operations and streams. A process graph also includes inventories, storage, measurements, control loops, events, and state variables.

The graph answers: *Where can influence travel in the process?*

## Generic Node Types

```
unit_operation
stream
inventory
storage
measurement
control_loop
event
state
source
sink
```

These are abstract. A crusher, flotation cell, kiln, and leach tank are all `unit_operation` nodes. A stockpile, bin, tank, and silo are all `inventory` or `storage` nodes.

## Generic Edge Types

```
material_flow
energy_flow
information_flow
control_signal
measurement_signal
recycle
bypass
storage_transfer
```

Edges may carry kernels (`EdgeKernelSpec`) that describe the temporal response function for influence propagation along that edge.

## Recycle Handling

Recycle support is mandatory. The first implementation should not pretend to solve full material accounting.

Recycle requirements:

```
explicit recycle edges
cycle detection
bounded unroll
max path length
max total lag
warning when path expansion is truncated
optional effective recycle kernel supplied by user
```

Do not assume the process graph is a DAG.

## Path Expansion Policy

Path expansion discovers upstream influence paths from a target node to candidate source nodes. Policies govern:

- max path length (number of edges)
- max total lag (summed kernel lags)
- edge filtering (include/exclude by edge type)
- node filtering (include/exclude by node type)
- cycle unroll depth
- warning when truncation occurs

## Effective Recycle Kernels

When cycles exist, the compiler may support an optional effective recycle kernel supplied by the user. This approximates the net effect of recycle without requiring full iterative convergence.

## Path Kernel Composition via `rtdfeatures`

Each discovered path becomes a composed kernel via `rtdfeatures.compose_kernels`. Edge kernels are convolved along the path to produce a path kernel. The path kernel is then used in a standard `KernelFeaturePlan` for feature generation.

`rtdfeatures` provides the kernel algebra. `processfeaturecompiler.graph` provides the path discovery.

## Feature Budgets

Feature plans should respect budgets:

- max features per target
- max features per node
- max total lag per feature
- max upstream depth per target

Budgets prevent unbounded feature explosion from large graphs.

## Graph Provenance

Every generated feature should carry optional provenance metadata:

- `source_node` — upstream node
- `target_node` — downstream node where feature is evaluated
- `graph_edge` — edge identifier
- `graph_path` — ordered list of edges in the path
- `edge_kernel_name` — kernel attached to the edge
- `path_kernel_name` — composed kernel for the path

Provenance is optional and must not add hard graph dependencies to `rtdfeatures`.

## Explicit Warning

```
Do not assume DAG.
Do not promise material balance.
Do not hard-code pyromet/flotation/crushing entities.
```

## Domain Examples as Examples Only

Pyrometallurgy stresses the abstraction: energy memory, longer residence times, large inventories, and batch/semi-continuous events. But pyromet, flotation, crushing, and leaching are all applications of the same generic process graph abstraction. Examples may reference them for concreteness. The API and schema must not hard-code them.
