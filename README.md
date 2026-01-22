# PERSIST

Persistence-Guided Multiscale Domain Identification in Spatial Transcriptomics

---

## Overview

PERSIST is a topology-guided framework for identifying spatial domains in spatial transcriptomics data without relying on fixed clustering resolutions, neighborhood radii, or biological priors.

The method uses persistent homology to infer intrinsic geometric scales supported by the tissue embedding and integrates these scales into a multiscale stability field. Spatial domains are then extracted as stable basins of this field, capturing both domain cores and transition regions.

PERSIST is designed as a general, unsupervised method that adapts to different tissue architectures, including heterogeneous tumors, compartmentalized immune organs, and highly ordered neural tissue.

---

## Key Ideas

- Scale discovery is decoupled from domain inference
- Spatial organization is modeled geometrically rather than via clustering
- Persistent homology is used to identify intrinsic spatial scales
- Domains are supported by multiscale stability, not resolution tuning
- Transition regions are preserved rather than forced into hard partitions

---

## Method Summary

The PERSIST pipeline consists of the following steps:

1. Filter genes for spatial structure using Moran’s I  
2. Construct a scalar field from spatially structured expression (default: PC1)  
3. Build Alpha complex filtrations on spatial coordinates  
4. Compute persistent homology to identify intrinsic spatial scales  
5. Define differential operators using discrete exterior calculus  
6. Aggregate multiscale transcriptional variation into a stability field  
7. Extract spatial domains via basin decomposition  
8. Assess robustness using spatial bootstrapping  

Marker genes and biological annotations are used only for post hoc interpretation.

---




