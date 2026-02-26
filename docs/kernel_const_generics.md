# Kernel Estimators Const Generics Documentation

This document explains the const generic parameters used in the Kernel-based Information Theory estimators (Mutual Information, Transfer Entropy, and their conditional versions).

## Overview

Due to Rust's current limitations with constant expressions in generic arguments (without `generic_const_exprs`), all dimensionalities—including derived ones—must be passed as explicit const generics. This ensures that the dimensionality of every internal `KernelEntropy<D>` estimator is known at compile-time, allowing for significant optimizations and type safety.

## 1. Mutual Information (MI)

**Struct**: `KernelMutualInformation<const N: usize, const D_ALL: usize, const D_MARGINALS: [usize; N]>`

- `N`: The number of random variables (series) being analyzed.
- `D_ALL`: The total dimensionality of the joint space (sum of dimensions of all `N` variables).
- `D_MARGINALS`: An array of size `N` containing the dimensionality of each individual random variable.

**Mathematical Relationship**: `D_ALL = sum(D_MARGINALS)`

**Implementation Note**: Specialized implementations for different `D_MARGINALS` combinations are provided via the `impl_marginal_densities!` macro.

## 2. Conditional Mutual Information (CMI)

**Struct**: `KernelConditionalMutualInformation<const N: usize, const D_JOINT: usize, const D_COND: usize, const D_MARGINALS_COND: [usize; N]>`

- `N`: The number of input random variables (excluding the condition).
- `D_JOINT`: The total dimensionality of the joint space, including the condition ($D_{X_1 \dots X_N Z}$).
- `D_COND`: The dimensionality of the conditioning variable ($Z$).
- `D_MARGINALS_COND`: An array of size `N` where each element $i$ is the dimensionality of the joint space of the $i$-th variable and the condition ($D_{X_i Z}$).

**Mathematical Relationships**:
- $D_{X_i Z} = D_{X_i} + D_Z$
- $D_{JOINT} = \sum D_{X_i} + D_Z$

## 3. Transfer Entropy (TE)

**Struct**: `KernelTransferEntropy<const SRC_HIST: usize, const DEST_HIST: usize, const STEP_SIZE: usize, const D_SOURCE: usize, const D_TARGET: usize, const D_JOINT: usize, const D_XP_YP: usize, const D_YP: usize, const D_YF_YP: usize>`

- `SRC_HIST`, `DEST_HIST`, `STEP_SIZE`: Embedding parameters (history lengths and lag).
- `D_SOURCE`, `D_TARGET`: Dimensionality of the raw source and destination variables.
- `D_JOINT`: Dim of $(Y_{future}, X_{past}, Y_{past})$.
- `D_XP_YP`: Dim of $(X_{past}, Y_{past})$.
- `D_YP`: Dim of $(Y_{past})$.
- `D_YF_YP`: Dim of $(Y_{future}, Y_{past})$.

**Mathematical Relationships**:
- $D_{X_{past}} = SRC\_HIST \times D_{SOURCE}$
- $D_{Y_{past}} = DEST\_HIST \times D_{TARGET}$
- $D_{Y_{future}} = D_{TARGET}$
- $D_{JOINT} = D_{TARGET} + (SRC\_HIST \times D_{SOURCE}) + (DEST\_HIST \times D_{TARGET})$
- $D_{XP\_YP} = (SRC\_HIST \times D_{SOURCE}) + (DEST\_HIST \times D_{TARGET})$
- $D_{YP} = DEST\_HIST \times D_{TARGET}$
- $D_{YF\_YP} = D_{TARGET} + (DEST\_HIST \times D_{TARGET})$

## 4. Conditional Transfer Entropy (CTE)

**Struct**: `KernelConditionalTransferEntropy<...>`

Extends TE by adding `D_COND` and `COND_HIST`. Derived dimensions must include the condition history dimension $D_{Z_{past}} = COND\_HIST \times D_{COND}$.

- $D_{JOINT} = D_{Y_{future}} + D_{X_{past}} + D_{Y_{past}} + D_{Z_{past}}$
- $D_{XP\_YP\_ZP} = D_{X_{past}} + D_{Y_{past}} + D_{Z_{past}}$
- $D_{YP\_ZP} = D_{Y_{past}} + D_{Z_{past}}$
- $D_{YF\_YP\_ZP} = D_{Y_{future}} + D_{Y_{past}} + D_{Z_{past}}$
