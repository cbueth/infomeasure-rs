# Kernel Estimator Dimensions

This document explains the relationship between the various const generic parameters used in the Kernel-based estimators (MI, CMI, TE, CTE).

To avoid relying on the unstable `generic_const_exprs` feature in Rust, we explicitly pass all required dimensions as const generics. These dimensions are mathematically dependent on each other.

## Transfer Entropy (TE)

$TE(X \to Y) = I(Y_{future}; X_{past} | Y_{past})$

The estimator `KernelTransferEntropy` uses:
- `SRC_HIST`, `DEST_HIST`: History lengths.
- `STEP_SIZE`: Delay between observations.
- `D_SOURCE`, `D_TARGET`: Dimensionality of individual samples in X and Y.
- `D_JOINT`: $D_{target} + (SRC\_HIST \times D_{source}) + (DEST\_HIST \times D_{target})$
- `D_XP_YP`: $(SRC\_HIST \times D_{source}) + (DEST\_HIST \times D_{target})$
- `D_YP`: $DEST\_HIST \times D_{target}$
- `D_YF_YP`: $D_{target} + (DEST\_HIST \times D_{target})$

### Dimension Relations for TE
- `D_JOINT` = `D_YF_YP` + `SRC_HIST` * `D_SOURCE`
- `D_XP_YP` = `D_YP` + `SRC_HIST` * `D_SOURCE`
- `D_YF_YP` = `D_TARGET` + `D_YP`
- `D_YP` = `DEST_HIST` * `D_TARGET`

## Conditional Transfer Entropy (CTE)

$CTE(X \to Y | Z) = I(Y_{future}; X_{past} | Y_{past}, Z_{past})$

The estimator `KernelConditionalTransferEntropy` uses:
- `D_JOINT`: $D_{target} + (SRC\_HIST \times D_{source}) + (DEST\_HIST \times D_{target}) + (COND\_HIST \times D_{cond})$
- `D_XP_YP_ZP`: $(SRC\_HIST \times D_{source}) + (DEST\_HIST \times D_{target}) + (COND\_HIST \times D_{cond})$
- `D_YP_ZP`: $(DEST\_HIST \times D_{target}) + (COND\_HIST \times D_{cond})$
- `D_YF_YP_ZP`: $D_{target} + (DEST\_HIST \times D_{target}) + (COND\_HIST \times D_{cond})$

### Dimension Relations for CTE
- `D_JOINT` = `D_YF_YP_ZP` + `SRC_HIST` * `D_SOURCE`
- `D_XP_YP_ZP` = `D_YP_ZP` + `SRC_HIST` * `D_SOURCE`
- `D_YF_YP_ZP` = `D_TARGET` + `D_YP_ZP`
- `D_YP_ZP` = (`DEST_HIST` * `D_TARGET`) + (`COND_HIST` * `D_COND`)

## Mutual Information (MI)

$I(X_1; X_2; \dots; X_n) = \log p(X_1, \dots, X_n) - \sum \log p(X_i)$

The estimators `KernelMutualInformation2` through `KernelMutualInformation6` support 2 to 6 random variables respectively.

- `D_JOINT`: $\sum_{i=1}^n D_i$
- `D1`, `D2`, ..., `Dn`: Dimensions of individual random variables.

## Conditional Mutual Information (CMI)

$I(X; Y | Z) = \log \frac{p(X,Y,Z)p(Z)}{p(X,Z)p(Y,Z)}$

- `D_JOINT`: $D_1 + D_2 + D_{cond}$
- `D1_COND`: $D_1 + D_{cond}$
- `D2_COND`: $D_2 + D_{cond}$

## Helper Macros

To simplify instantiation and automatically calculate these dimensions, use the following macros:
- `new_kernel_te!`
- `new_kernel_cte!`
- `new_kernel_mi!`
- `new_kernel_cmi!`
