// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Internal macros for documentation snippets to reduce verbatim repetition across modules.
//!
//! These macros are not part of the public API and are used to ensure consistency
//! in theoretical explanations, technical limitations, and cross-references.

macro_rules! doc_snippets {
    (facade_overview $measure:expr, $description:expr) => {
        concat!(
            $description, "\n\n",
            "This struct provides a unified interface for all ", $measure,
            " estimation techniques supported by the library. ",
            "It includes methods for discrete, kernel-based, ordinal, and ",
            "exponential family (k-NN) estimators.\n\n",
            "Each estimator can be used to compute the global value or local values ",
            "(if supported) using the [`GlobalValue`](crate::estimators::traits::GlobalValue) ",
            "and [`LocalValues`](crate::estimators::traits::LocalValues) traits."
        )
    };
    (const_generic_limitation) => {
        concat!(
            "Due to Rust's current limitations with constant expressions in generic arguments (without\n",
            "`generic_const_exprs`), all dimensionalities—including derived ones—must be passed as explicit\n",
            "const generics. This ensures that the dimensionality of every internal estimator is known\n",
            "at compile-time, allowing for significant optimizations and type safety."
        )
    };
    (macro_simplification_mi) => {
        concat!(
            "To simplify instantiation and automatically calculate these dimensions, use the following macros:\n",
            "- `new_kernel_mi!` - Creates a `KernelMutualInformation` estimator\n",
            "- `new_kernel_cmi!` - Creates a `KernelConditionalMutualInformation` estimator\n",
            "- `new_ksg_mi!` - Creates a `KsgMutualInformation` estimator\n",
            "- `new_ksg_cmi!` - Creates a `KsgConditionalMutualInformation` estimator\n",
            "- `new_renyi_mi!` - Creates a `RenyiMutualInformation` estimator\n",
            "- `new_renyi_cmi!` - Creates a `RenyiConditionalMutualInformation` estimator\n",
            "- `new_tsallis_mi!` - Creates a `TsallisMutualInformation` estimator\n",
            "- `new_tsallis_cmi!` - Creates a `TsallisConditionalMutualInformation` estimator\n",
            "- `new_kl_mi!` - Creates a KL-divergence based MI estimator\n",
            "- `new_jsd_mi!` - Creates a JSD-based MI estimator\n",
            "- `new_ordinal_mi!` - Creates an `OrdinalMutualInformation` estimator\n\n",
            "These macros handle the dimension calculations automatically based on the input\n",
            "dimensionalities you provide."
        )
    };
    (macro_simplification_te) => {
        concat!(
            "To simplify instantiation and automatically calculate these dimensions, use the following macros:\n",
            "- `new_kernel_te!` - Creates a `KernelTransferEntropy` estimator\n",
            "- `new_kernel_cte!` - Creates a `KernelConditionalTransferEntropy` estimator\n",
            "- `new_ksg_te!` - Creates a `KsgTransferEntropy` estimator\n",
            "- `new_ksg_cte!` - Creates a `KsgConditionalTransferEntropy` estimator\n",
            "- `new_renyi_te!` - Creates a `RenyiTransferEntropy` estimator\n",
            "- `new_renyi_cte!` - Creates a `RenyiConditionalTransferEntropy` estimator\n",
            "- `new_tsallis_te!` - Creates a `TsallisTransferEntropy` estimator\n",
            "- `new_tsallis_cte!` - Creates a `TsallisConditionalTransferEntropy` estimator\n",
            "- `new_ordinal_te!` - Creates an `OrdinalTransferEntropy` estimator\n",
            "- `new_ordinal_cte!` - Creates an `OrdinalConditionalTransferEntropy` estimator\n\n",
            "These macros handle the dimension calculations automatically based on the history lengths\n",
            "and input dimensionalities you provide."
        )
    };
    (mi_formula $approach:expr, $subscript:expr, $context:expr) => {
        doc_snippets!(mi_formula $approach, $subscript, $context, "H")
    };
    (mi_formula $approach:expr, $subscript:expr, $context:expr, $symbol:expr) => {
        concat!(
            $approach, " MI is estimated via the entropy-summation formula", $context, ":\n",
            "$$I", $subscript, "(X_1; \\ldots; X_n) = \\sum ", $symbol, $subscript, "(X_i) - ", $symbol, $subscript, "(X_1, \\ldots, X_n)$$\n\n",
            "See the [Mutual Information Guide](crate::guide::mutual_information) for conceptual background."
        )
    };
    (cmi_formula $approach:expr, $subscript:expr, $context:expr) => {
        doc_snippets!(cmi_formula $approach, $subscript, $context, "H")
    };
    (cmi_formula $approach:expr, $subscript:expr, $context:expr, $symbol:expr) => {
        concat!(
            $approach, " CMI is estimated via the entropy-summation formula", $context, ":\n",
            "$$I", $subscript, "(X; Y \\mid Z) = ", $symbol, $subscript, "(X, Z) + ", $symbol, $subscript, "(Y, Z) - ", $symbol, $subscript, "(X, Y, Z) - ", $symbol, $subscript, "(Z)$$\n\n",
            "See the [Conditional MI Guide](crate::guide::cond_mi) for conceptual background."
        )
    };
    (te_formula $approach:expr, $subscript:expr, $context:expr) => {
        doc_snippets!(te_formula $approach, $subscript, $context, "H")
    };
    (te_formula $approach:expr, $subscript:expr, $context:expr, $symbol:expr) => {
        concat!(
            $approach, " transfer entropy is estimated via the CMI entropy-summation formula", $context, ":\n",
            "$$T", $subscript, "(X \\to Y) = ", $symbol, $subscript, "(X_{past}, Y_{past}) + ", $symbol, $subscript, "(Y_{future}, Y_{past}) - ", $symbol, $subscript, "(X_{past}, Y_{future}, Y_{past}) - ", $symbol, $subscript, "(Y_{past})$$\n\n",
            "See the [Transfer Entropy Guide](crate::guide::transfer_entropy) for conceptual background."
        )
    };
    (cte_formula $approach:expr, $subscript:expr, $context:expr) => {
        doc_snippets!(cte_formula $approach, $subscript, $context, "H")
    };
    (cte_formula $approach:expr, $subscript:expr, $context:expr, $symbol:expr) => {
        concat!(
            $approach, " conditional transfer entropy is estimated via the CTE entropy-summation formula", $context, ":\n",
            "$$TE", $subscript, "(X \\to Y \\mid Z) = ", $symbol, $subscript, "(X_{past}, Y_{past}, Z_{past}) + ", $symbol, $subscript, "(Y_{future}, Y_{past}, Z_{past}) - ", $symbol, $subscript, "(X_{past}, Y_{future}, Y_{past}, Z_{past}) - ", $symbol, $subscript, "(Y_{past}, Z_{past})$$\n\n",
            "See the [Conditional TE Guide](crate::guide::cond_te) for conceptual background."
        )
    };
    (discrete_guide_ref) => {
        "See the [Discrete Entropy Guide](crate::guide::entropy::discrete) for conceptual background."
    };
}

pub(crate) use doc_snippets;
