#!/bin/bash
# Transform [name]<ref:key> to Rust intra-doc links [name](crate::guide::references#key)

cd "$(git rev-parse --show-toplevel)" || exit 1

# Transform <ref:key> to intra-doc links to references
find src -name "*.rs" -exec grep -l '<ref:' {} \; | while read -r file; do
    perl -i -pe 's/\[([^\]]+)\]<ref:([a-zA-Z0-9_]+)>/[$1](crate::guide::references#$2)/g' "$file"
    echo "Transformed <ref:>: $file"
done
