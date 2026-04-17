#!/bin/bash
# Transform [name]<ref:key> to [name](path#key) for docs.rs

cd "$(git rev-parse --show-toplevel)" || exit 1

find src -name "*.rs" -exec grep -l '<ref:' {} \; | while read -r file; do
    depth=$(echo "$file" | tr -cd '/' | wc -c)
    up=$(printf '../%.0s' $(seq 1 $depth))
    path="${up}guide/references/index.html"
    
    # Escape slashes for perl
    escaped_path=${path//\//\\\/}
    /usr/bin/perl -i -pe "s/\[([^\]]+)\]<ref:([a-zA-Z0-9_]+)>/\"[\$1](${escaped_path}#\$2)\"/g" "$file"
    echo "Transformed: $file"
done
