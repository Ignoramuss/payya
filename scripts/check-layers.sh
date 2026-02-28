#!/usr/bin/env bash
# Layer-dependency linter for the Payya monorepo.
#
# Enforces the architectural rule: a crate may only depend on crates in the
# same layer or lower.
#
# Layer assignments (from ARCHITECTURE.md):
#   Layer 0: crates/core/
#   Layer 1: crates/models/  crates/speech/
#   Layer 2: crates/training/  crates/inference/  crates/data/
#   Layer 3: crates/agent/  crates/safety/
#   Layer 4: crates/infra/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Map directory prefixes to layer numbers
get_layer() {
    local crate_name="$1"
    local cargo_toml

    # Find the crate's Cargo.toml by matching its name in the workspace
    cargo_toml=$(find "$REPO_ROOT/crates" -name "Cargo.toml" -path "*/$crate_name/Cargo.toml" 2>/dev/null | head -1)

    if [ -z "$cargo_toml" ]; then
        # Not an internal crate (external dependency) — skip
        echo "-1"
        return
    fi

    local rel_path="${cargo_toml#"$REPO_ROOT"/}"

    case "$rel_path" in
        crates/core/*)     echo "0" ;;
        crates/models/*)   echo "1" ;;
        crates/speech/*)   echo "1" ;;
        crates/training/*) echo "2" ;;
        crates/inference/*) echo "2" ;;
        crates/data/*)     echo "2" ;;
        crates/agent/*)    echo "3" ;;
        crates/safety/*)   echo "3" ;;
        crates/infra/*)    echo "4" ;;
        *)                 echo "-1" ;;
    esac
}

layer_name() {
    case "$1" in
        0) echo "Layer 0 (core)" ;;
        1) echo "Layer 1 (models/speech)" ;;
        2) echo "Layer 2 (training/inference/data)" ;;
        3) echo "Layer 3 (agent/safety)" ;;
        4) echo "Layer 4 (infra)" ;;
        *) echo "unknown" ;;
    esac
}

errors=0

# Iterate over every workspace crate
for cargo_toml in "$REPO_ROOT"/crates/*/payya-*/Cargo.toml; do
    crate_dir="$(dirname "$cargo_toml")"
    crate_name="$(basename "$crate_dir")"
    crate_layer=$(get_layer "$crate_name")

    if [ "$crate_layer" = "-1" ]; then
        continue
    fi

    # Extract internal dependencies (lines that reference workspace crates via path)
    # Match lines like: payya-foo = { path = "..." } or payya-foo.workspace = true
    deps=$(grep -E '^payya-' "$cargo_toml" 2>/dev/null | sed 's/\s*=.*//' || true)

    for dep in $deps; do
        dep_layer=$(get_layer "$dep")

        if [ "$dep_layer" = "-1" ]; then
            continue
        fi

        if [ "$dep_layer" -gt "$crate_layer" ]; then
            echo "ERROR: $crate_name ($(layer_name "$crate_layer")) depends on $dep ($(layer_name "$dep_layer"))"
            echo "  A crate may only depend on crates in the same layer or lower."
            errors=$((errors + 1))
        fi
    done
done

if [ "$errors" -gt 0 ]; then
    echo ""
    echo "Found $errors layer violation(s). See ARCHITECTURE.md for the dependency rules."
    exit 1
else
    echo "All layer dependency rules are satisfied."
    exit 0
fi
