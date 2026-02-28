#!/usr/bin/env bash
# One-time developer setup for the Payya monorepo.
# Configures git hooks and verifies toolchain.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Setting up Payya development environment..."

# Configure git to use the repo's hooks directory
git -C "$REPO_ROOT" config core.hooksPath .githooks
echo "  [ok] Git hooks configured (.githooks/)"

# Verify required tools
for tool in cargo rustfmt; do
    if command -v "$tool" &>/dev/null; then
        echo "  [ok] $tool found"
    else
        echo "  [!!] $tool not found — please install the Rust toolchain"
        exit 1
    fi
done

# Check for clippy
if cargo clippy --version &>/dev/null; then
    echo "  [ok] clippy found"
else
    echo "  [!!] clippy not found — run: rustup component add clippy"
fi

echo ""
echo "Setup complete. Pre-commit hooks will run 'cargo fmt --check' on staged Rust files."
