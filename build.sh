#!/bin/sh
# Build the Ionic compiler from the split source tree.
# Usage:
#   ./build.sh              — build ionic_new using existing ionic_self
#   ./build.sh --bootstrap  — rebuild ionic_self first via Cargo, then build ionic_new
#   IONIC=./ionic_self_v7 ./build.sh  — use a specific Ionic binary

set -e

IONIC="${IONIC:-./ionic_self}"
OUT="${OUT:-ionic_new}"

SOURCES="
  src/lexer/tokens.ionic
  src/lexer/lexer.ionic
  src/diagnostics.ionic
  src/parser/ast.ionic
  src/parser/parser.ionic
  src/semantic/checker.ionic
  src/codegen/native.ionic
  src/main.ionic
"

if [ "$1" = "--bootstrap" ]; then
    echo "==> Bootstrapping via Cargo..."
    cargo build --release
    cp target/release/ionic ./ionic
    echo "==> Compiling ionic_self from monolithic source..."
    ./ionic src/compiler.ionic src/codegen.ionic -o ionic_self
    echo "==> ionic_self ready"
fi

echo "==> Compiling $OUT from split source..."
$IONIC $SOURCES -o "$OUT"
echo "==> Done: $OUT"
