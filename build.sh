#!/bin/sh
# Build the Ionic compiler from the split source tree.
# Usage:
#   ./build.sh              — build ionic_new using ionic_self (committed bootstrap binary)
#   IONIC=./some_binary ./build.sh  — use a specific Ionic binary

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

echo "==> Compiling $OUT from split source..."
$IONIC $SOURCES -o "$OUT"
echo "==> Done: $OUT"
