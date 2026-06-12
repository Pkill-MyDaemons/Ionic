# Ionic

A statically-typed, self-hosting compiled language targeting native ARM64 (macOS). Ionic compiles directly to Mach-O object files — no LLVM required at runtime — and enforces hardware placement at the type level: `tensor@cpu` and `tensor@gpu` are distinct types and the compiler rejects code that crosses the boundary without an explicit transfer.

```ionic
fn fibonacci(int64 n) -> int64 {
    if (n <= 1) { return n; }
    mut a = 0; mut b = 1; mut i = 2;
    while (i <= n) { let c = a + b; a = b; b = c; i = i + 1; }
    return b;
}

mut i = 0;
while (i <= 10) {
    println(format("fib({}) = {}", int64_to_str(i), int64_to_str(fibonacci(i))));
    i = i + 1;
}
```

```
$ ./ionic_new fib.ionic -o fib && ./fib
fib(0) = 0
fib(1) = 1
fib(2) = 1
fib(3) = 2
...
fib(10) = 55
```

---

## Features

- **Self-hosting** — `ionic_new` is compiled by itself; the entire compiler is written in Ionic
- **Direct ARM64 code generation** — emits Mach-O `.o` files directly, linked with `ld`; no LLVM at runtime
- **Static types, inferred where obvious** — `let x = 42` is `int64`; `let f = 3.14` is `float64`
- **Full float64 support** — arithmetic, `sqrt`, `pow`, `floor`, `ceil`, `fabs`, `int64_to_float64`, `float64_to_str`
- **Rich string builtins** — `format`, `str_concat`, `str_len`, `str_slice`, `str_replace`, `str_contains`, `str_starts_with`, `str_ends_with`, `int64_to_str`
- **Arrays** — `[int64]` type, `.push`, `.len`, `arr_reset`, indexing and element assignment
- **Hardware-aware types** — `tensor@cpu` and `tensor@gpu` prevent accidental cross-device ops
- **Real ML backends** — GGUF models via llama.cpp with Metal GPU; ONNX/CoreML; Piper TTS
- **Human-readable errors** — multi-error reporting, source-line carets, column tracking, panic-mode recovery

---

## Build

### Prerequisites

- Rust toolchain + `cargo` (bootstrap only — not needed once `ionic_self` exists)
- Clang / `ld` (for linking)

### Quick start

```sh
git clone <repo>
cd AILANG
cargo build --release          # builds the bootstrap Rust compiler
./build.sh --bootstrap         # compiles ionic_self (Ionic→ARM64) and ionic_new
```

After that, `ionic_self` and `ionic_new` are both native ARM64 binaries. `ionic_new` is the primary compiler.

### Rebuild after source changes

```sh
./build.sh          # recompile ionic_new from split sources using ionic_self (~3s)
```

---

## Usage

```
./ionic_new <file.ionic> -o <output>
```

**Compile and run:**
```sh
./ionic_new hello.ionic -o hello && ./hello
```

**No `fn main` needed** — top-level statements run directly:
```ionic
let x = 6 * 7;
println(int64_to_str(x));   // prints 42
```

---

## Language quick reference

### Variables

```ionic
let x = 42;           // immutable int64
mut count = 0;        // mutable int64
let pi = 3.14159;     // float64
let msg = "hello";    // string
```

### Functions

```ionic
fn gcd(int64 a, int64 b) -> int64 {
    mut aa = a; mut bb = b;
    while (bb != 0) { let t = bb; bb = aa - (aa / bb) * bb; aa = t; }
    return aa;
}
```

### Control flow

```ionic
if (x > 0) { println("positive"); }
else { println("non-positive"); }

while (i < 10) { i = i + 1; }
```

### Arrays

```ionic
mut data = [0]; arr_reset(data);
data.push(10); data.push(20); data.push(30);
println(int64_to_str(data.len));     // 3
println(int64_to_str(data[1]));      // 20
data[1] = 99;
```

### String formatting

```ionic
let s = format("x={}, y={}", int64_to_str(x), float64_to_str(y));
println(s);
```

### Float math

```ionic
let pi = 3.14159265358979;
println(float64_to_str(sqrt(2.0)));          // 1.41421
println(float64_to_str(pow(pi, 2.0)));        // 9.8696
println(float64_to_str(int64_to_float64(n))); // cast int→float
```

---

## Project layout

```
src/
  codegen/         Code generator (ARM64 Mach-O emitter) — Ionic source
  parser/          Parser — Ionic source
  lexer/           Lexer — Ionic source
  semantic/        Type checker — Ionic source
  main.ionic       Compiler entry point
  diagnostics.ionic  Error reporting
  compiler.ionic   Monolithic source (bootstrap only)
  codegen.ionic    Monolithic codegen (bootstrap only)
  codegen.rs       Rust LLVM backend (bootstrap only)
  lexer.rs / parser.rs / semantic.rs   Rust frontend (bootstrap only)
  ionic_model_runtime.c   Native runtime: I/O, arrays, strings, math, ML
build.sh           Build script (--bootstrap for full rebuild)
ionic_new          Primary compiler binary (self-hosted)
ionic_self         Previous-generation compiler (used to build ionic_new)
```

---

## Self-hosting cycle

```
Rust compiler ──bootstrap──> ionic_self
ionic_self    ──build──────> ionic_new
ionic_new     ──build──────> ionic_new  (stable fixed point)
```

The Rust source (`src/*.rs`) is only needed for the initial bootstrap. Once `ionic_self` exists it is not required again unless you change the bootstrap compiler.
