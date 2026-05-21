# Ionic

A statically-typed, compiled language built for data science and AI workloads. Ionic compiles to native binaries via LLVM and enforces hardware placement at the type level — `tensor@cpu` and `tensor@gpu` are different types, and the compiler rejects code that crosses the boundary without an explicit transfer.

```ionic
import std.math.*;
import std.array.*;

fn main() -> int64 {
    let nums = [1, 2, 3, 4, 5];
    let total = arr_sum(nums);
    print(str_concat("Sum: ", int64_to_str(total)));

    let mid = lerp(0.0, 100.0, 0.5);
    print(str_concat("Midpoint: ", float64_to_str(mid)));

    return 0;
}
```

```
$ ionic hello.ionic -o hello
$ ./hello
Sum: 15
Midpoint: 50.000000
```

---

## Features

- **Static types, inferred where obvious** — `let x = 42` is `int64`; `mut y = 3.14` is `float64`
- **Hardware-aware types** — `tensor@cpu` and `tensor@gpu` prevent accidental cross-device ops
- **Real ML backends** — loads ONNX models (with CoreML on Apple Silicon) and GGUF models via llama.cpp with Metal GPU
- **Selective compilation** — wildcard imports (`std.math.*`) only compile the functions you actually call; zero dead-code overhead
- **Native speed** — LLVM backend, no GC, no runtime

---

## Install

### Prerequisites

- Rust toolchain (`cargo`)
- LLVM / Clang (for linking)
- *(Optional)* ONNX Runtime — for `.onnx` model inference
- *(Optional)* llama.cpp — for `.gguf` model inference

On macOS with Homebrew:

```sh
brew install llvm onnxruntime llama.cpp
```

### Build

```sh
git clone <repo>
cd AILANG
cargo build --release
```

The compiler binary lands at `target/release/ionic`. Add it to your PATH:

```sh
export PATH="$PWD/target/release:$PATH"
```

---

## Usage

```
ionic <file.ionic> [flags] [-o <output>]

  -o <name>       Output binary name (default: stem of source file)
  --emit-ir       Print LLVM IR to stdout, skip linking
  --dump-ast      Print parsed AST to stderr
  --dump-tokens   Print token stream to stderr
  --run           Compile and immediately execute
```

**Compile and run:**
```sh
ionic hello.ionic --run
```

**Compile to binary:**
```sh
ionic hello.ionic -o hello && ./hello
```

**Inspect generated IR:**
```sh
ionic hello.ionic --emit-ir
```

---

## Quick start

Save this as `hello.ionic`:

```ionic
fn main() -> int64 {
    print("Hello, Ionic!");
    return 0;
}
```

```sh
ionic hello.ionic --run
```

Programs with no `fn main` also work — top-level statements execute directly:

```ionic
let x = 6 * 7;
print(int64_to_str(x));
```

---

## Standard library

Import any module with `.*` for zero-overhead selective compilation:

```ionic
import std.math.*;
import std.str.*;
import std.array.*;
import std.io.*;
```

Only the symbols you actually reference are compiled into the output. See [docs/stdlib.md](docs/stdlib.md) for the full API.

---

## Examples

| File | What it shows |
|------|---------------|
| `examples/hello.ionic` | Hello world, arithmetic |
| `examples/showcase.ionic` | Types, loops, recursion, primes |
| `examples/stdlib_test.ionic` | All four stdlib modules together |

---

## Project layout

```
src/             Compiler source (Rust)
  lexer.rs       Tokeniser
  parser.rs      AST builder
  semantic.rs    Type checker
  codegen.rs     LLVM IR emitter
  imports.rs     Selective stdlib compilation (BFS reachability)
  ionic_model_runtime.c   Native ML runtime (ONNX, llama.cpp, WAV I/O)
lib/std/         Standard library (written in Ionic)
  math.ionic
  str.ionic
  array.ionic
  io.ionic
examples/        Sample programs
docs/            Language and stdlib reference
```

---

## Documentation

- [Language reference](docs/language.md) — types, syntax, control flow, structs, arrays, hardware placement, all built-ins
- [Standard library](docs/stdlib.md) — `std.math`, `std.str`, `std.array`, `std.io` APIs
- [Guide](docs/guide.md) — practical walkthrough: variables, functions, loops, arrays, ML models
