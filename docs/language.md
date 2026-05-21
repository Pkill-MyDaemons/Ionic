# Ionic Language Reference

Ionic is a statically-typed, curly-brace language for data science and AI workloads. It compiles to native binaries via LLVM and supports explicit hardware placement (`@cpu` / `@gpu`).

---

## Types

| Type | Description |
|------|-------------|
| `int64` | 64-bit signed integer |
| `float64` | 64-bit IEEE 754 double |
| `bool` | Boolean (`true` / `false`) |
| `string` | UTF-8 string (immutable value) |
| `tensor@cpu` | CPU-resident tensor |
| `tensor@gpu` | GPU-resident tensor |
| `model` | Opaque ML model handle |
| `[T]` | Dynamic array of element type T |
| `void` | No return value |
| `StructName` | User-defined struct |

---

## Variables

```ionic
let x = 42;               // immutable int64 (type inferred)
mut y = 3.14;             // mutable float64
let name: string = "hi";  // explicit type annotation
```

`let` variables cannot be reassigned. `mut` variables can.

---

## Functions

```ionic
fn add(int64 a, int64 b) -> int64 {
    return a + b;
}

fn greet(string name) -> string {
    return str_concat("Hello, ", name);
}
```

Parameters are typed before their name. Return type follows `->`. Recursion is fully supported.

---

## Control flow

```ionic
if (x > 0) {
    print("positive");
} else {
    print("non-positive");
}

while (x > 0) {
    x = x - 1;
}

for (i in 0..10) {        // exclusive upper bound
    print(int64_to_str(i));
}

break;     // exits the nearest loop
continue;  // skips to the next iteration
```

---

## Structs

```ionic
struct Point { x: float64, y: float64 }

let p = Point { x: 1.0, y: 2.0 };
print(float64_to_str(p.x));
p.y = 3.0;               // field assignment
```

Structs are heap-allocated. All fields are mutable regardless of whether the binding is `let` or `mut`.

---

## Arrays

```ionic
mut nums = [1, 2, 3];     // [int64]
nums.push(4);
let n = nums.len;          // int64
let first = nums[0];       // element access
nums[0] = 99;              // element assignment
```

Arrays grow dynamically. Element type is inferred from the first literal.

---

## Hardware placement

```ionic
@gpu fn train_step(...) -> ... { ... }   // run on GPU
@cpu fn load_data(...) -> ...  { ... }   // run on CPU

@gpu(0.5) fn partial(...) -> ... { ... } // use 50% of GPU

let weights: tensor@gpu = ...;
gpu {
    // Only tensor@gpu variables and compute are allowed here.
    // CPU I/O (print, file_read, file_write) is forbidden.
}
```

`.toGpu()` and `.toCpu()` convert tensors between hardware domains.

---

## Imports

```ionic
import std.math.*;                    // glob — all symbols, only used ones compiled
import std.str.{contains, trim};      // named — exactly these two symbols
import std.io;                        // module — all symbols (same as glob for now)
```

**Selective compilation:** when using wildcards, the compiler walks the AST and determines which library symbols are actually referenced — including transitive dependencies — and only emits IR for those. Unused library functions add zero overhead.

---

## Doc comments

```ionic
/// Returns the absolute value of x.
fn abs(float64 x) -> float64 {
    if (x < 0.0) { return -x; }
    return x;
}
```

`///` lines immediately before a `fn` or `struct` definition are attached to it as documentation. They appear in `--dump-ast` output and in generated docs.

---

## Built-in functions

### Printing
| Function | Description |
|----------|-------------|
| `print(string s)` | Print to stdout with newline |

### String operations
| Function | Description |
|----------|-------------|
| `str_concat(string a, string b) -> string` | Concatenate two strings |
| `str_len(string s) -> int64` | Length in bytes |
| `str_eq(string a, string b) -> bool` | Equality test |
| `str_index(string s, int64 i) -> int64` | ASCII code of character at index i |
| `int64_to_str(int64 n) -> string` | Format integer as string |
| `float64_to_str(float64 x) -> string` | Format float as string |
| `char_to_str(int64 code) -> string` | Single-character string from ASCII code |

### Conversion
| Function | Description |
|----------|-------------|
| `int64_to_float64(int64 n) -> float64` | Widen integer to float |
| `float64_to_int64(float64 x) -> int64` | Truncate float to integer |

### Math
| Function | Description |
|----------|-------------|
| `sqrt(float64 x) -> float64` | Square root |
| `abs(float64 x) -> float64` | Absolute value |

### File I/O
| Function | Description |
|----------|-------------|
| `file_read(string path) -> string` | Read entire file |
| `file_write(string path, string content)` | Write/overwrite file |
| `file_exists(string path) -> bool` | Check file existence |

### System
| Function | Description |
|----------|-------------|
| `get_arg(int64 n) -> string` | Command-line argument (0-indexed after program name) |
| `cpu_core_count() -> int64` | Number of logical CPU cores |

### ML model
| Function | Description |
|----------|-------------|
| `load_model(string path) -> model` | Load ONNX or GGUF model |
| `model_free(model m)` | Release model resources |
| `gguf_generate(model m, string prompt, int64 max_tokens) -> string` | LLM text generation |
| `gguf_set_temp(model m, float64 temp)` | Set sampling temperature |
| `gguf_set_top_p(model m, float64 p)` | Set nucleus sampling threshold |
| `piper_forward(model m, int64[] phonemes, float64 ns, float64 ls, float64 nw) -> float64[]` | Piper TTS inference |
| `write_wav(string path, float64[] samples, int64 n, int64 rate)` | Write PCM WAV file |

---

## CLI flags

```
ionic <file.ionic> [flags] -o <output>

  -o <name>       Output binary name (default: stem of source file)
  --emit-ir       Print LLVM IR to stdout, do not link
  --dump-ast      Print parsed AST to stderr
  --dump-tokens   Print token stream to stderr
  --run           Compile and immediately execute
```
