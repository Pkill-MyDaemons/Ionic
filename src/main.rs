mod lexer;
mod ast;
mod parser;
mod semantic;
mod codegen;

use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    let args: Vec<String> = env::args().collect();
    let (source_path, flags) = parse_args(&args);

    let source = fs::read_to_string(&source_path)
        .unwrap_or_else(|e| die(&format!("Cannot read '{}': {}", source_path, e)));

    // ── Lex ──────────────────────────────────────────────────────────────────
    let mut lexer = lexer::Lexer::new(&source);
    let tokens = lexer.tokenize().unwrap_or_else(|e| die(&format!("[Lex Error] {}", e)));

    if flags.dump_tokens {
        for t in &tokens {
            eprintln!("  [{:>3}:{:<3}] {:?}", t.line, t.col, t.node);
        }
        return;
    }

    // ── Parse ─────────────────────────────────────────────────────────────────
    let mut parser = parser::Parser::new(tokens);
    let program = parser.parse().unwrap_or_else(|e| die(&format!("[Parse Error] {}", e)));

    if flags.dump_ast {
        eprintln!("{:#?}", program);
        return;
    }

    // ── Semantic analysis ────────────────────────────────────────────────────
    let mut sem = semantic::SemanticAnalyzer::new();
    let errors = sem.analyze(&program);
    if !errors.is_empty() {
        for e in &errors {
            eprintln!("[Type Error] {}", e);
        }
        std::process::exit(1);
    }

    // ── Codegen ───────────────────────────────────────────────────────────────
    let mut cg = codegen::Codegen::new();
    let ir = cg.generate(&program);

    if flags.emit_ir {
        print!("{}", ir);
        return;
    }

    // ── Compile IR → native binary ────────────────────────────────────────────
    let stem = Path::new(&source_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("out");

    let out_bin = flags.output.clone().unwrap_or_else(|| stem.to_string());
    let ir_path = format!("/tmp/{}.ll", stem);

    fs::write(&ir_path, &ir)
        .unwrap_or_else(|e| die(&format!("Cannot write IR: {}", e)));

    let clang = find_clang();
    let status = Command::new(&clang)
        .args([&ir_path, "-o", &out_bin, "-lm"])
        .status()
        .unwrap_or_else(|e| die(&format!("Cannot run clang ({}): {}", clang, e)));

    if !status.success() {
        eprintln!("[Link Error] clang failed. IR saved at {}", ir_path);
        std::process::exit(1);
    }

    if flags.run {
        let status = Command::new(format!("./{}", out_bin))
            .status()
            .unwrap_or_else(|e| die(&format!("Cannot run binary: {}", e)));
        std::process::exit(status.code().unwrap_or(0));
    }

    eprintln!("Compiled: {} -> ./{}", source_path, out_bin);
}

struct Flags {
    dump_tokens: bool,
    dump_ast: bool,
    emit_ir: bool,
    run: bool,
    output: Option<String>,
}

fn parse_args(args: &[String]) -> (String, Flags) {
    if args.len() < 2 {
        eprintln!("ionic - Ionic compiler v0.1.0");
        eprintln!("");
        eprintln!("Usage: ionic <file.ionic> [options]");
        eprintln!("");
        eprintln!("Options:");
        eprintln!("  -o <name>      Output binary name");
        eprintln!("  --run          Compile and immediately execute");
        eprintln!("  --emit-ir      Print LLVM IR to stdout, do not compile");
        eprintln!("  --dump-ast     Print AST to stderr, do not compile");
        eprintln!("  --dump-tokens  Print token stream to stderr, do not compile");
        std::process::exit(0);
    }

    let source = args[1].clone();
    let mut flags = Flags {
        dump_tokens: false,
        dump_ast: false,
        emit_ir: false,
        run: false,
        output: None,
    };

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--dump-tokens" => flags.dump_tokens = true,
            "--dump-ast"    => flags.dump_ast = true,
            "--emit-ir"     => flags.emit_ir = true,
            "--run"         => flags.run = true,
            "-o" => {
                i += 1;
                flags.output = args.get(i).cloned();
            }
            unknown => {
                eprintln!("Unknown flag: {}", unknown);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    (source, flags)
}

fn find_clang() -> String {
    for candidate in &[
        "/opt/homebrew/opt/llvm/bin/clang",
        "/usr/local/opt/llvm/bin/clang",
        "clang",
    ] {
        if Command::new(candidate).arg("--version").output().is_ok() {
            return candidate.to_string();
        }
    }
    "clang".to_string()
}

fn die(msg: &str) -> ! {
    eprintln!("{}", msg);
    std::process::exit(1);
}
