//! Ionic import resolution and selective compilation.
//!
//! When a program contains `import std.math.*;` the compiler:
//!   1. Finds and parses `lib/std/math.ionic` into a LibPool.
//!   2. Walks the main program's AST to collect referenced names.
//!   3. BFS-expands through the pool so that library functions which call
//!      other library functions pull them in transitively.
//!   4. Only the reachable symbols are prepended to program.fns / .structs /
//!      .top_level before codegen — everything else is silently dropped.

use std::collections::{HashMap, HashSet, VecDeque};
use crate::ast::*;

// ── Library pool ──────────────────────────────────────────────────────────────

pub struct LibPool {
    pub fns:     HashMap<String, FnDef>,
    pub structs: HashMap<String, StructDef>,
    pub globals: HashMap<String, Stmt>,    // top-level let/mut (constants like PI)
}

impl LibPool {
    fn new() -> Self {
        LibPool { fns: HashMap::new(), structs: HashMap::new(), globals: HashMap::new() }
    }
}

// ── Library path resolution ───────────────────────────────────────────────────

fn resolve_lib_path(path: &[String]) -> Option<std::path::PathBuf> {
    let rel = path.join("/") + ".ionic";

    // 1. Dev mode: ./lib/<path>.ionic
    let dev = std::path::PathBuf::from("lib").join(&rel);
    if dev.exists() { return Some(dev); }

    // 2. Installed: <binary_dir>/../lib/<path>.ionic
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let inst = parent.join("../lib").join(&rel);
            if inst.exists() { return Some(inst.canonicalize().unwrap_or(inst)); }
        }
    }

    None
}

// ── Build pool from all imports ───────────────────────────────────────────────

pub fn build_lib_pool(imports: &[Import]) -> Result<LibPool, String> {
    let mut pool = LibPool::new();

    for imp in imports {
        let file_path = resolve_lib_path(&imp.path)
            .ok_or_else(|| format!(
                "Cannot find library module `{}` — searched ./lib/{}.ionic",
                imp.path.join("."),
                imp.path.join("/")
            ))?;

        let src = std::fs::read_to_string(&file_path)
            .map_err(|e| format!("Cannot read {}: {}", file_path.display(), e))?;

        let mut lexer = crate::lexer::Lexer::new(&src);
        let tokens = lexer.tokenize()
            .map_err(|e| format!("Lex error in {}: {}", file_path.display(), e))?;
        let mut parser = crate::parser::Parser::new(tokens);
        let lib_prog = parser.parse()
            .map_err(|e| format!("Parse error in {}: {}", file_path.display(), e))?;

        // Decide which names to admit
        let admitted: Option<HashSet<String>> = match &imp.kind {
            ImportKind::Named(names) => Some(names.iter().cloned().collect()),
            ImportKind::Glob | ImportKind::Module => None,
        };
        let admit = |name: &str| admitted.as_ref().map_or(true, |a| a.contains(name));

        for fd in lib_prog.fns {
            if admit(&fd.name) { pool.fns.insert(fd.name.clone(), fd); }
        }
        for sd in lib_prog.structs {
            if admit(&sd.name) { pool.structs.insert(sd.name.clone(), sd); }
        }
        for stmt in lib_prog.top_level {
            if let Stmt::Let { ref name, .. } | Stmt::Assign { target: AssignTarget::Ident(ref name), .. } = stmt {
                let n = name.clone();
                if admit(&n) { pool.globals.insert(n, stmt); }
            }
        }
    }

    Ok(pool)
}

// ── AST name-reference collector ─────────────────────────────────────────────

fn refs_expr(expr: &Expr, out: &mut HashSet<String>) {
    match expr {
        Expr::Ident(name)               => { out.insert(name.clone()); }
        Expr::Call { callee, args }     => {
            out.insert(callee.clone());
            for a in args { refs_expr(a, out); }
        }
        Expr::MethodCall { obj, args, .. } => {
            refs_expr(obj, out);
            for a in args { refs_expr(a, out); }
        }
        Expr::BinOp { lhs, rhs, .. }   => { refs_expr(lhs, out); refs_expr(rhs, out); }
        Expr::UnOp { expr, .. }         => { refs_expr(expr, out); }
        Expr::StructLit { name, fields } => {
            out.insert(name.clone());
            for (_, e) in fields { refs_expr(e, out); }
        }
        Expr::FieldAccess { obj, .. }   => { refs_expr(obj, out); }
        Expr::Index { obj, idx }        => { refs_expr(obj, out); refs_expr(idx, out); }
        Expr::ArrayLit(elems)           => { for e in elems { refs_expr(e, out); } }
        Expr::ToGpu(e) | Expr::ToCpu(e) => { refs_expr(e, out); }
        Expr::IntLit(_) | Expr::FloatLit(_)
        | Expr::StringLit(_) | Expr::BoolLit(_) => {}
    }
}

fn refs_stmt(stmt: &Stmt, out: &mut HashSet<String>) {
    match stmt {
        Stmt::Let { init, .. }              => refs_expr(init, out),
        Stmt::Assign { value, target }      => {
            refs_expr(value, out);
            match target {
                AssignTarget::Field(e, _)    => refs_expr(e, out),
                AssignTarget::Index(o, i)    => { refs_expr(o, out); refs_expr(i, out); }
                AssignTarget::Ident(_)       => {}
            }
        }
        Stmt::ExprStmt(e)                   => refs_expr(e, out),
        Stmt::Return(Some(e))               => refs_expr(e, out),
        Stmt::Return(None)                  => {}
        Stmt::If { cond, then_block, else_block } => {
            refs_expr(cond, out);
            for s in then_block { refs_stmt(s, out); }
            if let Some(eb) = else_block { for s in eb { refs_stmt(s, out); } }
        }
        Stmt::While { cond, body }          => {
            refs_expr(cond, out);
            for s in body { refs_stmt(s, out); }
        }
        Stmt::For { start, end, body, .. }  => {
            refs_expr(start, out);
            refs_expr(end, out);
            for s in body { refs_stmt(s, out); }
        }
        Stmt::GpuBlock(body)                => { for s in body { refs_stmt(s, out); } }
        Stmt::Import(_) | Stmt::Break | Stmt::Continue => {}
    }
}

fn refs_type(ty: &TypeAnnotation, out: &mut HashSet<String>) {
    match ty {
        TypeAnnotation::Named(n)    => { out.insert(n.clone()); }
        TypeAnnotation::Array(inner) => refs_type(inner, out),
        _ => {}
    }
}

fn refs_fn(fd: &FnDef, out: &mut HashSet<String>) {
    for p in &fd.params { refs_type(&p.ty, out); }
    refs_type(&fd.ret_ty, out);
    for s in &fd.body { refs_stmt(s, out); }
}

// ── BFS reachability ──────────────────────────────────────────────────────────

/// Returns `(lib_fns, lib_structs, lib_globals)` — only the symbols reachable
/// from the main program, in BFS visit order (callees before callers).
pub fn bfs_reachable(
    main: &Program,
    pool: &LibPool,
) -> (Vec<FnDef>, Vec<StructDef>, Vec<Stmt>) {
    // Seed: all names referenced by the main program
    let mut seed: HashSet<String> = HashSet::new();
    for f in &main.fns   { refs_fn(f, &mut seed); }
    for s in &main.top_level { refs_stmt(s, &mut seed); }
    for s in &main.structs {
        for f in &s.fields { refs_type(&f.ty, &mut seed); }
    }

    let in_pool = |name: &str| {
        pool.fns.contains_key(name)
            || pool.structs.contains_key(name)
            || pool.globals.contains_key(name)
    };

    // BFS queue primed with seed names that exist in pool
    let mut queue: VecDeque<String> = seed.iter()
        .filter(|n| in_pool(n))
        .cloned()
        .collect();
    let mut visited:   HashSet<String> = HashSet::new();
    let mut visit_ord: Vec<String>     = Vec::new();  // BFS order = callee-first

    while let Some(name) = queue.pop_front() {
        if visited.contains(&name) { continue; }
        visited.insert(name.clone());
        visit_ord.push(name.clone());

        // Expand through library fn bodies
        if let Some(fd) = pool.fns.get(&name) {
            let mut inner = HashSet::new();
            refs_fn(fd, &mut inner);
            for r in inner {
                if !visited.contains(&r) && in_pool(&r) { queue.push_back(r); }
            }
        }
        // Expand through library struct field types
        if let Some(sd) = pool.structs.get(&name) {
            let mut inner = HashSet::new();
            for f in &sd.fields { refs_type(&f.ty, &mut inner); }
            for r in inner {
                if !visited.contains(&r) && in_pool(&r) { queue.push_back(r); }
            }
        }
        // Globals (constants) — expand any references in their init expressions
        if let Some(Stmt::Let { init, .. }) = pool.globals.get(&name) {
            let mut inner = HashSet::new();
            refs_expr(init, &mut inner);
            for r in inner {
                if !visited.contains(&r) && in_pool(&r) { queue.push_back(r); }
            }
        }
    }

    // Collect in BFS visit order (naturally callee-before-caller)
    let lib_fns: Vec<FnDef> = visit_ord.iter()
        .filter_map(|n| pool.fns.get(n).cloned())
        .collect();
    let lib_structs: Vec<StructDef> = visit_ord.iter()
        .filter_map(|n| pool.structs.get(n).cloned())
        .collect();
    let lib_globals: Vec<Stmt> = visit_ord.iter()
        .filter_map(|n| pool.globals.get(n).cloned())
        .collect();

    (lib_fns, lib_structs, lib_globals)
}
