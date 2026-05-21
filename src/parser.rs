use crate::ast::*;
use crate::lexer::{SpannedToken, Token};

pub struct Parser {
    tokens: Vec<SpannedToken>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<SpannedToken>) -> Self {
        Parser { tokens, pos: 0 }
    }

    // ── Token navigation ──────────────────────────────────────────────────────

    fn peek(&self) -> &Token {
        &self.tokens[self.pos].node
    }

    fn peek_ahead(&self, offset: usize) -> &Token {
        let idx = (self.pos + offset).min(self.tokens.len() - 1);
        &self.tokens[idx].node
    }

    fn line(&self) -> usize {
        self.tokens[self.pos].line
    }

    fn advance(&mut self) -> Token {
        let t = self.tokens[self.pos].node.clone();
        if t != Token::EOF {
            self.pos += 1;
        }
        t
    }

    fn expect(&mut self, expected: &Token) -> Result<(), String> {
        if self.peek() == expected {
            self.advance();
            Ok(())
        } else {
            Err(format!(
                "Line {}: expected `{}`, got `{}`",
                self.line(), expected, self.peek()
            ))
        }
    }

    fn expect_ident(&mut self) -> Result<String, String> {
        match self.peek().clone() {
            Token::Ident(s) => { self.advance(); Ok(s) }
            t => Err(format!("Line {}: expected identifier, got `{}`", self.line(), t)),
        }
    }

    // ── Entry point ───────────────────────────────────────────────────────────

    pub fn parse(&mut self) -> Result<Program, String> {
        let mut imports = Vec::new();
        let mut structs = Vec::new();
        let mut fns = Vec::new();
        let mut top_level = Vec::new();

        while *self.peek() != Token::EOF {
            // Collect consecutive /// doc-comment lines
            let mut doc_lines: Vec<String> = Vec::new();
            while matches!(self.peek(), Token::DocComment(_)) {
                if let Token::DocComment(s) = self.advance() {
                    doc_lines.push(s);
                }
            }
            let doc: Option<String> = if doc_lines.is_empty() {
                None
            } else {
                Some(doc_lines.join("\n"))
            };

            let hw = self.parse_hw_annotation()?;
            match self.peek().clone() {
                Token::Import => {
                    imports.push(self.parse_import()?);
                }
                Token::Struct => {
                    let mut sd = self.parse_struct()?;
                    sd.hw = hw;
                    sd.doc = doc;
                    structs.push(sd);
                }
                Token::Fn => {
                    let mut fd = self.parse_fn()?;
                    fd.hw = hw;
                    fd.doc = doc;
                    fns.push(fd);
                }
                _ => {
                    top_level.push(self.parse_stmt_hw(hw)?);
                }
            }
        }

        Ok(Program { imports, structs, fns, top_level })
    }

    // ── Hardware annotation ───────────────────────────────────────────────────

    fn parse_hw_annotation(&mut self) -> Result<Option<HwAnnotation>, String> {
        let ann = match self.peek().clone() {
            Token::AtGpu => {
                self.advance();
                if *self.peek() == Token::LParen {
                    self.advance();
                    let frac = match self.peek().clone() {
                        Token::FloatLit(f) => { self.advance(); f }
                        Token::IntLit(n)   => { self.advance(); n as f64 }
                        t => return Err(format!("Line {}: expected fraction in @gpu(...), got `{}`", self.line(), t)),
                    };
                    self.expect(&Token::RParen)?;
                    HwAnnotation::GpuFrac(frac)
                } else {
                    HwAnnotation::Gpu
                }
            }
            Token::AtCpu => {
                self.advance();
                if *self.peek() == Token::LParen {
                    self.advance();
                    let frac = match self.peek().clone() {
                        Token::FloatLit(f) => { self.advance(); f }
                        Token::IntLit(n)   => { self.advance(); n as f64 }
                        t => return Err(format!("Line {}: expected fraction in @cpu(...), got `{}`", self.line(), t)),
                    };
                    self.expect(&Token::RParen)?;
                    HwAnnotation::CpuFrac(frac)
                } else {
                    HwAnnotation::Cpu
                }
            }
            _ => return Ok(None),
        };
        Ok(Some(ann))
    }

    // ── Import ────────────────────────────────────────────────────────────────

    fn parse_import(&mut self) -> Result<Import, String> {
        self.advance(); // consume `import`
        let mut path = vec![self.expect_ident()?];

        // Consume dotted path segments, stopping before `.*` or `.{` or `;`
        loop {
            if *self.peek() != Token::Dot { break; }
            // Peek one further: if it's Star or LBrace, stop path parsing
            match self.peek_ahead(1) {
                Token::Star | Token::LBrace => break,
                Token::EOF | Token::Semicolon => break,
                _ => {}
            }
            self.advance(); // consume `.`
            path.push(self.expect_ident()?);
        }

        let kind = if *self.peek() == Token::Dot {
            self.advance(); // consume `.`
            if *self.peek() == Token::Star {
                self.advance(); // consume `*`
                ImportKind::Glob
            } else if *self.peek() == Token::LBrace {
                self.advance(); // consume `{`
                let mut names = Vec::new();
                while *self.peek() != Token::RBrace && *self.peek() != Token::EOF {
                    names.push(self.expect_ident()?);
                    if *self.peek() == Token::Comma { self.advance(); }
                }
                self.expect(&Token::RBrace)?;
                ImportKind::Named(names)
            } else {
                return Err(format!("Line {}: expected `*` or `{{` after `.` in import", self.line()));
            }
        } else {
            ImportKind::Module
        };

        self.expect(&Token::Semicolon)?;
        Ok(Import { path, kind })
    }

    // ── Struct definition ─────────────────────────────────────────────────────

    fn parse_struct(&mut self) -> Result<StructDef, String> {
        self.advance(); // `struct`
        let name = self.expect_ident()?;
        self.expect(&Token::LBrace)?;
        let mut fields = Vec::new();
        while *self.peek() != Token::RBrace && *self.peek() != Token::EOF {
            let field_name = self.expect_ident()?;
            self.expect(&Token::Colon)?;
            let ty = self.parse_type_annotation()?;
            fields.push(FieldDef { name: field_name, ty });
            if *self.peek() == Token::Comma {
                self.advance();
            }
        }
        self.expect(&Token::RBrace)?;
        Ok(StructDef { name, fields, hw: None, doc: None })
    }

    // ── Function definition ───────────────────────────────────────────────────

    fn parse_fn(&mut self) -> Result<FnDef, String> {
        let line = self.line();
        self.advance(); // `fn`
        let name = self.expect_ident()?;
        self.expect(&Token::LParen)?;

        let mut params = Vec::new();
        while *self.peek() != Token::RParen && *self.peek() != Token::EOF {
            // optional @gpu/@cpu annotation on parameter
            let phw = self.parse_hw_annotation()?;
            let ty = self.parse_type_annotation()?;
            let pname = self.expect_ident()?;
            params.push(Param { name: pname, ty, hw: phw });
            if *self.peek() == Token::Comma { self.advance(); }
        }
        self.expect(&Token::RParen)?;

        let ret_ty = if *self.peek() == Token::Arrow {
            self.advance();
            self.parse_type_annotation()?
        } else {
            TypeAnnotation::Void
        };

        let body = self.parse_block()?;
        Ok(FnDef { name, params, ret_ty, body, line, hw: None, doc: None })
    }

    // ── Type annotation ───────────────────────────────────────────────────────

    fn parse_type_annotation(&mut self) -> Result<TypeAnnotation, String> {
        match self.peek().clone() {
            Token::KwInt64   => { self.advance(); Ok(TypeAnnotation::Int64) }
            Token::KwFloat64 => { self.advance(); Ok(TypeAnnotation::Float64) }
            Token::KwBool    => { self.advance(); Ok(TypeAnnotation::Bool) }
            Token::KwString  => { self.advance(); Ok(TypeAnnotation::Str) }
            Token::KwVoid    => { self.advance(); Ok(TypeAnnotation::Void) }
            Token::KwTensor  => {
                self.advance();
                let hw = match self.peek() {
                    Token::AtGpu => { self.advance(); Some(HwTarget::Gpu) }
                    Token::AtCpu => { self.advance(); Some(HwTarget::Cpu) }
                    _ => None,
                };
                Ok(TypeAnnotation::Tensor(hw))
            }
            Token::KwModel => {
                self.advance();
                let hw = match self.peek() {
                    Token::AtGpu => { self.advance(); Some(HwTarget::Gpu) }
                    Token::AtCpu => { self.advance(); Some(HwTarget::Cpu) }
                    _ => None,
                };
                Ok(TypeAnnotation::Model(hw))
            }
            Token::LBracket => {
                self.advance();
                let inner = self.parse_type_annotation()?;
                self.expect(&Token::RBracket)?;
                Ok(TypeAnnotation::Array(Box::new(inner)))
            }
            Token::Ident(name) => {
                self.advance();
                Ok(TypeAnnotation::Named(name))
            }
            t => Err(format!("Line {}: expected type, got `{}`", self.line(), t)),
        }
    }

    // ── Block ─────────────────────────────────────────────────────────────────

    fn parse_block(&mut self) -> Result<Block, String> {
        self.expect(&Token::LBrace)?;
        let mut stmts = Vec::new();
        while *self.peek() != Token::RBrace && *self.peek() != Token::EOF {
            stmts.push(self.parse_stmt()?);
        }
        self.expect(&Token::RBrace)?;
        Ok(stmts)
    }

    // ── Statements ────────────────────────────────────────────────────────────

    fn parse_stmt(&mut self) -> Result<Stmt, String> {
        self.parse_stmt_hw(None)
    }

    fn parse_stmt_hw(&mut self, hw: Option<HwAnnotation>) -> Result<Stmt, String> {
        // Allow @gpu/@cpu inside blocks too (e.g., @gpu let x = ...)
        let hw = if hw.is_some() {
            hw
        } else {
            self.parse_hw_annotation()?
        };
        match self.peek().clone() {
            Token::Let => self.parse_let(false, hw),
            Token::Mut => self.parse_let(true, hw),

            Token::Return => {
                self.advance();
                if *self.peek() == Token::Semicolon {
                    self.advance();
                    Ok(Stmt::Return(None))
                } else {
                    let e = self.parse_expr()?;
                    self.expect(&Token::Semicolon)?;
                    Ok(Stmt::Return(Some(e)))
                }
            }

            Token::Break => {
                self.advance();
                self.expect(&Token::Semicolon)?;
                Ok(Stmt::Break)
            }

            Token::Continue => {
                self.advance();
                self.expect(&Token::Semicolon)?;
                Ok(Stmt::Continue)
            }

            Token::If    => self.parse_if(),
            Token::While => self.parse_while(),
            Token::For   => self.parse_for(),

            Token::Gpu => {
                self.advance();
                let body = self.parse_block()?;
                Ok(Stmt::GpuBlock(body))
            }

            Token::Import => {
                let imp = self.parse_import()?;
                Ok(Stmt::Import(imp))
            }

            Token::Ident(name) => {
                // Look ahead to detect assignment: x = , x.field = , x[i] =
                if self.is_assignment_ahead() {
                    self.parse_assign()
                } else {
                    let e = self.parse_expr()?;
                    self.expect(&Token::Semicolon)?;
                    Ok(Stmt::ExprStmt(e))
                }
            }

            _ => {
                let e = self.parse_expr()?;
                self.expect(&Token::Semicolon)?;
                Ok(Stmt::ExprStmt(e))
            }
        }
    }

    /// Peek ahead to determine if this looks like an assignment statement.
    fn is_assignment_ahead(&self) -> bool {
        let mut i = 1;
        loop {
            match self.peek_ahead(i) {
                Token::Eq => return true,
                Token::Dot => {
                    i += 1; // skip method/field name
                    i += 1;
                    // check if next is =
                }
                Token::LBracket => {
                    // skip until matching ]
                    let mut depth = 1;
                    i += 1;
                    while depth > 0 {
                        match self.peek_ahead(i) {
                            Token::LBracket => { depth += 1; i += 1; }
                            Token::RBracket => { depth -= 1; i += 1; }
                            Token::EOF => return false,
                            _ => { i += 1; }
                        }
                    }
                }
                Token::LParen | Token::EqEq | Token::BangEq | Token::Lt | Token::Gt
                | Token::Plus | Token::Minus | Token::Star | Token::Slash | Token::Semicolon
                | Token::EOF => return false,
                _ => return false,
            }
            if i > 10 { return false; }
        }
    }

    fn parse_assign(&mut self) -> Result<Stmt, String> {
        // Parse an lvalue expression then `=` then rvalue
        let lhs_expr = self.parse_lvalue()?;
        self.expect(&Token::Eq)?;
        let value = self.parse_expr()?;
        self.expect(&Token::Semicolon)?;

        let target = match lhs_expr {
            Expr::Ident(name) => AssignTarget::Ident(name),
            Expr::FieldAccess { obj, field } => AssignTarget::Field(obj, field),
            Expr::Index { obj, idx } => AssignTarget::Index(obj, idx),
            _ => return Err(format!("Line {}: invalid assignment target", self.line())),
        };

        Ok(Stmt::Assign { target, value })
    }

    fn parse_lvalue(&mut self) -> Result<Expr, String> {
        let name = self.expect_ident()?;
        let mut base: Expr = Expr::Ident(name);
        loop {
            match self.peek() {
                Token::Dot => {
                    self.advance();
                    let field = self.expect_ident()?;
                    base = Expr::FieldAccess { obj: Box::new(base), field };
                }
                Token::LBracket => {
                    self.advance();
                    let idx = self.parse_expr()?;
                    self.expect(&Token::RBracket)?;
                    base = Expr::Index { obj: Box::new(base), idx: Box::new(idx) };
                }
                _ => break,
            }
        }
        Ok(base)
    }

    fn parse_let(&mut self, mutable: bool, hw: Option<HwAnnotation>) -> Result<Stmt, String> {
        self.advance(); // `let` or `mut`
        let name = self.expect_ident()?;
        let ty = if *self.peek() == Token::Colon {
            self.advance();
            Some(self.parse_type_annotation()?)
        } else {
            None
        };
        self.expect(&Token::Eq)?;
        let init = self.parse_expr()?;
        self.expect(&Token::Semicolon)?;
        Ok(Stmt::Let { mutable, name, ty, init, hw })
    }

    fn parse_if(&mut self) -> Result<Stmt, String> {
        self.advance(); // `if`
        self.expect(&Token::LParen)?;
        let cond = self.parse_expr()?;
        self.expect(&Token::RParen)?;
        let then_block = self.parse_block()?;
        let else_block = if *self.peek() == Token::Else {
            self.advance();
            if *self.peek() == Token::If {
                // else if chain
                Some(vec![self.parse_if()?])
            } else {
                Some(self.parse_block()?)
            }
        } else {
            None
        };
        Ok(Stmt::If { cond, then_block, else_block })
    }

    fn parse_while(&mut self) -> Result<Stmt, String> {
        self.advance(); // `while`
        self.expect(&Token::LParen)?;
        let cond = self.parse_expr()?;
        self.expect(&Token::RParen)?;
        let body = self.parse_block()?;
        Ok(Stmt::While { cond, body })
    }

    fn parse_for(&mut self) -> Result<Stmt, String> {
        self.advance(); // `for`
        self.expect(&Token::LParen)?;
        let var = self.expect_ident()?;
        self.expect(&Token::In)?;
        let start = self.parse_expr()?;
        self.expect(&Token::DotDot)?;
        let end = self.parse_expr()?;
        self.expect(&Token::RParen)?;
        let body = self.parse_block()?;
        Ok(Stmt::For { var, start, end, body })
    }

    // ── Expressions (precedence climbing) ────────────────────────────────────

    pub fn parse_expr(&mut self) -> Result<Expr, String> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_and()?;
        while *self.peek() == Token::OrOr {
            self.advance();
            let rhs = self.parse_and()?;
            lhs = Expr::BinOp { op: BinOp::Or, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_eq()?;
        while *self.peek() == Token::AndAnd {
            self.advance();
            let rhs = self.parse_eq()?;
            lhs = Expr::BinOp { op: BinOp::And, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
        Ok(lhs)
    }

    fn parse_eq(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_cmp()?;
        loop {
            let op = match self.peek() {
                Token::EqEq  => BinOp::EqEq,
                Token::BangEq => BinOp::NotEq,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_cmp()?;
            lhs = Expr::BinOp { op, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
        Ok(lhs)
    }

    fn parse_cmp(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_add()?;
        loop {
            let op = match self.peek() {
                Token::Lt   => BinOp::Lt,
                Token::Gt   => BinOp::Gt,
                Token::LtEq => BinOp::LtEq,
                Token::GtEq => BinOp::GtEq,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_add()?;
            lhs = Expr::BinOp { op, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
        Ok(lhs)
    }

    fn parse_add(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_mul()?;
        loop {
            let op = match self.peek() {
                Token::Plus  => BinOp::Add,
                Token::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_mul()?;
            lhs = Expr::BinOp { op, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
        Ok(lhs)
    }

    fn parse_mul(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                Token::Star    => BinOp::Mul,
                Token::Slash   => BinOp::Div,
                Token::Percent => BinOp::Mod,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_unary()?;
            lhs = Expr::BinOp { op, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<Expr, String> {
        match self.peek().clone() {
            Token::Minus => {
                self.advance();
                Ok(Expr::UnOp { op: UnOp::Neg, expr: Box::new(self.parse_unary()?) })
            }
            Token::Bang => {
                self.advance();
                Ok(Expr::UnOp { op: UnOp::Not, expr: Box::new(self.parse_unary()?) })
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<Expr, String> {
        let mut base = self.parse_primary()?;
        loop {
            match self.peek() {
                Token::Dot => {
                    self.advance();
                    let name = self.expect_ident()?;

                    if name == "toGpu" {
                        self.expect(&Token::LParen)?;
                        self.expect(&Token::RParen)?;
                        base = Expr::ToGpu(Box::new(base));
                        continue;
                    }
                    if name == "toCpu" {
                        self.expect(&Token::LParen)?;
                        self.expect(&Token::RParen)?;
                        base = Expr::ToCpu(Box::new(base));
                        continue;
                    }

                    if *self.peek() == Token::LParen {
                        self.advance();
                        let args = self.parse_arg_list()?;
                        base = Expr::MethodCall { obj: Box::new(base), method: name, args };
                    } else {
                        base = Expr::FieldAccess { obj: Box::new(base), field: name };
                    }
                }
                Token::LBracket => {
                    self.advance();
                    let idx = self.parse_expr()?;
                    self.expect(&Token::RBracket)?;
                    base = Expr::Index { obj: Box::new(base), idx: Box::new(idx) };
                }
                _ => break,
            }
        }
        Ok(base)
    }

    fn parse_arg_list(&mut self) -> Result<Vec<Expr>, String> {
        let mut args = Vec::new();
        while *self.peek() != Token::RParen && *self.peek() != Token::EOF {
            args.push(self.parse_expr()?);
            if *self.peek() == Token::Comma { self.advance(); }
        }
        self.expect(&Token::RParen)?;
        Ok(args)
    }

    fn parse_primary(&mut self) -> Result<Expr, String> {
        match self.peek().clone() {
            Token::IntLit(n)    => { self.advance(); Ok(Expr::IntLit(n)) }
            Token::FloatLit(f)  => { self.advance(); Ok(Expr::FloatLit(f)) }
            Token::StringLit(s) => { self.advance(); Ok(Expr::StringLit(s)) }
            Token::True         => { self.advance(); Ok(Expr::BoolLit(true)) }
            Token::False        => { self.advance(); Ok(Expr::BoolLit(false)) }

            Token::LParen => {
                self.advance();
                let e = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(e)
            }

            // Array literal: [a, b, c]
            Token::LBracket => {
                self.advance();
                let mut elems = Vec::new();
                while *self.peek() != Token::RBracket && *self.peek() != Token::EOF {
                    elems.push(self.parse_expr()?);
                    if *self.peek() == Token::Comma { self.advance(); }
                }
                self.expect(&Token::RBracket)?;
                Ok(Expr::ArrayLit(elems))
            }

            Token::Ident(name) => {
                self.advance();
                match self.peek() {
                    Token::LParen => {
                        // Function call
                        self.advance();
                        let args = self.parse_arg_list()?;
                        Ok(Expr::Call { callee: name, args })
                    }
                    Token::LBrace => {
                        // Struct literal: Name { field: expr, ... }
                        self.advance();
                        let mut fields = Vec::new();
                        while *self.peek() != Token::RBrace && *self.peek() != Token::EOF {
                            let fname = self.expect_ident()?;
                            self.expect(&Token::Colon)?;
                            let val = self.parse_expr()?;
                            fields.push((fname, val));
                            if *self.peek() == Token::Comma { self.advance(); }
                        }
                        self.expect(&Token::RBrace)?;
                        Ok(Expr::StructLit { name, fields })
                    }
                    _ => Ok(Expr::Ident(name)),
                }
            }

            t => Err(format!("Line {}: unexpected `{}` in expression", self.line(), t)),
        }
    }
}
