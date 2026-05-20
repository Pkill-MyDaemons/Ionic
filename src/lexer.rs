use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Let,
    Mut,
    Fn,
    Return,
    If,
    Else,
    While,
    For,
    In,
    Import,
    Gpu,
    Cpu,
    True,
    False,
    Struct,
    Break,
    Continue,
    // Primitive types
    KwInt64,
    KwFloat64,
    KwBool,
    KwString,
    KwTensor,
    KwModel,
    KwVoid,
    // Hardware annotations
    AtGpu,
    AtCpu,
    // Literals
    IntLit(i64),
    FloatLit(f64),
    StringLit(String),
    // Identifier
    Ident(String),
    // Arithmetic operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    // Comparison operators
    EqEq,
    BangEq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    // Logical operators
    AndAnd,
    OrOr,
    Bang,
    // Assignment
    Eq,
    // Punctuation
    Arrow,    // ->
    Dot,
    DotDot,   // ..
    Colon,
    Semicolon,
    Comma,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    // End of file
    EOF,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Let => write!(f, "let"),
            Token::Mut => write!(f, "mut"),
            Token::Fn => write!(f, "fn"),
            Token::Return => write!(f, "return"),
            Token::If => write!(f, "if"),
            Token::Else => write!(f, "else"),
            Token::While => write!(f, "while"),
            Token::For => write!(f, "for"),
            Token::In => write!(f, "in"),
            Token::Import => write!(f, "import"),
            Token::Gpu => write!(f, "gpu"),
            Token::Cpu => write!(f, "cpu"),
            Token::True => write!(f, "true"),
            Token::False => write!(f, "false"),
            Token::Struct => write!(f, "struct"),
            Token::Break => write!(f, "break"),
            Token::Continue => write!(f, "continue"),
            Token::KwInt64 => write!(f, "int64"),
            Token::KwFloat64 => write!(f, "float64"),
            Token::KwBool => write!(f, "bool"),
            Token::KwString => write!(f, "string"),
            Token::KwTensor => write!(f, "tensor"),
            Token::KwModel  => write!(f, "model"),
            Token::KwVoid   => write!(f, "void"),
            Token::AtGpu => write!(f, "@gpu"),
            Token::AtCpu => write!(f, "@cpu"),
            Token::IntLit(n) => write!(f, "{}", n),
            Token::FloatLit(n) => write!(f, "{}", n),
            Token::StringLit(s) => write!(f, "\"{}\"", s),
            Token::Ident(s) => write!(f, "{}", s),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Star => write!(f, "*"),
            Token::Slash => write!(f, "/"),
            Token::Percent => write!(f, "%"),
            Token::EqEq => write!(f, "=="),
            Token::BangEq => write!(f, "!="),
            Token::Lt => write!(f, "<"),
            Token::Gt => write!(f, ">"),
            Token::LtEq => write!(f, "<="),
            Token::GtEq => write!(f, ">="),
            Token::AndAnd => write!(f, "&&"),
            Token::OrOr => write!(f, "||"),
            Token::Bang => write!(f, "!"),
            Token::Eq => write!(f, "="),
            Token::Arrow => write!(f, "->"),
            Token::Dot => write!(f, "."),
            Token::DotDot => write!(f, ".."),
            Token::Colon => write!(f, ":"),
            Token::Semicolon => write!(f, ";"),
            Token::Comma => write!(f, ","),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::LBrace => write!(f, "{{"),
            Token::RBrace => write!(f, "}}"),
            Token::LBracket => write!(f, "["),
            Token::RBracket => write!(f, "]"),
            Token::EOF => write!(f, "<EOF>"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Spanned<T> {
    pub node: T,
    pub line: usize,
    pub col: usize,
}

pub type SpannedToken = Spanned<Token>;

pub struct Lexer {
    src: Vec<char>,
    pos: usize,
    line: usize,
    col: usize,
}

impl Lexer {
    pub fn new(source: &str) -> Self {
        Lexer {
            src: source.chars().collect(),
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    pub fn tokenize(&mut self) -> Result<Vec<SpannedToken>, String> {
        let mut tokens = Vec::new();
        loop {
            let tok = self.next_token()?;
            let is_eof = tok.node == Token::EOF;
            tokens.push(tok);
            if is_eof {
                break;
            }
        }
        Ok(tokens)
    }

    fn peek(&self) -> Option<char> {
        self.src.get(self.pos).copied()
    }

    fn peek2(&self) -> Option<char> {
        self.src.get(self.pos + 1).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.src.get(self.pos).copied();
        if let Some(ch) = c {
            self.pos += 1;
            if ch == '\n' {
                self.line += 1;
                self.col = 1;
            } else {
                self.col += 1;
            }
        }
        c
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            while matches!(self.peek(), Some(c) if c.is_whitespace()) {
                self.advance();
            }
            if self.peek() == Some('/') && self.peek2() == Some('/') {
                while self.peek().is_some() && self.peek() != Some('\n') {
                    self.advance();
                }
            } else {
                break;
            }
        }
    }

    fn spanned(&self, node: Token, line: usize, col: usize) -> SpannedToken {
        Spanned { node, line, col }
    }

    fn next_token(&mut self) -> Result<SpannedToken, String> {
        self.skip_whitespace_and_comments();
        let line = self.line;
        let col = self.col;

        let ch = match self.peek() {
            None => return Ok(self.spanned(Token::EOF, line, col)),
            Some(c) => c,
        };

        // String literal
        if ch == '"' {
            self.advance();
            let mut s = String::new();
            loop {
                match self.advance() {
                    None => return Err(format!("Unterminated string at line {}", line)),
                    Some('"') => break,
                    Some('\\') => match self.advance() {
                        Some('n') => s.push('\n'),
                        Some('t') => s.push('\t'),
                        Some('"') => s.push('"'),
                        Some('\\') => s.push('\\'),
                        Some(e) => s.push(e),
                        None => return Err("Unterminated escape".to_string()),
                    },
                    Some(c) => s.push(c),
                }
            }
            return Ok(self.spanned(Token::StringLit(s), line, col));
        }

        // Hardware annotations @gpu / @cpu
        if ch == '@' {
            self.advance();
            let kw = self.read_ident();
            let tok = match kw.as_str() {
                "gpu" => Token::AtGpu,
                "cpu" => Token::AtCpu,
                _ => return Err(format!("Unknown annotation @{} at line {}", kw, line)),
            };
            return Ok(self.spanned(tok, line, col));
        }

        // Numbers
        if ch.is_ascii_digit() {
            return self.read_number(line, col);
        }

        // Identifiers and keywords
        if ch.is_alphabetic() || ch == '_' {
            let ident = self.read_ident();
            let tok = match ident.as_str() {
                "let" => Token::Let,
                "mut" => Token::Mut,
                "fn" => Token::Fn,
                "return" => Token::Return,
                "if" => Token::If,
                "else" => Token::Else,
                "while" => Token::While,
                "for" => Token::For,
                "in" => Token::In,
                "import" => Token::Import,
                "gpu" => Token::Gpu,
                "cpu" => Token::Cpu,
                "true" => Token::True,
                "false" => Token::False,
                "struct" => Token::Struct,
                "break" => Token::Break,
                "continue" => Token::Continue,
                "int64" => Token::KwInt64,
                "float64" => Token::KwFloat64,
                "bool" => Token::KwBool,
                "string" => Token::KwString,
                "tensor" => Token::KwTensor,
                "model"  => Token::KwModel,
                "void"   => Token::KwVoid,
                _ => Token::Ident(ident),
            };
            return Ok(self.spanned(tok, line, col));
        }

        self.advance();

        let tok = match ch {
            '+' => Token::Plus,
            '*' => Token::Star,
            '%' => Token::Percent,
            '.' => {
                if self.peek() == Some('.') {
                    self.advance();
                    Token::DotDot
                } else {
                    Token::Dot
                }
            }
            ',' => Token::Comma,
            ';' => Token::Semicolon,
            ':' => Token::Colon,
            '(' => Token::LParen,
            ')' => Token::RParen,
            '{' => Token::LBrace,
            '}' => Token::RBrace,
            '[' => Token::LBracket,
            ']' => Token::RBracket,
            '-' => {
                if self.peek() == Some('>') {
                    self.advance();
                    Token::Arrow
                } else {
                    Token::Minus
                }
            }
            '/' => Token::Slash,
            '=' => {
                if self.peek() == Some('=') {
                    self.advance();
                    Token::EqEq
                } else {
                    Token::Eq
                }
            }
            '!' => {
                if self.peek() == Some('=') {
                    self.advance();
                    Token::BangEq
                } else {
                    Token::Bang
                }
            }
            '<' => {
                if self.peek() == Some('=') {
                    self.advance();
                    Token::LtEq
                } else {
                    Token::Lt
                }
            }
            '>' => {
                if self.peek() == Some('=') {
                    self.advance();
                    Token::GtEq
                } else {
                    Token::Gt
                }
            }
            '&' => {
                if self.peek() == Some('&') {
                    self.advance();
                    Token::AndAnd
                } else {
                    return Err(format!("Unexpected '&' at line {line}, col {col}"));
                }
            }
            '|' => {
                if self.peek() == Some('|') {
                    self.advance();
                    Token::OrOr
                } else {
                    return Err(format!("Unexpected '|' at line {line}, col {col}"));
                }
            }
            c => return Err(format!("Unexpected character '{}' at line {}, col {}", c, line, col)),
        };

        Ok(self.spanned(tok, line, col))
    }

    fn read_ident(&mut self) -> String {
        let mut s = String::new();
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                s.push(c);
                self.advance();
            } else {
                break;
            }
        }
        s
    }

    fn read_number(&mut self, line: usize, col: usize) -> Result<SpannedToken, String> {
        let mut s = String::new();
        let mut is_float = false;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                s.push(c);
                self.advance();
            } else if c == '.' && self.peek2().map(|x| x.is_ascii_digit()).unwrap_or(false) {
                is_float = true;
                s.push(c);
                self.advance();
            } else {
                break;
            }
        }
        if is_float {
            let f: f64 = s.parse().map_err(|e| format!("Invalid float '{}': {}", s, e))?;
            Ok(self.spanned(Token::FloatLit(f), line, col))
        } else {
            let n: i64 = s.parse().map_err(|e| format!("Invalid integer '{}': {}", s, e))?;
            Ok(self.spanned(Token::IntLit(n), line, col))
        }
    }
}
