/// Hardware target for memory and compute
#[derive(Debug, Clone, PartialEq)]
pub enum HwTarget {
    Cpu,
    Gpu,
}

/// Resolved type in the type system
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int64,
    Float64,
    Bool,
    Str,
    Tensor(Box<HwTarget>),
    Array(Box<Type>),          // [T]
    Struct(String),            // named struct
    Void,
    Unknown,
}

/// Type annotation as written in source (before full resolution)
#[derive(Debug, Clone)]
pub enum TypeAnnotation {
    Int64,
    Float64,
    Bool,
    Str,
    Tensor(Option<HwTarget>),
    Array(Box<TypeAnnotation>),
    Named(String),             // struct type by name
    Void,
}

impl TypeAnnotation {
    pub fn to_type(&self) -> Type {
        match self {
            TypeAnnotation::Int64 => Type::Int64,
            TypeAnnotation::Float64 => Type::Float64,
            TypeAnnotation::Bool => Type::Bool,
            TypeAnnotation::Str => Type::Str,
            TypeAnnotation::Tensor(Some(hw)) => Type::Tensor(Box::new(hw.clone())),
            TypeAnnotation::Tensor(None) => Type::Tensor(Box::new(HwTarget::Cpu)),
            TypeAnnotation::Array(inner) => Type::Array(Box::new(inner.to_type())),
            TypeAnnotation::Named(n) => Type::Struct(n.clone()),
            TypeAnnotation::Void => Type::Void,
        }
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Int64 => write!(f, "int64"),
            Type::Float64 => write!(f, "float64"),
            Type::Bool => write!(f, "bool"),
            Type::Str => write!(f, "string"),
            Type::Tensor(hw) => match hw.as_ref() {
                HwTarget::Cpu => write!(f, "tensor@cpu"),
                HwTarget::Gpu => write!(f, "tensor@gpu"),
            },
            Type::Array(inner) => write!(f, "[{}]", inner),
            Type::Struct(name) => write!(f, "{}", name),
            Type::Void => write!(f, "void"),
            Type::Unknown => write!(f, "?"),
        }
    }
}

/// Binary operators
#[derive(Debug, Clone, PartialEq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod,
    EqEq, NotEq, Lt, Gt, LtEq, GtEq,
    And, Or,
}

/// Unary operators
#[derive(Debug, Clone, PartialEq)]
pub enum UnOp {
    Neg,
    Not,
}

/// A struct field in a struct definition
#[derive(Debug, Clone)]
pub struct FieldDef {
    pub name: String,
    pub ty: TypeAnnotation,
}

/// Expression nodes
#[derive(Debug, Clone)]
pub enum Expr {
    IntLit(i64),
    FloatLit(f64),
    StringLit(String),
    BoolLit(bool),
    Ident(String),
    BinOp {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    UnOp {
        op: UnOp,
        expr: Box<Expr>,
    },
    Call {
        callee: String,
        args: Vec<Expr>,
    },
    MethodCall {
        obj: Box<Expr>,
        method: String,
        args: Vec<Expr>,
    },
    // struct Foo { x: 1, y: 2 }
    StructLit {
        name: String,
        fields: Vec<(String, Expr)>,
    },
    // expr.field
    FieldAccess {
        obj: Box<Expr>,
        field: String,
    },
    // arr[idx]
    Index {
        obj: Box<Expr>,
        idx: Box<Expr>,
    },
    // [a, b, c]
    ArrayLit(Vec<Expr>),
    // Memory migration
    ToGpu(Box<Expr>),
    ToCpu(Box<Expr>),
}

/// A function parameter
#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: TypeAnnotation,
}

/// Statement nodes
#[derive(Debug, Clone)]
pub enum Stmt {
    Let {
        mutable: bool,
        name: String,
        ty: Option<TypeAnnotation>,
        init: Expr,
    },
    Assign {
        target: AssignTarget,
        value: Expr,
    },
    ExprStmt(Expr),
    Return(Option<Expr>),
    If {
        cond: Expr,
        then_block: Block,
        else_block: Option<Block>,
    },
    While {
        cond: Expr,
        body: Block,
    },
    For {
        var: String,
        start: Expr,
        end: Expr,
        body: Block,
    },
    GpuBlock(Block),
    Import(Vec<String>),
    Break,
    Continue,
}

/// Assignment targets (x = ..., x.field = ..., x[i] = ...)
#[derive(Debug, Clone)]
pub enum AssignTarget {
    Ident(String),
    Field(Box<Expr>, String),
    Index(Box<Expr>, Box<Expr>),
}

pub type Block = Vec<Stmt>;

/// Struct definition
#[derive(Debug, Clone)]
pub struct StructDef {
    pub name: String,
    pub fields: Vec<FieldDef>,
}

/// Top-level function definition
#[derive(Debug, Clone)]
pub struct FnDef {
    pub name: String,
    pub params: Vec<Param>,
    pub ret_ty: TypeAnnotation,
    pub body: Block,
    pub line: usize,
}

/// Root of the AST
#[derive(Debug, Clone)]
pub struct Program {
    pub imports: Vec<Vec<String>>,
    pub structs: Vec<StructDef>,
    pub fns: Vec<FnDef>,
    pub top_level: Block,
}
