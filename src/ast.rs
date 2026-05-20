/// Hardware placement annotation on any declaration
#[derive(Debug, Clone, PartialEq)]
pub enum HwAnnotation {
    Cpu,
    Gpu,
    CpuFrac(f64),   // @cpu(0.8) — pin fraction of CPU capacity to this item
    GpuFrac(f64),   // @gpu(0.5) — pin fraction of GPU capacity to this item
}

impl HwAnnotation {
    pub fn is_gpu(&self) -> bool {
        matches!(self, HwAnnotation::Gpu | HwAnnotation::GpuFrac(_))
    }
    pub fn is_cpu(&self) -> bool {
        matches!(self, HwAnnotation::Cpu | HwAnnotation::CpuFrac(_))
    }
    pub fn frac(&self) -> f64 {
        match self {
            HwAnnotation::GpuFrac(f) | HwAnnotation::CpuFrac(f) => *f,
            _ => 1.0,
        }
    }
    pub fn ir_comment(&self) -> String {
        match self {
            HwAnnotation::Cpu => "; placement=cpu".to_string(),
            HwAnnotation::Gpu => "; placement=gpu".to_string(),
            HwAnnotation::CpuFrac(f) => format!("; placement=cpu cores={:.3}", f),
            HwAnnotation::GpuFrac(f) => format!("; placement=gpu alloc={:.3}", f),
        }
    }
}

/// Hardware target for tensor/model memory
#[derive(Debug, Clone, PartialEq)]
pub enum HwTarget {
    Cpu,
    Gpu,
}

/// ML model format (for load_model hint — runtime auto-detects by extension too)
#[derive(Debug, Clone, PartialEq)]
pub enum ModelFormat {
    Auto,       // detect from extension
    Pt,         // PyTorch .pt / .pth
    Onnx,       // ONNX .onnx
    H5,         // Keras HDF5 .h5
    MlModel,    // Apple CoreML .mlmodel
}

/// Resolved type in the type system
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int64,
    Float64,
    Bool,
    Str,
    Tensor(Box<HwTarget>),
    Model(Box<HwTarget>),   // ML model handle, placed on cpu or gpu
    Array(Box<Type>),
    Struct(String),
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
    Model(Option<HwTarget>),
    Array(Box<TypeAnnotation>),
    Named(String),
    Void,
}

impl TypeAnnotation {
    pub fn to_type(&self) -> Type {
        match self {
            TypeAnnotation::Int64   => Type::Int64,
            TypeAnnotation::Float64 => Type::Float64,
            TypeAnnotation::Bool    => Type::Bool,
            TypeAnnotation::Str     => Type::Str,
            TypeAnnotation::Tensor(Some(hw)) => Type::Tensor(Box::new(hw.clone())),
            TypeAnnotation::Tensor(None)     => Type::Tensor(Box::new(HwTarget::Cpu)),
            TypeAnnotation::Model(Some(hw))  => Type::Model(Box::new(hw.clone())),
            TypeAnnotation::Model(None)      => Type::Model(Box::new(HwTarget::Cpu)),
            TypeAnnotation::Array(inner)     => Type::Array(Box::new(inner.to_type())),
            TypeAnnotation::Named(n)         => Type::Struct(n.clone()),
            TypeAnnotation::Void             => Type::Void,
        }
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Int64   => write!(f, "int64"),
            Type::Float64 => write!(f, "float64"),
            Type::Bool    => write!(f, "bool"),
            Type::Str     => write!(f, "string"),
            Type::Tensor(hw) => match hw.as_ref() {
                HwTarget::Cpu => write!(f, "tensor@cpu"),
                HwTarget::Gpu => write!(f, "tensor@gpu"),
            },
            Type::Model(hw) => match hw.as_ref() {
                HwTarget::Cpu => write!(f, "model@cpu"),
                HwTarget::Gpu => write!(f, "model@gpu"),
            },
            Type::Array(inner) => write!(f, "[{}]", inner),
            Type::Struct(name) => write!(f, "{}", name),
            Type::Void    => write!(f, "void"),
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
    StructLit {
        name: String,
        fields: Vec<(String, Expr)>,
    },
    FieldAccess {
        obj: Box<Expr>,
        field: String,
    },
    Index {
        obj: Box<Expr>,
        idx: Box<Expr>,
    },
    ArrayLit(Vec<Expr>),
    ToGpu(Box<Expr>),
    ToCpu(Box<Expr>),
}

/// A function parameter (optionally annotated with placement)
#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: TypeAnnotation,
    pub hw: Option<HwAnnotation>,
}

/// Statement nodes
#[derive(Debug, Clone)]
pub enum Stmt {
    Let {
        mutable: bool,
        name: String,
        ty: Option<TypeAnnotation>,
        init: Expr,
        hw: Option<HwAnnotation>,   // @gpu/@cpu on variable declaration
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

/// Assignment targets
#[derive(Debug, Clone)]
pub enum AssignTarget {
    Ident(String),
    Field(Box<Expr>, String),
    Index(Box<Expr>, Box<Expr>),
}

pub type Block = Vec<Stmt>;

/// Struct definition (optionally annotated with placement)
#[derive(Debug, Clone)]
pub struct StructDef {
    pub name: String,
    pub fields: Vec<FieldDef>,
    pub hw: Option<HwAnnotation>,
}

/// Top-level function definition (optionally annotated with placement)
#[derive(Debug, Clone)]
pub struct FnDef {
    pub name: String,
    pub params: Vec<Param>,
    pub ret_ty: TypeAnnotation,
    pub body: Block,
    pub line: usize,
    pub hw: Option<HwAnnotation>,
}

/// Root of the AST
#[derive(Debug, Clone)]
pub struct Program {
    pub imports: Vec<Vec<String>>,
    pub structs: Vec<StructDef>,
    pub fns: Vec<FnDef>,
    pub top_level: Block,
}
