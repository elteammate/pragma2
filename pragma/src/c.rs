use crate::smol_str2::SmolStr2;

#[derive(Debug)]
pub struct Struct {
    pub fields: Vec<CType>,
}

#[derive(Debug)]
pub struct Module {
    pub includes: Vec<SmolStr2>,
    pub structs: Vec<Struct>,
    pub functions: Vec<Function>,
    pub externals: Vec<ExternalFunction>,
    pub main: Option<usize>,
}

#[derive(Debug)]
pub struct ExternalFunction {
    pub name: SmolStr2,
}

#[derive(Debug)]
pub struct Function {
    // TODO: rename parameters to arguments everywhere
    pub parameters: Vec<usize>,
    pub body: Vec<Statement>,
    pub locals: Vec<CType>,
    pub return_type: CType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CType {
    Int,
    Char,
    Void,
    Struct(usize),
    Pointer(Box<CType>),
    Function(Vec<CType>, Box<CType>),
}

impl CType {
    pub fn is_sized(&self) -> bool {
        match self {
            CType::Int => true,
            CType::Char => true,
            CType::Void => false,
            CType::Struct(_) => true,
            CType::Pointer(_) => true,
            CType::Function(_, _) => false,
        }
    }
}

#[derive(Debug)]
pub enum Statement {
    Expression(Expr),
    Return(Expr),
    ReturnVoid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Precedence {
    Lowest,
    SuffixUnary,
    PrefixUnary,
    Multiply,
    Add,
    Shift,
    Compare,
    Equality,
    BitAnd,
    BitXor,
    BitOr,
    And,
    Or,
    Ternary,
    Assign,
    Comma,
    Highest,
}

#[derive(Debug)]
pub enum Expr {
    Int { x: i64 },
    String { s: String },
    Local { id: usize },
    Global { id: usize },
    External { id: usize },
    Call { f: Box<Expr>, args: Vec<Expr> },
    Assign { lhs: usize, rhs: Box<Expr> },
    Plus { lhs: Box<Expr>, rhs: Box<Expr> },
    Minus { lhs: Box<Expr>, rhs: Box<Expr> },
    Multiply { lhs: Box<Expr>, rhs: Box<Expr> },
    UnaryPlus { x: Box<Expr> },
    UnaryMinus { x: Box<Expr> },
    Ref { x: Box<Expr> },
    Deref { x: Box<Expr> },
    Cast { x: Box<Expr>, ty: CType },
    StructAccess { x: Box<Expr>, field: usize },
    StructBuild { id: usize, fields: Vec<Expr> },
}

impl Expr {
    pub fn prec(&self) -> Precedence {
        match self {
            Expr::Int { .. } => Precedence::Lowest,
            Expr::String { .. } => Precedence::Lowest,
            Expr::Local { .. } => Precedence::Lowest,
            Expr::Global { .. } => Precedence::Lowest,
            Expr::External { .. } => Precedence::Lowest,
            Expr::Call { .. } => Precedence::SuffixUnary,
            Expr::Assign { .. } => Precedence::Assign,
            Expr::Plus { .. } => Precedence::Add,
            Expr::Minus { .. } => Precedence::Add,
            Expr::Multiply { .. } => Precedence::Multiply,
            Expr::UnaryPlus { .. } => Precedence::PrefixUnary,
            Expr::UnaryMinus { .. } => Precedence::PrefixUnary,
            Expr::Ref { .. } => Precedence::PrefixUnary,
            Expr::Deref { .. } => Precedence::PrefixUnary,
            Expr::Cast { .. } => Precedence::PrefixUnary,
            Expr::StructAccess { .. } => Precedence::SuffixUnary,
            Expr::StructBuild { .. } => Precedence::Lowest,
        }
    }
}
