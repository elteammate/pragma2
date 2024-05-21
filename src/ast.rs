use smol_str::SmolStr;
use crate::span::Span;

pub struct Ast<T> {
    pub span: Span,
    pub node: T,
}

pub trait NodeExt: Sized {
    fn spanned(self, span: Span) -> Ast<Self>;
}

impl<T> NodeExt for T {
    fn spanned(self, span: Span) -> Ast<Self> {
        Ast { span, node: self }
    }
}

pub struct PunctComma;
pub struct PunctSemi;
pub struct PunctEq;
pub struct PunctColon;
pub struct PunctArrow;
pub struct PunctLBrace;
pub struct PunctRBrace;
pub struct PunctLParen;
pub struct PunctRParen;
pub struct PunctLBracket;
pub struct PunctRBracket;
pub struct KwStruct;
pub struct KwType;
pub struct KwFnTy;
pub struct KwFn;
pub struct KwIf;
pub struct KwWhile;
pub struct KwReturn;


pub struct Module {
    pub items: Vec<Ast<Item>>,
}

pub struct Item {
    pub ident: Ast<SmolStr>,
    pub params: Vec<(Ast<Param>, Option<Ast<PunctComma>>)>,
    pub arrow: Option<Ast<PunctArrow>>,
    pub ret_ty: Option<Ast<Expr>>,
    pub eq: Option<Ast<PunctEq>>,
    pub body: Option<Ast<Expr>>,
    pub semi: Ast<PunctSemi>,
}

pub struct Param {
    pub ident: Option<Ast<SmolStr>>,
    pub colon: Ast<PunctColon>,
    pub ty: Ast<Expr>,
}

pub enum BinaryOp {
    Add,
    Sub,
    Mul,
}

pub enum UnaryOp {
    Neg,
    Pos,
    Ref,
    Deref,
}

pub enum Expr {
    Ident(SmolStr),
    Number(u128),
    String(SmolStr),
    Bool(bool),
    Hole,
    Uninit,
    StructDecl {
        kw_struct: Ast<KwStruct>,
        lbrace: Ast<PunctLBrace>,
        fields: Vec<(Ast<SmolStr>, Ast<PunctColon>, Ast<Expr>, Option<Ast<PunctComma>>)>,
        rbrace: Ast<PunctRBrace>,
    },
    StructInit {
        struct_: Box<Ast<Expr>>,
        lbrace: Ast<PunctLBrace>,
        fields: Vec<(Ast<SmolStr>, Ast<PunctColon>, Ast<Expr>, Option<Ast<PunctComma>>)>,
        rbrace: Ast<PunctRBrace>,
    },
    FnDecl {
        kw_fn: Ast<KwFn>,
        lparen: Ast<PunctLParen>,
        args: Vec<(Ast<SmolStr>, Ast<Expr>, Option<Ast<PunctComma>>)>,
        rparen: Ast<PunctRParen>,
        ret_ty: Option<(Ast<PunctColon>, Box<Ast<Expr>>)>,
        body: Box<Ast<Expr>>,
    },
    FnType {
        kw_fn: Ast<KwFnTy>,
        lparen: Ast<PunctLParen>,
        args: Vec<Ast<Expr>>,
        rparen: Ast<PunctRParen>,
        arrow: Ast<PunctArrow>,
        ret: Box<Ast<Expr>>,
    },
    Type {
        kw_type: Ast<KwType>,
        lbracket: Ast<PunctLBracket>,
        indicators: Vec<(Ast<Expr>, Option<Ast<PunctComma>>)>,
        rbracket: Ast<PunctRBracket>,
    },
    Block {
        lbrace: Ast<PunctLBrace>,
        stmts: Vec<(Ast<Expr>, Ast<PunctSemi>)>,
        ret: Option<Box<Ast<Expr>>>,
        rbrace: Ast<PunctRBrace>,
    },
    If {
        kw_if: Ast<KwIf>,
        lparen: Ast<PunctLParen>,
        cond: Box<Ast<Expr>>,
        rparen: Ast<PunctRParen>,
        then: Box<Ast<Expr>>,
        else_: Option<Box<Ast<Expr>>>,
    },
    While {
        kw_while: Ast<KwWhile>,
        lparen: Ast<PunctLParen>,
        cond: Box<Ast<Expr>>,
        rparen: Ast<PunctRParen>,
        body: Box<Ast<Expr>>,
    },
    Decl {
        ident: Ast<SmolStr>,
        colon: Ast<PunctColon>,
        ty: Option<Box<Ast<Expr>>>,
        eq: Ast<PunctEq>,
        val: Box<Ast<Expr>>,
    },
    Assign {
        lvalue: Box<Ast<Expr>>,
        eq: Ast<PunctEq>,
        value: Box<Ast<Expr>>,
    },
    Call {
        callee: Box<Ast<Expr>>,
        lparen: Ast<PunctLParen>,
        args: Vec<(Ast<Expr>, Option<Ast<PunctComma>>)>,
        rparen: Ast<PunctRParen>,
    },
    App {
        callee: Box<Ast<Expr>>,
        lbracket: Ast<PunctLBracket>,
        params: Vec<(Ast<Expr>, Option<Ast<PunctComma>>)>,
        rbracket: Ast<PunctRBracket>,
    },
    Binary {
        op: Ast<BinaryOp>,
        lhs: Box<Ast<Expr>>,
        rhs: Box<Ast<Expr>>,
    },
    Unary {
        op: Ast<UnaryOp>,
        expr: Box<Ast<Expr>>,
    },
    Return {
        kw_return: Ast<KwReturn>,
        expr: Option<Box<Ast<Expr>>>,
    },
    Paren {
        lparen: Ast<PunctLParen>,
        expr: Box<Ast<Expr>>,
        rparen: Ast<PunctRParen>,
    },
}
