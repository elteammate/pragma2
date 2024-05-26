use serde::Serialize;
use crate::smol_str2::SmolStr2;
use crate::span::Span;

#[derive(Debug, Serialize)]
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

#[derive(Debug, Serialize)]
pub struct PunctComma;

#[derive(Debug, Serialize)]
pub struct PunctSemi;

#[derive(Debug, Serialize)]
pub struct PunctEq;

#[derive(Debug, Serialize)]
pub struct PunctColon;

#[derive(Debug, Serialize)]
pub struct PunctArrow;

#[derive(Debug, Serialize)]
pub struct PunctLBrace;

#[derive(Debug, Serialize)]
pub struct PunctRBrace;

#[derive(Debug, Serialize)]
pub struct PunctLParen;

#[derive(Debug, Serialize)]
pub struct PunctRParen;

#[derive(Debug, Serialize)]
pub struct PunctLBracket;

#[derive(Debug, Serialize)]
pub struct PunctRBracket;

#[derive(Debug, Serialize)]
pub struct KwStruct;

#[derive(Debug, Serialize)]
pub struct KwType;

#[derive(Debug, Serialize)]
pub struct KwFnTy;

#[derive(Debug, Serialize)]
pub struct KwFn;

#[derive(Debug, Serialize)]
pub struct KwIf;

#[derive(Debug, Serialize)]
pub struct KwWhile;

#[derive(Debug, Serialize)]
pub struct KwReturn;


#[derive(Debug, Serialize)]
pub struct Module {
    pub items: Vec<Ast<Item>>,
}

#[derive(Debug, Serialize)]
pub struct Item {
    pub ident: Ast<SmolStr2>,
    pub params: Vec<(Ast<Param>, Option<Ast<PunctComma>>)>,
    pub arrow: Option<Ast<PunctArrow>>,
    pub ret_ty: Option<Ast<Expr>>,
    pub eq: Option<Ast<PunctEq>>,
    pub body: Option<Ast<Expr>>,
    pub semi: Ast<PunctSemi>,
}

#[derive(Debug, Serialize)]
pub enum Param {
    Generic {
        ident: Option<Ast<SmolStr2>>,
        colon: Ast<PunctColon>,
        ty: Ast<Expr>,
    },
    Value {
        lbracket: Ast<PunctLBracket>,
        value: Ast<Expr>,
        rbracket: Ast<PunctRBracket>,
    },    
}

#[derive(Debug, Serialize, Copy, Clone)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
}

#[derive(Debug, Serialize, Copy, Clone)]
pub enum UnaryOp {
    Neg,
    Pos,
    Ref,
    Deref,
}

#[derive(Debug, Serialize)]
pub enum Expr {
    Ident(SmolStr2),
    Int(u128),
    String(SmolStr2),
    Bool(bool),
    Hole,
    Unit,
    Uninit,
    StructDecl {
        kw_struct: Ast<KwStruct>,
        lbrace: Ast<PunctLBrace>,
        fields: Vec<(Ast<SmolStr2>, Ast<PunctColon>, Ast<Expr>, Option<Ast<PunctComma>>)>,
        rbrace: Ast<PunctRBrace>,
    },
    StructInit {
        struct_: Box<Ast<Expr>>,
        lbrace: Ast<PunctLBrace>,
        fields: Vec<(Ast<SmolStr2>, Ast<PunctColon>, Ast<Expr>, Option<Ast<PunctComma>>)>,
        rbrace: Ast<PunctRBrace>,
    },
    FnDecl {
        kw_fn: Ast<KwFn>,
        lparen: Ast<PunctLParen>,
        args: Vec<(Ast<SmolStr2>, Ast<PunctColon>, Ast<Expr>, Option<Ast<PunctComma>>)>,
        rparen: Ast<PunctRParen>,
        ret_ty: Option<(Ast<PunctArrow>, Box<Ast<Expr>>)>,
        body: Box<Ast<Expr>>,
    },
    FnType {
        kw_fn: Ast<KwFnTy>,
        lparen: Ast<PunctLParen>,
        args: Vec<(Ast<Expr>, Option<Ast<PunctComma>>)>,
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
    PlainType {
        kw_type: Ast<KwType>,   
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
        ident: Ast<SmolStr2>,
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
