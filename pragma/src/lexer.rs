use std::iter::Peekable;
use logos::{Logos, SpannedIter};
use crate::smol_str2::SmolStr2;

#[derive(logos::Logos, Debug, PartialEq)]
#[logos(skip r"\s+|//[^\n]*\n")]
pub enum Token {
    #[token(";")]
    Semi,
    #[token(",")]
    Comma,
    #[token(":")]
    Colon,
    #[token("->")]
    Arrow,
    #[token("=")]
    Eq,
    #[token("fn")]
    Fn,
    #[token("Fn")]
    FnTy,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("while")]
    While,
    #[token("struct")]
    Struct,
    #[token("type")]
    Type,
    #[token("---")]
    Uninit,
    #[token("_", priority = 3)]
    Hole,
    #[token("return")]
    Return,
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("|")]
    Pipe,
    #[token("^")]
    Caret,
    #[token("!")]
    Bang,
    #[token("==")]
    EqEq,
    #[token("!=")]
    NotEq,
    #[token("<")]
    Lt,
    #[token("<=")]
    Le,
    #[token(">")]
    Gt,
    #[token(">=")]
    Ge,
    #[token(".")]
    Dot,
    #[token("&")]
    Ampersand,
    #[regex(r"\d+", |lex| {
        lex.slice().parse::<u128>().map(Some).unwrap_or(None)
    })]
    Number(Option<u128>),
    #[regex(r#""(\\"|\\n|\\t|\\r|[^"\n])*""#, |lex| {
        let s = lex.slice();
        SmolStr2::from(&s[1..s.len()-1])
    })]
    String(SmolStr2),
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| {
        SmolStr2::from(lex.slice())
    })]
    Ident(SmolStr2),
}

pub type Lex<'s> = Peekable<SpannedIter<'s, Token>>;

pub fn lex(program: &str) -> Lex {
    Token::lexer(program).spanned().peekable()
}
