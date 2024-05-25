use bitflags::bitflags;
use serde::Serialize;
use crate::ast::*;
use crate::lexer::{lex, Lex, Token};
use crate::smol_str2::SmolStr2;
use crate::span::Span;
use crate::compound_result::{CompoundResult, ResultExt};
use crate::{vmrg, cmrg};

#[derive(Serialize, Debug)]
pub enum ParseError {
    UnexpectedToken(Span),
    UnexpectedEof(String),
    Spanned(Span, String),
}

type Result<T> = CompoundResult<T, ParseError>;

trait ParseResultExt<T> {
    fn recover(self, lex: &mut Lex, target: RecoveryTarget) -> Result<T>;

}

impl<T> ParseResultExt<T> for Result<T> {
    fn recover(self, lex: &mut Lex, target: RecoveryTarget) -> Result<T> {
        match self {
            Ok(x) => Ok(x),
            Err(mut errors) => {
                match try_recover(lex, target) {
                    Ok(()) => Err(errors),
                    Err(mut new_errors) => {
                        errors.append(&mut new_errors);
                        Err(errors)
                    }
                }
            }
        }
    }
}

trait OkAndSome {
    fn ok_and_some(&self) -> bool;
    fn ok_and_none(&self) -> bool;
    fn err_or_none(&self) -> bool;
}

impl<T> OkAndSome for Result<Option<T>> {
    fn ok_and_some(&self) -> bool {
        matches!(self, Ok(Some(_)))
    }

    fn ok_and_none(&self) -> bool {
        matches!(self, Ok(None))
    }

    fn err_or_none(&self) -> bool {
        self.is_err() || self.ok_and_none()
    }
}

pub fn parse(input: &str) -> Result<Module> {
    let mut lex = lex(input);

    let mut items = Vec::new();

    while lex.peek().is_some() {
        items.push(parse_item(&mut lex))
    }

    let items = vmrg!(items)?;
    Ok(Module { items })
}

fn parse_item(lex: &mut Lex) -> Result<Ast<Item>> {
    let mut ident = parse_ident(lex);

    let mut params = Vec::new();
    while let Some((t, span)) = lex.peek() {
        match t {
            Ok(Token::Colon | Token::Ident(_) | Token::LBracket) => {
                let param = parse_item_param(lex)
                    .recover(lex, RecoveryTarget::Eq | RecoveryTarget::Semi | RecoveryTarget::Arrow);
                let comma = maybe_parse_comma(lex);
                let stop = comma.ok_and_none();
                params.push(cmrg!(param, comma));
                if stop { break }
            }
            Ok(Token::Arrow | Token::Eq) => break,
            Ok(_) => {
                let span = span.into();
                lex.next();
                params.push(Err(vec![ParseError::Spanned(
                    span,
                    "Expected `:`, identifier, or continuation of the declaration".to_string()
                )]))
            },
            Err(_) => params.push(Err(vec![ParseError::UnexpectedToken(span.into())])),
        }
    }

    let params = vmrg!(params).recover(lex, RecoveryTarget::Arrow);

    let (arrow, ret_ty) = if matches!(lex.peek(), Some((Ok(Token::Arrow), _))) {
        let arrow = parse_arrow(lex).recover(lex, RecoveryTarget::Semi | RecoveryTarget::Eq);
        let ret_ty = parse_expr(lex, ParseUntil::Eq).recover(lex, RecoveryTarget::Semi | RecoveryTarget::Eq);
        (Some(arrow).transpose(), Some(ret_ty).transpose())
    } else {
        (Ok(None), Ok(None))
    };

    let (eq, body) = if matches!(lex.peek(), Some((Ok(Token::Eq), _))) {
        let eq = parse_eq(lex).recover(lex, RecoveryTarget::Semi);
        let body = parse_expr(lex, ParseUntil::Termination).recover(lex, RecoveryTarget::Semi);
        (Some(eq).transpose(), Some(body).transpose())
    } else {
        (Ok(None), Ok(None))
    };

    let semi = parse_semi(lex);

    if let (Ok(i), Ok(None), Ok(None)) = (&mut ident, &ret_ty, &body) {
        Result::<()>::Err(vec![ParseError::Spanned(
            i.span,
            "Expected a return type or a body".to_string()
        )]).delegate(&mut ident);
    }

    let (ident, params, arrow, ret_ty, eq, body, semi) =
        cmrg!(ident, params, arrow, ret_ty, eq, body, semi)?;

    let span = ident.span.merge(semi.span);
    Ok(Item {
        ident,
        params,
        arrow,
        ret_ty,
        eq,
        body,
        semi,
    }.spanned(span))
}

fn parse_item_param(lex: &mut Lex) -> Result<Ast<Param>> {
    match lex.peek() {
        Some((Ok(Token::Colon), _)) => {
            let colon = parse_colon(lex);
            let ty = parse_expr(lex, ParseUntil::Eq | ParseUntil::Arrow);
            let (colon, ty) = cmrg!(colon, ty)
                .recover(lex, RecoveryTarget::Comma | RecoveryTarget::Eq | RecoveryTarget::Arrow | RecoveryTarget::Semi)?;
            let span = colon.span.merge(ty.span);
            Ok(Param::Generic {
                ident: None,
                colon,
                ty,
            }.spanned(span))
        }
        Some((Ok(Token::Ident(_)), _)) => {
            let ident = parse_ident(lex);
            let colon = parse_colon(lex);
            let ty = parse_expr(lex, ParseUntil::Eq | ParseUntil::Arrow);
            let (ident, colon, ty) = cmrg!(ident, colon, ty).recover(lex, RecoveryTarget::Comma)?;
            let span = ident.span.merge(ty.span);
            Ok(Param::Generic {
                ident: Some(ident),
                colon,
                ty,
            }.spanned(span))
        }
        Some((Ok(Token::LBracket), _)) => {
            let lbracket = parse_lbracket(lex);
            let value = parse_expr(lex, ParseUntil::Eq | ParseUntil::Arrow)
                .recover(lex, RecoveryTarget::Comma | RecoveryTarget::Eq | RecoveryTarget::Arrow | RecoveryTarget::Semi | RecoveryTarget::RBracket);
            let rbracket = parse_rbracket(lex);
            let (lbracket, value, rbracket) = cmrg!(lbracket, value, rbracket)?;
            let span = lbracket.span.merge(rbracket.span);
            Ok(Param::Value {
                lbracket,
                value,
                rbracket
            }.spanned(span))
        }
        Some((Ok(_), span)) => 
            Err(vec![ParseError::Spanned(span.into(), "Expected identifier or comma".to_string())]),
        Some((Err(_), span)) => 
            Err(vec![ParseError::UnexpectedToken(span.into())]),
        None => 
            Err(vec![ParseError::UnexpectedEof("Expected a parameter declaration".to_string())]),
    }
}

bitflags! {
    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    struct ParseUntil : u32 {
        const Termination = !0;
        const Eq = 1 << 0;
        const Arrow = 1 << 1;
        const Colon = 1 << 2;
    }
}

#[derive(Ord, PartialOrd, Eq, PartialEq)]
enum Precedence {
    Lowest,
    Assign,
    AddSub,
    MulDiv,
    Unary,
}

fn parse_expr(lex: &mut Lex, until: ParseUntil) -> Result<Ast<Expr>> {
    let end = RecoveryTarget::Semi | RecoveryTarget::RBrace | RecoveryTarget::RParen;
    parse_expr_impl(lex, until)
        .recover(
            lex,
            end |
                if until.contains(ParseUntil::Eq) { RecoveryTarget::Eq } else { RecoveryTarget::empty() } |
                if until.contains(ParseUntil::Arrow) { RecoveryTarget::Arrow } else { RecoveryTarget::empty() } |
                if until.contains(ParseUntil::Colon) { RecoveryTarget::Colon } else { RecoveryTarget::empty() }
        )
}

fn parse_expr_impl(lex: &mut Lex, until: ParseUntil) -> Result<Ast<Expr>> {
    let lhs = parse_primary(lex, until)?;
    
    if matches!(until, ParseUntil::Termination) {
        match lex.peek() {
            Some((Ok(Token::Colon), _)) => {
                let ident = extract_pattern(lhs);
                let colon = parse_colon(lex);
                let ty = if let Some((Ok(Token::Eq), _)) = lex.peek() {
                    Some(parse_expr(lex, ParseUntil::Eq)).transpose()
                } else {
                    Ok(None)
                };
                let eq = parse_eq(lex);
                let rhs = parse_expr(lex, ParseUntil::Termination);
                let (ident, colon, ty, eq, rhs) = cmrg!(ident, colon, ty, eq, rhs)
                    .recover(lex, RecoveryTarget::Semi)?;
                let span = ident.span.merge(rhs.span);
                return Ok(Expr::Decl {
                    ident,
                    colon,
                    ty: ty.map(Box::new),
                    eq,
                    val: Box::new(rhs),
                }.spanned(span));
            }
            Some((Ok(Token::Eq), _)) => {
                let eq = parse_eq(lex);
                let rhs = parse_expr(lex, ParseUntil::Termination);
                let (eq, rhs) = cmrg!(eq, rhs).recover(lex, RecoveryTarget::Semi)?;
                let span = lhs.span.merge(rhs.span);
                return Ok(Expr::Assign {
                    lvalue: Box::new(lhs),
                    eq,
                    value: Box::new(rhs),
                }.spanned(span));
            }
            _ => {}
        }
    }

    let mut stack = vec![(lhs, Precedence::Lowest, None)];

    loop {
        let op = match lex.peek() {
            Some((Ok(Token::Plus), span)) => BinaryOp::Add.spanned(span.into()),
            Some((Ok(Token::Minus), span)) => BinaryOp::Sub.spanned(span.into()),
            Some((Ok(Token::Star), span)) => BinaryOp::Mul.spanned(span.into()),
            _ => break,
        };
        lex.next();
        let prec = match op.node {
            BinaryOp::Add | BinaryOp::Sub => Precedence::AddSub,
            BinaryOp::Mul => Precedence::MulDiv,
        };

        let mut operand = parse_primary(lex, until)?;

        let mut op = Some(op);
        while let Some((_, prec2, _)) = stack.last() {
            if *prec2 < prec {
                break;
            }

            let (rhs, _, op2) = stack.pop().unwrap();
            let (lhs, p, prev_op) = stack.pop().unwrap();
            let span = lhs.span.merge(rhs.span);
            stack.push((
                Expr::Binary {
                    op: op2.unwrap(),
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                }.spanned(span),
                p,
                prev_op,
            ));
        }

        stack.push((operand, prec, op));
    }

    while let Some((rhs, _, op)) = stack.pop() {
        let Some((lhs, _, prev_op)) = stack.pop() else {
            return Ok(rhs);
        };
        let span = lhs.span.merge(rhs.span);
        stack.push((
            Expr::Binary {
                op: op.unwrap(),
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            }.spanned(span),
            Precedence::Lowest,
            prev_op,
        ));
    }

    unreachable!()
}

fn parse_primary(lex: &mut Lex, until: ParseUntil) -> Result<Ast<Expr>> {
    let mut expr = match lex.peek() {
        Some((Ok(Token::Ident(_)), _)) => {
            let ident = parse_ident(lex)?;
            let span = ident.span;
            Expr::Ident(ident.node).spanned(span)
        }
        Some((Ok(Token::Number(n)), span)) => {
            let n = n.ok_or(vec![ParseError::Spanned(span.into(), "Couldn't parse number".to_string())])?;
            let result = Expr::Int(n).spanned(span.into());
            lex.next();
            result
        }
        Some((Ok(Token::String(s)), span)) => {
            let result = Expr::String(s.clone()).spanned(span.into());
            lex.next();
            result
        }
        Some((Ok(Token::True), span)) => {
            let result = Expr::Bool(true).spanned(span.into());
            lex.next();
            result
        }
        Some((Ok(Token::False), span)) => {
            let result = Expr::Bool(false).spanned(span.into());
            lex.next();
            result
        }
        Some((Ok(Token::Hole), span)) => {
            let result = Expr::Hole.spanned(span.into());
            lex.next();
            result
        }
        Some((Ok(Token::Uninit), span)) => {
            let result = Expr::Uninit.spanned(span.into());
            lex.next();
            result
        }
        Some((Ok(Token::LParen), _)) => {
            let lparen = parse_lparen(lex);
            if matches!(lex.peek(), Some((Ok(Token::RParen), _)) | None) {
                let rparen = parse_rparen(lex);
                let (lparen, rparen) = cmrg!(lparen, rparen)?;
                let span = lparen.span.merge(rparen.span);
                Expr::Unit.spanned(span)
            } else {
                let expr = parse_expr(lex, ParseUntil::Termination).recover(lex, RecoveryTarget::RParen);
                let rparen = parse_rparen(lex);
                let (lparen, expr, rparen) = cmrg!(lparen, expr, rparen)?;
                let span = lparen.span.merge(rparen.span);
                Expr::Paren {
                    lparen,
                    expr: Box::new(expr),
                    rparen,
                }.spanned(span)
            }
        }
        Some((Ok(tok@(Token::Plus | Token::Minus | Token::Star | Token::Ampersand)), span)) => {
            let span = span.into();
            let op = match tok {
                Token::Plus => UnaryOp::Pos,
                Token::Minus => UnaryOp::Neg,
                Token::Star => UnaryOp::Deref,
                Token::Ampersand => UnaryOp::Ref,
                _ => unreachable!(),
            }.spanned(span);
            lex.next();
            let expr = parse_primary(lex, until)?;
            let span = op.span.merge(expr.span);
            Expr::Unary {
                op,
                expr: Box::new(expr),
            }.spanned(span)
        }

        Some((Ok(Token::LBrace), _)) => parse_block(lex)?,
        Some((Ok(Token::Struct), _)) => parse_struct_decl(lex)?,
        Some((Ok(Token::Fn), _)) => parse_fn_decl(lex, until)?,
        Some((Ok(Token::FnTy), _)) => parse_fn_type(lex, until)?,
        Some((Ok(Token::Type), _)) => parse_type(lex)?,
        Some((Ok(Token::If), _)) => parse_if(lex, until)?,
        Some((Ok(Token::While), _)) => parse_while(lex, until)?,
        Some((Ok(Token::Return), _)) => parse_return(lex, until)?,

        Some((Ok(_), span)) => return Err(vec![ParseError::Spanned(span.into(), "Expected expression".to_string())]),
        Some((Err(_), span)) => return Err(vec![ParseError::UnexpectedToken(span.into())]),
        None => return Err(vec![ParseError::UnexpectedEof("Expected expression".to_string())]),
    };

    loop {
        expr = match lex.peek() {
            Some((Ok(Token::LParen), _)) => {
                let (lparen, args, rparen) = parse_args(lex)?;
                let span = expr.span.merge(rparen.span);
                Expr::Call {
                    callee: Box::new(expr),
                    lparen,
                    args,
                    rparen,
                }.spanned(span)
            }

            Some((Ok(Token::LBracket), _)) => {
                let (lbracket, params, rbracket) = parse_params(lex)?;
                let span = expr.span.merge(rbracket.span);
                Expr::App {
                    callee: Box::new(expr),
                    lbracket,
                    params,
                    rbracket,
                }.spanned(span)
            }

            Some((Ok(Token::LBrace), _)) => {
                let lbrace = parse_lbrace(lex);
                let mut fields = Vec::new();
                let mut stop = false;
                while !matches!(lex.peek(), Some((Ok(Token::RBrace), _)) | None) && !stop {
                    let ident = parse_ident(lex);
                    let colon = parse_colon(lex);
                    let value = parse_expr(lex, ParseUntil::Termination);
                    let comma = maybe_parse_comma(lex);
                    stop = comma.ok_and_none();
                    let field = cmrg!(ident, colon, value, comma)
                        .recover(lex, RecoveryTarget::Comma | RecoveryTarget::RBrace | RecoveryTarget::Semi);

                    fields.push(field);
                }

                let fields = vmrg!(fields).recover(lex, RecoveryTarget::RBrace);
                let rbrace = parse_rbrace(lex);
                let (lbrace, fields, rbrace) = cmrg!(lbrace, fields, rbrace)
                    .recover(lex, RecoveryTarget::Semi | RecoveryTarget::RBrace)?;

                let span = expr.span.merge(rbrace.span);
                Expr::StructInit {
                    struct_: Box::new(expr),
                    lbrace,
                    fields,
                    rbrace,
                }.spanned(span)
            }

            _ => return Ok(expr),
        }
    }
}

fn extract_pattern(expr: Ast<Expr>) -> Result<Ast<SmolStr2>> {
    match expr.node {
        Expr::Ident(ident) => Ok(ident.spanned(expr.span)),
        _ => Err(vec![ParseError::Spanned(expr.span, "Expected identifier, not expression".to_string())]),
    }
}

fn parse_block(lex: &mut Lex) -> Result<Ast<Expr>> {
    let lbrace = parse_lbrace(lex);
    let mut stmts = Vec::new();
    let mut stop = false;
    while !matches!(lex.peek(), Some((Ok(Token::RBrace), _)) | None) && !stop {
        let stmt = parse_expr(lex, ParseUntil::Termination)
            .recover(lex, RecoveryTarget::Semi | RecoveryTarget::RBrace);
        let semi = maybe_parse_semi(lex);
        stop = semi.ok_and_none();
        stmts.push(cmrg!(stmt, semi));
        if stop { break }
    }
    let stmts = vmrg!(stmts).recover(lex, RecoveryTarget::RBrace);
    let rbrace = parse_rbrace(lex);
    let (lbrace, mut stmts, rbrace) = cmrg!(lbrace, stmts, rbrace)?;
    let ret = if !stop {
        stmts.pop().map(|(s, c)| {
            assert!(c.is_none());
            Box::new(s)
        })
    } else {
        None
    };

    let span = lbrace.span.merge(rbrace.span);

    Ok(Expr::Block {
        lbrace,
        stmts: stmts.into_iter().map(|(s, c)| (s, c.unwrap())).collect(),
        ret,
        rbrace,
    }.spanned(span))
}

fn parse_struct_decl(lex: &mut Lex) -> Result<Ast<Expr>> {
    let kw_struct = parse_struct_kw(lex);

    let lbrace = parse_lbrace(lex);
    let mut fields = Vec::new();
    let mut stop = false;
    // TODO: remove duplicate
    while !matches!(lex.peek(), Some((Ok(Token::RBrace), _)) | None) && !stop {
        let ident = parse_ident(lex);
        let colon = parse_colon(lex);
        let value = parse_expr(lex, ParseUntil::Termination);
        let comma = maybe_parse_comma(lex);
        stop = comma.ok_and_none();
        let field = cmrg!(ident, colon, value, comma)
            .recover(lex, RecoveryTarget::Comma | RecoveryTarget::RBrace | RecoveryTarget::Semi);

        fields.push(field);
    }

    let fields = vmrg!(fields).recover(lex, RecoveryTarget::RBrace);
    let rbrace = parse_rbrace(lex);
    let (kw_struct, lbrace, fields, rbrace) = cmrg!(kw_struct, lbrace, fields, rbrace)
        .recover(lex, RecoveryTarget::Semi | RecoveryTarget::RBrace)?;

    let span = kw_struct.span.merge(rbrace.span);
    Ok(Expr::StructDecl {
        kw_struct,
        lbrace,
        fields,
        rbrace,
    }.spanned(span))
}

fn parse_fn_decl(lex: &mut Lex, until: ParseUntil) -> Result<Ast<Expr>> {
    todo!()
}

fn parse_fn_type(lex: &mut Lex, until: ParseUntil) -> Result<Ast<Expr>> {
    todo!()
}

fn parse_type(lex: &mut Lex) -> Result<Ast<Expr>> {
    let kw_type = parse_type_kw(lex)?;
    if let Some((Ok(Token::LBracket), _)) = lex.peek() {
        let (lbracket, indicators, rbracket) = parse_params(lex)?;
        let span = kw_type.span.merge(rbracket.span);
        Ok(Expr::Type {
            kw_type,
            lbracket,
            indicators,
            rbracket,
        }.spanned(span))
    } else {
        let span = kw_type.span;
        Ok(Expr::PlainType {
            kw_type,
        }.spanned(span))
    }
}

fn parse_if(lex: &mut Lex, until: ParseUntil) -> Result<Ast<Expr>> {
    todo!()
}

fn parse_while(lex: &mut Lex, until: ParseUntil) -> Result<Ast<Expr>> {
    todo!()
}

fn parse_return(lex: &mut Lex, until: ParseUntil) -> Result<Ast<Expr>> {
    todo!()
}

fn parse_args(lex: &mut Lex) -> Result<(Ast<PunctLParen>, Vec<(Ast<Expr>, Option<Ast<PunctComma>>)>, Ast<PunctRParen>)> {
    let lparen = parse_lparen(lex);
    let mut args = Vec::new();
    let mut stop = false;
    while !matches!(lex.peek(), Some((Ok(Token::RParen), _)) | None) && !stop {
        let arg = parse_expr(lex, ParseUntil::Termination)
            .recover(lex, RecoveryTarget::Comma | RecoveryTarget::RParen);
        let comma = maybe_parse_comma(lex);
        stop = comma.ok_and_none();
        args.push(cmrg!(arg, comma));
        if stop { break }
    }
    let args = vmrg!(args).recover(lex, RecoveryTarget::RParen);
    let rparen = parse_rparen(lex);
    let (lparen, args, rparen) = cmrg!(lparen, args, rparen)?;
    Ok((lparen, args, rparen))
}

fn parse_params(lex: &mut Lex) -> Result<(Ast<PunctLBracket>, Vec<(Ast<Expr>, Option<Ast<PunctComma>>)>, Ast<PunctRBracket>)> {
    let lbracket = parse_lbracket(lex);
    let mut args = Vec::new();
    let mut stop = false;
    while !matches!(lex.peek(), Some((Ok(Token::RBracket), _)) | None) && !stop {
        let arg = parse_expr(lex, ParseUntil::Termination)
            .recover(lex, RecoveryTarget::Comma | RecoveryTarget::RBracket);
        let comma = maybe_parse_comma(lex);
        stop = comma.ok_and_none();
        args.push(cmrg!(arg, comma));
        if stop { break }
    }
    let args = vmrg!(args).recover(lex, RecoveryTarget::RBracket);
    let rbracket = parse_rbracket(lex);
    let (lbracket, args, rbracket) = cmrg!(lbracket, args, rbracket)?;
    Ok((lbracket, args, rbracket))
}

macro_rules! simple_parser {
    ($name:ident, $ty:ty: $p:pat => $e:expr, $err:expr) => {
        fn $name(lex: &mut Lex) -> Result<Ast<$ty>> {
            match lex.next() {
                Some((Ok($p), span)) => Ok(Ast { span: span.into(), node: $e }),
                Some((Ok(_), span)) => Err(vec![ParseError::Spanned(span.into(), $err.into())]),
                Some((Err(_), span)) => Err(vec![ParseError::UnexpectedToken(span.into())]),
                None => Err(vec![ParseError::UnexpectedEof($err.into())]),
            }
        }
    };
}

simple_parser!(parse_semi, PunctSemi: Token::Semi => PunctSemi, "Expected `;`");
simple_parser!(parse_comma, PunctComma: Token::Comma => PunctComma, "Expected `,`");
simple_parser!(parse_colon, PunctColon: Token::Colon => PunctColon, "Expected `:`");
simple_parser!(parse_arrow, PunctArrow: Token::Arrow => PunctArrow, "Expected `->`");
simple_parser!(parse_lbrace, PunctLBrace: Token::LBrace => PunctLBrace, "Expected `{`");
simple_parser!(parse_rbrace, PunctRBrace: Token::RBrace => PunctRBrace, "Expected `}`");
simple_parser!(parse_lbracket, PunctLBracket: Token::LBracket => PunctLBracket, "Expected `[`");
simple_parser!(parse_rbracket, PunctRBracket: Token::RBracket => PunctRBracket, "Expected `]`");
simple_parser!(parse_lparen, PunctLParen: Token::LParen => PunctLParen, "Expected `(`");
simple_parser!(parse_rparen, PunctRParen: Token::RParen => PunctRParen, "Expected `)`");
simple_parser!(parse_struct_kw, KwStruct: Token::Struct => KwStruct, "Expected `struct`");
simple_parser!(parse_type_kw, KwType: Token::Type => KwType, "Expected `type`");
simple_parser!(parse_eq, PunctEq: Token::Eq => PunctEq, "Expected `=`");
simple_parser!(parse_fn, KwFn: Token::Fn => KwFn, "Expected `fn`");
simple_parser!(parse_fn_ty, KwFnTy: Token::FnTy => KwFnTy, "Expected `Fn`");
simple_parser!(parse_return_kw, KwReturn: Token::Return => KwReturn, "Expected `return`");
simple_parser!(parse_if_kw, KwIf: Token::If => KwIf, "Expected `if`");
simple_parser!(parse_while_kw, KwWhile: Token::While => KwWhile, "Expected `while`");
simple_parser!(parse_ident, SmolStr2: Token::Ident(s) => s, "Expected identifier");

fn maybe_parse_comma(lex: &mut Lex) -> Result<Option<Ast<PunctComma>>> {
    Ok(if let Some((Ok(Token::Comma), _)) = lex.peek() {
        Some(parse_comma(lex)?)
    } else {
        None
    })
}

fn maybe_parse_semi(lex: &mut Lex) -> Result<Option<Ast<PunctSemi>>> {
    Ok(if let Some((Ok(Token::Semi), _)) = lex.peek() {
        Some(parse_semi(lex)?)
    } else {
        None
    })
}

bitflags! {
    struct RecoveryTarget: u32 {
        const Semi = 1 << 0;
        const Comma = 1 << 1;
        const Colon = 1 << 2;
        const Arrow = 1 << 3;
        const Eq = 1 << 4;
        const RBrace = 1 << 5;
        const RParen = 1 << 6;
        const RBracket = 1 << 7;
    }
}

fn try_recover(lex: &mut Lex, target: RecoveryTarget) -> Result<()> {
    enum ParenKind {
        Paren(Span),
        Bracket(Span),
        Brace(Span),
    }

    let mut paren_stack = Vec::new();
    let mut errors = Vec::new();
    while let Some((t, span)) = lex.peek() {
        let ok = match (t, paren_stack.len()) {
            (Ok(Token::RParen), 0) if target.intersects(RecoveryTarget::RParen) => true,
            (Ok(Token::RBracket), 0) if target.intersects(RecoveryTarget::RBracket) => true,
            (Ok(Token::RBrace), 0) if target.intersects(RecoveryTarget::RBrace) => true,
            (Ok(Token::Semi), 0) if target.intersects(RecoveryTarget::Semi) => true,
            (Ok(Token::Comma), 0) if target.intersects(RecoveryTarget::Comma) => true,
            (Ok(Token::Colon), 0) if target.intersects(RecoveryTarget::Colon) => true,
            (Ok(Token::Arrow), 0) if target.intersects(RecoveryTarget::Arrow) => true,
            (Ok(Token::Eq), 0) if target.intersects(RecoveryTarget::Eq) => true,
            (Ok(Token::LParen), _) => {
                paren_stack.push(ParenKind::Paren(span.into()));
                false
            }
            (Ok(Token::LBracket), _) => {
                paren_stack.push(ParenKind::Bracket(span.into()));
                false
            }
            (Ok(Token::LBrace), _) => {
                paren_stack.push(ParenKind::Brace(span.into()));
                false
            }
            (Ok(Token::RParen), _) => {
                if let Some(ParenKind::Paren(start)) = paren_stack.pop() {
                    errors.push(ParseError::Spanned(start, "Unmatched `(`".into()));
                }
                false
            }
            (Ok(Token::RBracket), _) => {
                if let Some(ParenKind::Bracket(start)) = paren_stack.pop() {
                    errors.push(ParseError::Spanned(start, "Unmatched `[`".into()));
                }
                false
            }
            (Ok(Token::RBrace), _) => {
                if let Some(ParenKind::Brace(start)) = paren_stack.pop() {
                    errors.push(ParseError::Spanned(start, "Unmatched `{`".into()));
                }
                false
            }
            (Ok(_), _) => false,
            (Err(_), _) => {
                errors.push(ParseError::UnexpectedToken(span.into()));
                false
            },
        };
        if ok {
            return if errors.is_empty() {
                Ok(())
            } else {
                Err(errors)
            };
        } else {
            lex.next();
        }
    }

    Err(vec![ParseError::UnexpectedEof("Unexpected EOF".into())])
}
