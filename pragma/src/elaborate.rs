use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::rc::Rc;
use indexmap::IndexMap;
use serde::Serialize;
use crate::ast::{Ast, BinaryOp, Expr, Item, Module, NodeExt, Param, UnaryOp};
use crate::{c, cmrg};
use crate::compound_result::{CompoundResult, err};
use crate::intrinsics::{Intrinsic, make_intrinsics};
use crate::smol_str2::SmolStr2;
use crate::span::Span;

#[derive(Debug, PartialEq)]
pub enum Term {
    Var(SmolStr2),
    Meta(usize),
    Undef,
    Int(i64),
    String(SmolStr2),
    Bool(bool),
    Unit,
    Type(Vec<Rc<Term>>),
    IntTy,
    StringTy,
    BoolTy,
    UnitTy,
    Intrinsic(&'static dyn Intrinsic),
    Fn { args: Vec<(SmolStr2, Rc<Term>)>, body: Rc<Term> },
    FnTy { args: Vec<Rc<Term>>, ret_ty: Rc<Term> },
    Call { obj: Rc<Term>, args: Vec<Rc<Term>> },
    Pi { ty: Rc<Term>, ident: SmolStr2, body: Rc<Term> },
    Lam { ty: Rc<Term>, ident: SmolStr2, body: Rc<Term> },
    App { obj: Rc<Term>, arg: Rc<Term> },
    Match { obj: Rc<Term>, cases: Vec<(Rc<Term>, Rc<Term>)>, default: Rc<Term> },
    Source { span: Span, term: Rc<Term> },
    Decl { name: SmolStr2, ty: Rc<Term>, val: Rc<Term> },
    Block { stmts: Vec<Rc<Term>>, ret: Rc<Term> },
    Typed { term: Rc<Term>, ty: Rc<Term> },
    If { cond: Rc<Term>, then: Rc<Term>, else_: Rc<Term> },
    Assign { lhs: SmolStr2, rhs: Rc<Term> },
    While { cond: Rc<Term>, body: Rc<Term> },
}

#[derive(Debug, Serialize)]
pub enum ElaborateError {
    ArithmeticError(Span, String),
    UnificationFailed(Span, String),
    Commented(Span, String),
}

pub type Result<T> = CompoundResult<T, ElaborateError>;

trait ResultExt<T> {
    fn with_comment(self, comment: impl Display) -> Self;
}

impl<T> ResultExt<T> for Result<T> {
    fn with_comment(self, comment: impl Display) -> Result<T> {
        match self {
            Ok(value) => Ok(value),
            Err(errors) => Err(
                errors.into_iter().map(|e| {
                    match e {
                        ElaborateError::Commented(span, c) =>
                            ElaborateError::Commented(span, format!("{}: {}", comment, c)),
                        ElaborateError::UnificationFailed(span, c) =>
                            ElaborateError::Commented(span, format!("{}: {}", comment, c)),
                        ElaborateError::ArithmeticError(span, c) =>
                            ElaborateError::Commented(span, format!("{}: {}", comment, c)),
                    }
                }).collect()
            )
        }
    }
}

pub struct Ctx {
    star: Rc<Term>,
    undef: Rc<Term>,
    anon: SmolStr2,
    anon_term: Rc<Term>,
    unit: Rc<Term>,
    unit_ty: Rc<Term>,
    pub items: IndexMap<SmolStr2, (Rc<Term>, Rc<Term>)>,
    locals: Vec<(SmolStr2, (Rc<Term>, Rc<Term>))>,
    span: Span,
    metas: Vec<(Option<Rc<Term>>, Rc<Term>)>,
    block_boundaries: Vec<usize>,
    stop_substitution_mark: bool,
}

impl Ctx {
    fn lookup(&self, ident: SmolStr2) -> Option<(Rc<Term>, Rc<Term>)> {
        for (name, value) in self.locals.iter().rev() {
            if name == &ident {
                return Some(value.clone());
            }
        }
        self.items.get(&ident).cloned()
    }

    fn lookup_ident(&self, ident: &Ast<SmolStr2>) -> Result<(Rc<Term>, Rc<Term>)> {
        match self.lookup(ident.node.clone()) {
            Some(value) => Ok(value),
            None => err(ElaborateError::Commented(ident.span, format!("Identifier {:?} not found", ident.node))),
        }
    }

    fn with_local<T>(&mut self, ident: SmolStr2, value: Rc<Term>, ty: Rc<Term>, f: impl FnOnce(&mut Self) -> T) -> T {
        self.locals.push((ident.clone(), (value, ty)));
        let result = f(self);
        self.locals.pop();
        result
    }

    fn with_locals<T>(&mut self, decls: &[(SmolStr2, (Rc<Term>, Rc<Term>))], f: impl FnOnce(&mut Self) -> T) -> T {
        for (ident, value) in decls {
            self.locals.push((ident.clone(), value.clone()));
        }
        let result = f(self);
        for _ in decls {
            self.locals.pop();
        }
        result
    }

    fn with_local_decl<T>(&mut self, ident: SmolStr2, ty: Rc<Term>, f: impl FnOnce(&mut Self) -> T) -> T {
        self.with_local(ident.clone(), local(ident), ty.clone(), |ctx| {
            f(ctx)
        })
    }

    fn with_local_decls<T>(&mut self, decls: &[(SmolStr2, Rc<Term>)], f: impl FnOnce(&mut Self) -> T) -> T {
        for (ident, ty) in decls {
            self.locals.push((ident.clone(), (local(ident.clone()), ty.clone())));
        }
        let result = f(self);
        for _ in decls {
            self.locals.pop();
        }
        result
    }

    fn with_block<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        self.block_boundaries.push(self.locals.len());
        let result = f(self);
        let bound = self.block_boundaries.pop().unwrap();
        self.locals.truncate(bound);
        self.stop_substitution_mark = false;
        result
    }

    fn with_span<T>(&mut self, span: Span, f: impl FnOnce(&mut Self) -> T) -> T {
        let old = self.span;
        self.span = span;
        let result = f(self);
        self.span = old;
        result
    }

    fn with_case<T>(&mut self, obj: Rc<Term>, pat: Rc<Term>, f: impl FnOnce(&mut Self) -> T) -> T {
        if let Term::Var(name) = &*obj {
            self.with_local(name.clone(), pat.clone(), pat.clone(), f)
        } else {
            f(self)
        }
    }

    fn consider(&mut self, term: &Rc<Term>) {
        assert!(!self.block_boundaries.is_empty());
        match &**term {
            Term::Decl { name, ty, val } => {
                self.locals.push((name.clone(), (local(name.clone()), ty.clone())));
                self.consider(val);
            }
            Term::Call { obj, args } => {
                self.consider(obj);
                for arg in args {
                    self.consider(arg);
                }
            }
            Term::Typed { term, .. } => {
                self.consider(term);
            }
            Term::Source { term, .. } => {
                self.consider(term);
            }
            _ => {}
        }
    }

    fn fresh(&mut self, old: SmolStr2) -> SmolStr2 {
        if self.lookup(old.clone()).is_none() {
            old
        } else {
            for i in 0.. {
                let new = SmolStr2::from(format!("{}@{}", &old[..], i));
                if self.lookup(new.clone()).is_none() {
                    return new;
                }
            }
            unreachable!()
        }
    }

    fn fresh_type_meta(&mut self) -> Rc<Term> {
        let meta = Rc::new(Term::Meta(self.metas.len()));
        self.metas.push((None, Rc::new(Term::Type(Vec::new()))));
        meta
    }

    fn fresh_meta(&mut self) -> Rc<Term> {
        let meta = Rc::new(Term::Meta(self.metas.len()));
        let ty_meta = self.fresh_type_meta();
        self.metas.push((None, ty_meta));
        meta
    }
}

fn local(name: SmolStr2) -> Rc<Term> {
    Rc::new(Term::Var(name))
}

fn fresh_common(ctx: &mut Ctx, a: SmolStr2, b: SmolStr2) -> SmolStr2 {
    ctx.with_local(a.clone(), ctx.undef.clone(), ctx.undef.clone(), |ctx| {
        ctx.with_local(b.clone(), ctx.undef.clone(), ctx.undef.clone(), |ctx| {
            ctx.fresh(b)
        })
    })
}

fn type_of(ctx: &mut Ctx, term: Rc<Term>) -> Result<Rc<Term>> {
    Ok(match &*term {
        Term::Var(ident) => ctx.lookup(ident.clone()).expect("Ident not found").1.clone(),
        Term::Meta(id) => ctx.metas[*id].1.clone(),
        Term::Undef => ctx.undef.clone(),
        Term::Int(_) => Rc::new(Term::IntTy),
        Term::String(_) => Rc::new(Term::StringTy),
        Term::Bool(_) => Rc::new(Term::BoolTy),
        Term::Unit => Rc::new(Term::UnitTy),
        Term::Type(_) => term.clone(),
        Term::IntTy => ctx.star.clone(),
        Term::StringTy => ctx.star.clone(),
        Term::BoolTy => ctx.star.clone(),
        Term::UnitTy => ctx.star.clone(),
        Term::Call { obj, .. } => {
            let obj_ty = type_of(ctx, obj.clone())?;
            let obj_ty = nf(ctx, obj_ty).expect("Typed code failed to normalize");
            match &*obj_ty {
                Term::FnTy { ret_ty, .. } => ret_ty.clone(),
                _ => return err(ElaborateError::Commented(
                    ctx.span,
                    format!("Expected a function, got `{}`", obj_ty),
                )),
            }
        }
        Term::Pi { .. } => ctx.star.clone(), // TODO: maybe filter out indicators properly
        Term::Lam { ident, ty, body } => {
            ctx.with_local(ident.clone(), local(ident.clone()), ty.clone(), |ctx| {
                let body = type_of(ctx, body.clone())?;
                Result::Ok(Rc::new(Term::Pi { ident: ident.clone(), ty: ty.clone(), body }))
            })?
        }
        Term::FnTy { .. } => ctx.star.clone(),
        Term::App { obj, arg } => {
            let obj_ty = type_of(ctx, obj.clone())?;
            Rc::new(Term::App {
                obj: obj_ty.clone(),
                arg: arg.clone(),
            })
        }
        Term::Match { obj, cases, default } => {
            let cases = cases.iter().map(|(pat, body)| {
                ctx.with_case(obj.clone(), pat.clone(), |ctx| {
                    let body_ty = type_of(ctx, body.clone())?;
                    Ok((pat.clone(), body_ty))
                })
            }).collect::<Result<_>>()?;
            let default_ty = type_of(ctx, default.clone())?;
            Rc::new(Term::Match { obj: obj.clone(), cases, default: default_ty.clone() })
        }
        Term::Fn { args, body } => {
            let ret_ty = ctx.with_local_decls(args, |ctx| {
                type_of(ctx, body.clone())
            })?;
            Rc::new(Term::FnTy { args: args.iter().map(|(_, ty)| ty).cloned().collect(), ret_ty })
        }
        Term::Source { term, .. } => return type_of(ctx, term.clone()),
        Term::Block { ret, stmts } => {
            return ctx.with_block(|ctx| {
                for stmt in stmts {
                    ctx.consider(stmt);
                }
                type_of(ctx, ret.clone())
            })
        }
        Term::Decl { .. } => ctx.unit_ty.clone(),
        Term::Typed { ty, .. } => ty.clone(),
        Term::Intrinsic(intrinsic) => intrinsic.type_of(),
        Term::If { then, .. } => return type_of(ctx, then.clone()),
        Term::Assign { .. } => ctx.unit_ty.clone(),
        Term::While { .. } => ctx.unit_ty.clone(),
    })
}

pub fn elaborate(module: &Module) -> Result<Ctx> {
    let anon = SmolStr2::from("<anon>");
    let mut ctx = Ctx {
        star: Rc::new(Term::Type(Vec::new())),
        undef: Rc::new(Term::Undef),
        anon: anon.clone(),
        anon_term: Rc::new(Term::Var(anon.clone())),
        unit: Rc::new(Term::Unit),
        unit_ty: Rc::new(Term::UnitTy),
        items: make_intrinsics(),
        locals: Vec::new(),
        span: Span(0, 0),
        metas: Vec::new(),
        block_boundaries: Vec::new(),
        stop_substitution_mark: false,
    };

    for item in &module.items {
        elaborate_item(&mut ctx, &item.node)?;
    }

    Ok(ctx)
}

fn elaborate_item(ctx: &mut Ctx, item: &Item) -> Result<()> {
    fn elaborate_item_impl(ctx: &mut Ctx, item: &Item, depth: usize) -> Result<(Rc<Term>, Rc<Term>)> {
        let result = match item.params.get(depth).map(|p| &p.0.node) {
            Some(Param::Value { value, .. }) => {
                let inferred = infer(ctx, value);
                let elaborated = elaborate_item_impl(ctx, item, depth + 1);
                let ((value, ty), (body, out_ty)) = cmrg!(inferred, elaborated)?;
                (
                    Rc::new(Term::Lam {
                        ident: ctx.anon.clone(),
                        ty: ty.clone(),
                        body: Rc::new(Term::Match {
                            obj: ctx.anon_term.clone(),
                            cases: vec![(value.clone(), body)],
                            default: ctx.undef.clone(),
                        }),
                    }),
                    Rc::new(Term::Pi {
                        ident: ctx.anon.clone(),
                        ty,
                        body: Rc::new(Term::Match {
                            obj: ctx.anon_term.clone(),
                            cases: vec![(value, out_ty)],
                            default: ctx.undef.clone(),
                        }),
                    }),
                )
            }
            Some(Param::Generic { ident, ty, .. }) => {
                let ty = check(ctx, ty, &mut ctx.star.clone())?;
                let (body, out_ty) = if let Some(ident) = ident {
                    ctx.with_local(ident.node.clone(), local(ident.node.clone()), ty.clone(), |ctx| {
                        elaborate_item_impl(ctx, item, depth + 1)
                    })?
                } else {
                    elaborate_item_impl(ctx, item, depth + 1)?
                };
                (
                    Rc::new(Term::Lam {
                        ident: ident.as_ref().map(|i| i.node.clone()).unwrap_or(ctx.anon.clone()),
                        ty: ty.clone(),
                        body,
                    }),
                    Rc::new(Term::Pi {
                        ident: ident.as_ref().map(|i| i.node.clone()).unwrap_or(ctx.anon.clone()),
                        ty,
                        body: out_ty,
                    }),
                )
            }
            None => {
                if let Some(annotation) = &item.ret_ty {
                    let mut ty = check(ctx, annotation, &mut ctx.star.clone())?;
                    let body = if let Some(body) = &item.body {
                        check(ctx, body, &mut ty)?
                    } else {
                        ctx.undef.clone()
                    };
                    (body, ty)
                } else if let Some(body) = &item.body {
                    let (body, ty) = infer(ctx, body)?;
                    (body, ty)
                } else {
                    (ctx.undef.clone(), ctx.undef.clone())
                }
            }
        };

        Ok(result)
    }

    ctx.with_span(item.ident.span, |ctx| {
        let (new, _) = elaborate_item_impl(ctx, item, 0)?;

        let new = nf(ctx, new)?;
        let new_ty = type_of(ctx, new.clone())?;
        let new_ty = nf(ctx, new_ty)?;

        if contains_metas(&new) || contains_metas(&new_ty) {
            return err(ElaborateError::Commented(
                ctx.span,
                format!("The item contains unsolved metas: `{}` having type `{}`", new, new_ty),
            ));
        }

        let (existing, existing_ty) = match ctx.items.get(&item.ident.node) {
            Some((existing, existing_ty)) => (existing.clone(), existing_ty.clone()),
            None => {
                ctx.items.insert(item.ident.node.clone(), (new.clone(), new_ty.clone()));
                return Ok(());
            }
        };

        let ty1 = merge_into(ctx, existing_ty.clone(), new_ty.clone())
            .with_comment(format!("Cannot merge the existing item with this because of type mismatch: `{}` and `{}`", existing_ty, new_ty))?;
        let value1 = merge_into(ctx, existing.clone(), new.clone())
            .with_comment(format!("Cannot merge the existing item with this because of value mismatch: `{}` and `{}`", existing, new))?;
        let (value, _) = (value1, ty1);

        let value = nf(ctx, value)?;
        // let ty = nf(ctx, ty)?;
        let ty = type_of(ctx, value.clone())?;
        let ty = nf(ctx, ty)?;

        ctx.items.insert(item.ident.node.clone(), (value, ty));
        Ok(())
    })
}

/// Infer the largest type of `expr` and return its value.
fn infer(ctx: &mut Ctx, expr: &Ast<Expr>) -> Result<(Rc<Term>, Rc<Term>)> {
    ctx.with_span(expr.span, |ctx| {
        let (val, ty) = match &expr.node {
            Expr::Int(value) => {
                let value = match (*value).try_into() {
                    Ok(value) => value,
                    Err(_) => return err(ElaborateError::ArithmeticError(expr.span, "Integer overflow".to_string()))
                };
                (Term::Int(value), Term::IntTy)
            }
            Expr::String(value) => (Term::String(value.clone()), Term::StringTy),
            Expr::Bool(value) => (Term::Bool(*value), Term::BoolTy),
            Expr::Unit => (Term::Unit, Term::UnitTy),
            Expr::PlainType { .. } => (Term::Type(Vec::new()), Term::Type(Vec::new())),
            Expr::Ident(ident) => return ctx.lookup_ident(&ident.clone().spanned(expr.span)),
            Expr::Paren { expr, .. } => return infer(ctx, expr),
            Expr::Hole => {
                let meta = ctx.fresh_meta();
                return Ok((meta.clone(), type_of(ctx, meta)?));
            }
            Expr::Uninit => (Term::Undef, Term::Undef),

            Expr::App { callee, params, .. } => {
                let (mut callee, mut callee_ty) = infer(ctx, callee)?;
                for param in params {
                    callee_ty = unify(ctx, callee_ty.clone(), Rc::new(Term::Pi {
                        ident: ctx.anon.clone(),
                        ty: ctx.undef.clone(),
                        body: ctx.undef.clone(),
                    })).with_comment(format!("y `{}` to anything because it has type `{}`", callee, callee_ty))?;
                    let param = &param.0;
                    let (param, param_ty) = infer(ctx, param)?;
                    if unify(ctx, callee_ty.clone(), Rc::new(Term::Pi {
                        ident: ctx.anon.clone(),
                        ty: param_ty.clone(),
                        body: ctx.undef.clone(),
                    })).is_err() {
                        return err(ElaborateError::Commented(
                            ctx.span,
                            format!("Cannot apply `{}` to `{}` because it has type `{}`", callee, param, callee_ty),
                        ));
                    }
                    (callee, callee_ty) = (
                        Rc::new(Term::App { obj: callee, arg: param.clone() }),
                        Rc::new(Term::App { obj: callee_ty, arg: param }),
                    )
                }
                return Ok((callee, callee_ty));
            }

            Expr::FnDecl { body, args, ret_ty, .. } => {
                let args = args.iter().map(|(ident, _, ty, _)| {
                    let ty = check(ctx, ty, &mut ctx.star.clone())?;
                    Ok((ident.node.clone(), ty))
                }).collect::<Result<Vec<_>>>()?;

                let (body, inferred_ret) = ctx.with_local_decls(&args, |ctx| {
                    infer(ctx, body)
                })?;

                if let Some((_, ret_ty)) = ret_ty {
                    let ret_ty = check(ctx, ret_ty, &mut ctx.star.clone())?;
                    unify(ctx, inferred_ret.clone(), ret_ty.clone())?;
                }

                (
                    Term::Fn {
                        args: args.clone(),
                        body,
                    },
                    Term::FnTy {
                        args: args.iter().map(|(_, ty)| ty.clone()).collect(),
                        ret_ty: inferred_ret,
                    },
                )
            }

            Expr::Call { callee, args, .. } => {
                let (callee, callee_ty) = infer(ctx, callee)?;
                return match &*callee_ty {
                    Term::FnTy { args: expected_args, ret_ty } => {
                        if args.len() != expected_args.len() {
                            return err(ElaborateError::Commented(
                                ctx.span,
                                format!("Expected {} arguments, got {}", expected_args.len(), args.len()),
                            ));
                        }
                        let args = args.iter().zip(expected_args).map(|(arg, ty)| {
                            check(ctx, &arg.0, &mut ty.clone())
                        }).collect::<Result<Vec<_>>>()?;
                        Ok((Rc::new(Term::Call { obj: callee, args }), ret_ty.clone()))
                    }
                    _ => err(ElaborateError::Commented(
                        ctx.span,
                        format!("Expected a function, got `{}`", callee_ty),
                    )),
                };
            }

            Expr::FnType { args, ret, .. } => {
                let args = args.iter().map(|(ty, _)| {
                    let ty = check(ctx, ty, &mut ctx.star.clone())?;
                    Ok(ty)
                }).collect::<Result<Vec<_>>>()?;

                let ret_ty = check(ctx, ret, &mut ctx.star.clone())?;

                return Ok((
                    Rc::new(Term::FnTy {
                        args,
                        ret_ty,
                    }),
                    ctx.star.clone(),
                ));
            }

            Expr::Unary { op, expr } => {
                let (expr, ty) = infer(ctx, expr)?;
                let item = SmolStr2::from(match op.node {
                    UnaryOp::Neg => "neg",
                    UnaryOp::Pos => "pos",
                    UnaryOp::Not => "not",
                    UnaryOp::Ref => "ref",
                    UnaryOp::Deref => "deref",
                });

                if ctx.lookup(item.clone()).is_none() {
                    return err(ElaborateError::Commented(
                        op.span,
                        format!("Unary operator `{}` not found", &item[..]),
                    ));
                }

                let term = Rc::new(Term::Call {
                    obj: Rc::new(Term::App {
                        obj: Rc::new(Term::Var(item.clone())),
                        arg: ty.clone(),
                    }),
                    args: vec![expr],
                });

                let ty = type_of(ctx, term.clone())?;
                return Ok((term, ty));
            }

            Expr::Binary { op, lhs, rhs } => {
                let (lhs, lhs_ty) = infer(ctx, lhs)?;
                let (rhs, rhs_ty) = infer(ctx, rhs)?;

                let item = SmolStr2::from(match op.node {
                    BinaryOp::Add => "add",
                    BinaryOp::Sub => "sub",
                    BinaryOp::Mul => "mul",
                    BinaryOp::Div => "div",
                    BinaryOp::Rem => "rem",
                    BinaryOp::And => "and",
                    BinaryOp::Or => "or",
                    BinaryOp::Xor => "xor",
                    BinaryOp::Eq => "eq",
                    BinaryOp::Ne => "ne",
                    BinaryOp::Lt => "lt",
                    BinaryOp::Le => "le",
                    BinaryOp::Gt => "gt",
                    BinaryOp::Ge => "ge",
                });

                if ctx.lookup(item.clone()).is_none() {
                    return err(ElaborateError::Commented(
                        op.span,
                        format!("Binary operator `{}` not found", &item[..]),
                    ));
                }

                let term = Rc::new(Term::Call {
                    obj: Rc::new(Term::App {
                        obj: Rc::new(Term::App {
                            obj: Rc::new(Term::Var(item.clone())),
                            arg: lhs_ty.clone(),
                        }),
                        arg: rhs_ty.clone(),
                    }),
                    args: vec![lhs, rhs],
                });

                let ty = type_of(ctx, term.clone())?;
                return Ok((term, ty));
            }

            Expr::Block { stmts, ret, .. } => {
                return ctx.with_block(|ctx| {
                    let stmts = stmts.iter().map(
                        |stmt| {
                            let (val, ty) = infer(ctx, &stmt.0)?;
                            ctx.consider(&val);
                            Ok((val, ty))
                        }
                    ).collect::<Result<Vec<_>>>()?;

                    let (ret, ret_ty) = if let Some(ret) = ret {
                        infer(ctx, ret)?
                    } else {
                        (ctx.unit.clone(), ctx.unit_ty.clone())
                    };

                    Ok((
                        Rc::new(Term::Block {
                            stmts: stmts.iter().map(|(stmt, _)| stmt.clone()).collect(),
                            ret,
                        }),
                        ret_ty,
                    ))
                })
            }

            Expr::Decl { ident, ty, val , .. } => {
                let ty = ty.as_ref().map(|ty| check(ctx, ty, &mut ctx.star.clone())).transpose()?;
                let (val, ty) = if let Some(mut ty) = ty {
                    let val = check(ctx, val, &mut ty)?;
                    (val, ty)
                } else {
                    infer(ctx, val)?
                };
                return Ok((
                    Rc::new(Term::Decl {
                        name: ident.node.clone(),
                        ty,
                        val,
                    }),
                    ctx.unit_ty.clone(),
                ))
            }

            Expr::If { cond, then, else_, .. } => {
                let cond = check(ctx, &cond, &mut Rc::new(Term::BoolTy))?;
                let (then, then_ty) = infer(ctx, then)?;

                let (else_, else_ty) = if let Some(else_) = else_ {
                    infer(ctx, else_)?
                } else {
                    (ctx.unit.clone(), ctx.unit_ty.clone())
                };

                let ty = unify(ctx, then_ty.clone(), else_ty.clone())
                    .with_comment("If branches types must match")?;

                return Ok((
                    Rc::new(Term::If {
                        cond,
                        then,
                        else_,
                    }),
                    ty,
                ))
            }
            
            Expr::Assign { lvalue, value, .. } => {
                let (lvalue, mut ty) = infer(ctx, lvalue)?;
                let value = check(ctx, value, &mut ty)
                    .with_comment("Can't assign because of type mismatch")?;
                
                return Ok((
                    Rc::new(Term::Assign {
                        lhs: match &*lvalue {
                            Term::Var(name) => name.clone(),
                            _ => return err(ElaborateError::Commented(
                                ctx.span,
                                format!("Can't assign to `{}`", lvalue),
                            )),
                        },
                        rhs: value,
                    }),
                    ctx.unit_ty.clone(),
                ))
            }
            
            Expr::While { cond, body, .. } => {
                let cond = check(ctx, &cond, &mut Rc::new(Term::BoolTy))?;
                let (body, _) = infer(ctx, body)?;
                return Ok((
                    Rc::new(Term::While {
                        cond,
                        body,
                    }),
                    ctx.unit_ty.clone(),
                ))
            }
            

            _ => todo!("Can't infer type of expression yet {:#?}", expr.node),
        };

        Ok((Rc::new(val), Rc::new(ty)))
    })
}

/// Check that `expr` has type `ty` and return the inferred value,
/// possibly strengthening the type in the process.
fn check(ctx: &mut Ctx, expr: &Ast<Expr>, ty: &mut Rc<Term>) -> Result<Rc<Term>> {
    ctx.with_span(expr.span, |ctx| {
        match &expr.node {
            _ => {
                let (value, inferred) = infer(ctx, expr)?;
                *ty = unify(ctx, inferred.clone(), ty.clone())
                    .with_comment("Typecheck failed")?;
                Ok(value)
            }
        }
    })
}

/// Return the largest type that is a subtype of both `a` and `b`.
fn unify(ctx: &mut Ctx, a: Rc<Term>, b: Rc<Term>) -> Result<Rc<Term>> {
    let a = nf(ctx, a)?;
    let b = nf(ctx, b)?;
    use Term::*;

    let result = match (&*a, &*b) {
        (_, Undef) => return Ok(a.clone()),
        (Undef, _) => return Ok(b.clone()),
        (Meta(id), _) => {
            let a_ty = type_of(ctx, a.clone())?;
            let b_ty = type_of(ctx, b.clone())?;
            unify(ctx, a_ty, b_ty.clone())?;
            if let Some(value) = &ctx.metas[*id].0 {
                return unify(ctx, value.clone(), b);
            }
            ctx.metas[*id] = (Some(b.clone()), b_ty);
            return Ok(b);
        }
        (_, Meta(id)) => {
            let a_ty = type_of(ctx, a.clone())?;
            let b_ty = type_of(ctx, b.clone())?;
            unify(ctx, a_ty.clone(), b_ty)?;
            if let Some(value) = &ctx.metas[*id].0 {
                return unify(ctx, a, value.clone());
            }
            ctx.metas[*id] = (Some(a.clone()), a_ty);
            return Ok(a);
        }
        (UnitTy, UnitTy) => UnitTy,
        (IntTy, IntTy) => IntTy,
        (StringTy, StringTy) => StringTy,
        (BoolTy, BoolTy) => BoolTy,
        (Unit, Unit) => Unit,
        (Int(x), Int(y)) if x == y => Int(*x),
        (String(x), String(y)) if x == y => String(x.clone()),
        (Bool(x), Bool(y)) if x == y => Bool(*x),
        (Var(x), Var(y)) if x == y => Var(x.clone()),

        (Type(xs), Pi { .. }) if xs.is_empty() => return Ok(b.clone()),
        (Pi { .. }, Type(xs)) if xs.is_empty() => return Ok(b.clone()),

        (Type(xs), Type(ys)) => {
            let result = Type(Vec::new());
            assert!(xs.is_empty() && ys.is_empty(), "TODO: Unify type parameters");
            result
        }

        (Pi { ident: ident1, ty: ty1, body: body1 }, Pi { ident: ident2, ty: ty2, body: body2 }) => {
            let ident = fresh_common(ctx, ident1.clone(), ident2.clone());

            let ty = unify(ctx, ty1.clone(), ty2.clone())?;
            let body1 = substitute(ctx, body1.clone(), ident1.clone(), local(ident.clone()));
            let body2 = substitute(ctx, body2.clone(), ident2.clone(), local(ident.clone()));
            let body = unify(ctx, body1, body2)?;

            Pi { ident, ty, body }
        }

        (
            Match { obj: obj1, cases: cases1, default: default1 },
            Match { obj: obj2, cases: cases2, default: default2 },
        ) => {
            let obj = unify(ctx, obj1.clone(), obj2.clone())
                .with_comment("Only match expressions acting on the same object can be unified")?;

            for (pat, _) in cases1 {
                if !is_concrete(pat) {
                    return err(ElaborateError::Commented(
                        ctx.span,
                        format!("Cannot unify match expression with non-concrete pattern `{}`", pat),
                    ));
                }
            }

            for (pat, _) in cases2 {
                if !is_concrete(pat) {
                    return err(ElaborateError::Commented(
                        ctx.span,
                        format!("Cannot unify match expression with non-concrete pattern `{}`", pat),
                    ));
                }
            }

            let mut cases = Vec::new();

            let mut matched1 = vec![false; cases1.len()];
            let mut matched2 = vec![false; cases2.len()];

            for (i, (pat1, body1)) in cases1.iter().enumerate() {
                for (j, (pat2, body2)) in cases2.iter().enumerate() {
                    if matched2[j] {
                        continue;
                    }

                    if unify(ctx, pat1.clone(), pat2.clone()).is_ok() {
                        ctx.with_case(obj.clone(), pat1.clone(), |ctx| {
                            let body = unify(ctx, body1.clone(), body2.clone())?;
                            cases.push((pat1.clone(), body));
                            matched1[i] = true;
                            matched2[j] = true;
                            Result::Ok(())
                        })?;
                    }
                }
            }

            let mut default = unify(ctx, default1.clone(), default2.clone())?;

            for (matched, (_, body)) in matched1.iter().zip(cases1.iter()) {
                if !matched {
                    default = ctx.with_case(obj.clone(), body.clone(), |ctx| {
                        merge_into(ctx, default.clone(), body.clone())
                    })?;
                }
            }

            for (matched, (_, body)) in matched2.iter().zip(cases2.iter()) {
                if !matched {
                    default = ctx.with_case(obj.clone(), body.clone(), |ctx| {
                        merge_into(ctx, default.clone(), body.clone())
                    })?;
                }
            }

            Match { obj, cases, default }
        }

        (_, Match { obj, cases, default }) => {
            Match {
                obj: obj.clone(),
                cases: cases.iter().map(|(pat, body)| {
                    let pat = unify(ctx, pat.clone(), ctx.undef.clone())?;
                    let body = unify(ctx, body.clone(), ctx.undef.clone())?;
                    Ok((pat, body))
                }).collect::<Result<_>>()?,
                default: unify(ctx, a, default.clone())?,
            }
        }

        (Match { obj, cases, default }, _) => {
            Match {
                obj: obj.clone(),
                cases: cases.iter().map(|(pat, body)| {
                    let pat = unify(ctx, pat.clone(), ctx.undef.clone())?;
                    let body = unify(ctx, body.clone(), ctx.undef.clone())?;
                    Ok((pat, body))
                }).collect::<Result<_>>()?,
                default: unify(ctx, default.clone(), b)?,
            }
        }

        (
            FnTy { args: args1, ret_ty: ret1, .. },
            FnTy { args: args2, ret_ty: ret2, .. },
        ) => {
            if args1.len() != args2.len() {
                return err(ElaborateError::UnificationFailed(
                    ctx.span,
                    format!("Cannot unify functions with different number of arguments: `{}` and `{}`", a, b),
                ));
            }

            let args = args1.iter().zip(args2).map(|(arg1, arg2)| {
                unify(ctx, arg1.clone(), arg2.clone())
            }).collect::<Result<Vec<_>>>()?;

            let ret = unify(ctx, ret1.clone(), ret2.clone())?;

            FnTy { args, ret_ty: ret }
        }

        _ => return err(ElaborateError::UnificationFailed(
            ctx.span,
            format!("Cannot unify `{}` and `{}`", a, b)),
        )
    };

    Ok(Rc::new(result))
}

fn merge_into(ctx: &mut Ctx, term: Rc<Term>, add: Rc<Term>) -> Result<Rc<Term>> {
    let term = nf(ctx, term)?;
    let add = nf(ctx, add)?;

    match (&*term, &*add) {
        (
            Term::Pi { ident: ident1, ty: ty1, body: body1 },
            Term::Pi { ident: ident2, ty: ty2, body: body2 },
        ) => {
            let ident = fresh_common(ctx, ident1.clone(), ident2.clone());
            let ty = merge_into(ctx, ty1.clone(), ty2.clone())?;
            let body1 = substitute(ctx, body1.clone(), ident1.clone(), local(ident.clone()));
            let body2 = substitute(ctx, body2.clone(), ident2.clone(), local(ident.clone()));
            ctx.with_local(ident.clone(), local(ident.clone()), ty.clone(), |ctx| {
                let body = merge_into(ctx, body1, body2)?;
                Ok(Rc::new(Term::Pi { ident, ty, body }))
            })
        }

        (
            Term::Lam { ident: ident1, ty: ty1, body: body1 },
            Term::Lam { ident: ident2, ty: ty2, body: body2 },
        ) => {
            let ident = fresh_common(ctx, ident1.clone(), ident2.clone());
            let ty = merge_into(ctx, ty1.clone(), ty2.clone())?;
            let body1 = substitute(ctx, body1.clone(), ident1.clone(), local(ident.clone()));
            let body2 = substitute(ctx, body2.clone(), ident2.clone(), local(ident.clone()));
            ctx.with_local(ident.clone(), local(ident.clone()), ty.clone(), |ctx| {
                let body = merge_into(ctx, body1, body2)?;
                Ok(Rc::new(Term::Lam { ident, ty, body }))
            })
        }

        (
            Term::Match { obj: obj1, default: default1, cases: cases1 },
            Term::Match { obj: obj2, default: default2, cases: cases2 },
        ) => {
            let obj = unify(ctx, obj1.clone(), obj2.clone())
                .with_comment("Only match expressions acting on the same object can be unified")?;

            let mut cases = Vec::new();
            let mut cases2 = cases2.clone();

            for (pat, body) in cases1 {
                if !is_concrete(pat) {
                    return err(ElaborateError::Commented(
                        ctx.span,
                        format!("Cannot merge match expression with non-concrete pattern `{}`", pat),
                    ));
                }

                let mut found = false;

                for (i, (pat2, body2)) in cases2.iter_mut().enumerate() {
                    if !is_concrete(pat2) {
                        return err(ElaborateError::Commented(
                            ctx.span,
                            format!("Cannot merge match expression with non-concrete pattern `{}`", pat2),
                        ));
                    }

                    if unify(ctx, pat.clone(), pat2.clone()).is_ok() {
                        ctx.with_case(obj.clone(), pat.clone(), |ctx| {
                            let body = merge_into(ctx, body.clone(), body2.clone())?;
                            cases.push((pat.clone(), body));
                            found = true;
                            Result::Ok(())
                        })?;
                        if found {
                            cases2.remove(i);
                            break;
                        }
                    }
                }

                if !found {
                    cases.push((pat.clone(), body.clone()));
                }
            }

            if let Term::Undef = &**default1 {
                cases.append(&mut cases2);
                Ok(Rc::new(Term::Match {
                    obj,
                    cases,
                    default: default2.clone(),
                }))
            } else {
                /*
                let new_match = Rc::new(Term::Match {
                    obj: obj.clone(),
                    cases: cases2,
                    default: default2.clone(),
                });
                Ok(Rc::new(Term::Match {
                    obj,
                    cases,
                    default: merge_into(ctx, default1.clone(), new_match)?,
                }))
                 */
                // TODO: not ideal, but doesn't overflow the stack
                cases.append(&mut cases2);
                Ok(Rc::new(Term::Match {
                    obj,
                    cases,
                    default: default1.clone(),
                }))
            }
        }

        (
            Term::Match { obj, cases, default },
            _,
        ) => {
            Ok(Rc::new(Term::Match {
                obj: obj.clone(),
                cases: cases.clone(),
                default: merge_into(ctx, default.clone(), add)?,
            }))
        }

        (
            _,
            Term::Match { obj, cases, default },
        ) => {
            Ok(Rc::new(Term::Match {
                obj: obj.clone(),
                cases: cases.clone(),
                default: merge_into(ctx, term, default.clone())?,
            }))
        }

        _ => unify(ctx, term, add).with_comment("Cannot merge these terms"),
    }
}

fn is_concrete(term: &Rc<Term>) -> bool {
    match &**term {
        Term::Var(_) => false,
        Term::Meta(_) => false,
        Term::Undef => true,
        Term::Int(_) => true,
        Term::String(_) => true,
        Term::Bool(_) => true,
        Term::Unit => true,
        Term::Type(indicators) => indicators.iter().all(is_concrete),
        Term::IntTy => true,
        Term::StringTy => true,
        Term::BoolTy => true,
        Term::UnitTy => true,
        Term::Fn { body, args } =>
            is_concrete(body) && args.iter().all(|(_, ty)| is_concrete(ty)),
        Term::Call { obj, args } =>
            is_concrete(obj) && args.iter().all(is_concrete),
        Term::FnTy { args, ret_ty } =>
            args.iter().all(is_concrete) && is_concrete(ret_ty),
        Term::Source { term, .. } => is_concrete(term),
        Term::Pi { body, ty, .. } => is_concrete(body) && is_concrete(ty),
        Term::Lam { body, ty, .. } => is_concrete(body) && is_concrete(ty),
        Term::App { obj, arg } => is_concrete(obj) && is_concrete(arg),
        Term::Match { .. } => false,
        Term::Decl { ty, val, .. } => is_concrete(ty) && is_concrete(val),
        Term::Block { stmts, ret } => stmts.iter().all(is_concrete) && is_concrete(ret),
        Term::Typed { ty, term } => is_concrete(ty) && is_concrete(term),
        Term::If { cond, then, else_ } => is_concrete(cond) && is_concrete(then) && is_concrete(else_),
        Term::Intrinsic(_) => true,
        Term::Assign { rhs, .. } => is_concrete(rhs),
        Term::While { .. } => false,
    }
}

fn contains_metas(term: &Rc<Term>) -> bool {
    match &**term {
        Term::Var(_) => false,
        Term::Meta(_) => true,
        Term::Undef => false,
        Term::Int(_) => false,
        Term::String(_) => false,
        Term::Bool(_) => false,
        Term::Unit => false,
        Term::Type(indicators) => indicators.iter().any(contains_metas),
        Term::IntTy => false,
        Term::StringTy => false,
        Term::BoolTy => false,
        Term::UnitTy => false,
        Term::Fn { body, args } =>
            contains_metas(body) ||
                args.iter().any(|(_, ty)| contains_metas(ty)),
        Term::FnTy { args, ret_ty } =>
            args.iter().any(contains_metas) ||
                contains_metas(ret_ty),
        Term::Call { obj, args } =>
            contains_metas(obj) ||
                args.iter().any(contains_metas),
        Term::Pi { body, ty, .. } => contains_metas(body) || contains_metas(ty),
        Term::Lam { body, ty, .. } => contains_metas(body) || contains_metas(ty),
        Term::App { obj, arg } => contains_metas(obj) || contains_metas(arg),
        Term::Match { obj, cases, default } =>
            contains_metas(obj) ||
                cases.iter().any(|(pat, body)| contains_metas(pat) || contains_metas(body)) ||
                contains_metas(default),
        Term::Source { term, .. } => contains_metas(term),
        Term::Decl { ty, val, .. } => contains_metas(ty) || contains_metas(val),
        Term::Block { stmts, ret } => stmts.iter().any(contains_metas) || contains_metas(ret),
        Term::Typed { ty, term } => contains_metas(ty) || contains_metas(term),
        Term::If { cond, then, else_ } => contains_metas(cond) || contains_metas(then) || contains_metas(else_),
        Term::Intrinsic(_) => false,
        Term::Assign { rhs, .. } => contains_metas(rhs),
        Term::While { cond, body } => contains_metas(cond) || contains_metas(body),
    }
}

fn nf(ctx: &mut Ctx, term: Rc<Term>) -> Result<Rc<Term>> {
    match &*term {
        Term::Var(ident) => {
            if let Some((value, _)) = ctx.lookup(ident.clone()) {
                if let Term::Var(ident2) = &*value {
                    if ident == ident2 {
                        return Ok(value);
                    }
                }
                nf(ctx, value)
            } else {
                Ok(term)
            }
        }
        Term::Meta(id) => {
            if let Some(value) = &ctx.metas[*id].0 {
                nf(ctx, value.clone())
            } else {
                Ok(term)
            }
        }
        Term::Type(indicators) => {
            Ok(Rc::new(Term::Type(
                indicators.iter().map(|i| nf(ctx, i.clone())).collect::<Result<_>>()?
            )))
        }
        Term::Pi { ident, ty, body } => {
            let ty = nf(ctx, ty.clone())?;
            ctx.with_local(ident.clone(), local(ident.clone()), ty.clone(), |ctx| {
                let body = nf(ctx, body.clone())?;
                Ok(Rc::new(Term::Pi { ident: ident.clone(), ty, body }))
            })
        }
        Term::Lam { ident, ty, body } => {
            let ty = nf(ctx, ty.clone())?;
            ctx.with_local(ident.clone(), local(ident.clone()), ty.clone(), |ctx| {
                let body = nf(ctx, body.clone())?;
                Ok(Rc::new(Term::Lam { ident: ident.clone(), ty, body }))
            })
        }
        Term::App { obj, arg } => {
            let obj = nf(ctx, obj.clone())?;
            let arg = nf(ctx, arg.clone())?;
            match &*obj {
                Term::Lam { ident, ty: _, body } => {
                    let body = substitute(ctx, body.clone(), ident.clone(), arg.clone());
                    let result = nf(ctx, body)?;
                    if let Term::Undef = &*result {
                        err(ElaborateError::Commented(
                            ctx.span,
                            format!("Value `{}` evaluated to undefined", term),
                        ))
                    } else {
                        Ok(result)
                    }
                }
                Term::Pi { ident, ty: _, body } => {
                    let body = substitute(ctx, body.clone(), ident.clone(), arg.clone());
                    let result = nf(ctx, body)?;
                    if let Term::Undef = &*result {
                        err(ElaborateError::Commented(
                            ctx.span,
                            format!("Type `{}` evaluated to undefined", term),
                        ))
                    } else {
                        Ok(result)
                    }
                }
                Term::Match { cases, default, .. } => {
                    let cases = cases.iter().map(|(pat, body)| {
                        ctx.with_case(obj.clone(), pat.clone(), |ctx| {
                            Ok((pat.clone(), nf(ctx, Rc::new(Term::App {
                                obj: body.clone(),
                                arg: arg.clone(),
                            })).unwrap_or(ctx.undef.clone())))
                        })
                    }).collect::<Result<Vec<_>>>()?;
                    let default = nf(ctx, Rc::new(Term::App {
                        obj: default.clone(),
                        arg: arg.clone(),
                    })).unwrap_or(ctx.undef.clone());
                    Ok(Rc::new(Term::Match { obj, cases, default }))
                }
                _ => Ok(Rc::new(Term::App { obj, arg })),
            }
        }
        Term::Match { obj, cases, default } => {
            let obj = nf(ctx, obj.clone())?;
            let cases = cases.iter().map(|(pat, body)| {
                ctx.with_case(obj.clone(), pat.clone(), |ctx| {
                    let pat = nf(ctx, pat.clone())?;
                    let body = nf(ctx, body.clone())?;
                    Ok((pat, body))
                })
            }).collect::<Result<Vec<_>>>()?;
            let default = nf(ctx, default.clone())?;
            let mut all_applicable = true;
            for (pat, body) in &cases {
                let applicable = is_concrete(&obj) && is_concrete(pat);
                if applicable {
                    if unify(ctx, obj.clone(), pat.clone()).is_ok() {
                        return nf(ctx, body.clone());
                    }
                } else {
                    all_applicable = false;
                }
            }
            Ok(if all_applicable {
                default
            } else {
                Rc::new(Term::Match { obj, cases, default })
            })
        }
        Term::Source { term, span } => {
            ctx.with_span(*span, |ctx| nf(ctx, term.clone()))
        }
        Term::Typed { ty, term } => {
            let ty = nf(ctx, ty.clone())?;
            let term = nf(ctx, term.clone())?;
            Ok(Rc::new(Term::Typed { ty, term }))
        }
        Term::Call { obj, args } => {
            let obj = nf(ctx, obj.clone())?;
            let args = args.iter().map(|arg| nf(ctx, arg.clone())).collect::<Result<Vec<_>>>()?;
            match &*obj {
                Term::Fn { args: args_decl, body } => {
                    if args.iter().all(is_concrete) {
                        let locals = args.iter().zip(args_decl)
                            .map(|(val, (ident, ty))| {
                                (ident.clone(), (val.clone(), ty.clone()))
                            }).collect::<Vec<_>>();
                        ctx.with_locals(&locals, |ctx| {
                            nf(ctx, body.clone())
                        })
                    } else {
                        Ok(Rc::new(Term::Call { obj, args }))
                    }
                }
                Term::Intrinsic(intrinsic) => {
                    if intrinsic.can_call() && args.iter().all(is_concrete) {
                        intrinsic.call(args)
                    } else {
                        Ok(Rc::new(Term::Call { obj, args }))
                    }
                }
                Term::Match { obj, cases, default } => {
                    let cases = cases.iter().map(|(pat, body)| {
                        ctx.with_case(obj.clone(), pat.clone(), |ctx| {
                            Ok((pat.clone(), nf(ctx, Rc::new(Term::Call {
                                obj: body.clone(),
                                args: args.clone(),
                            })).unwrap_or(ctx.undef.clone())))
                        })
                    }).collect::<Result<Vec<_>>>()?;
                    let default = nf(ctx, Rc::new(Term::Call {
                        obj: default.clone(),
                        args: args.clone(),
                    })).unwrap_or(ctx.undef.clone());
                    Ok(Rc::new(Term::Match { obj: obj.clone(), cases, default }))
                }
                _ => Ok(Rc::new(Term::Call { obj, args })),
            }
        }
        Term::Fn { args, body } => {
            let args = args.iter().map(|(ident, ty)| {
                let ty = nf(ctx, ty.clone())?;
                Ok((ident.clone(), ty))
            }).collect::<Result<Vec<_>>>()?;
            let body = ctx.with_local_decls(&args, |ctx| nf(ctx, body.clone()))?;
            Ok(Rc::new(Term::Fn { args, body }))
        }
        Term::FnTy { args, ret_ty } => {
            let args = args.iter().map(|ty| nf(ctx, ty.clone())).collect::<Result<Vec<_>>>()?;
            let ret_ty = nf(ctx, ret_ty.clone())?;
            Ok(Rc::new(Term::FnTy { args, ret_ty }))
        }
        Term::Decl { val, ty, name } => {
            let val = nf(ctx, val.clone())?;
            let ty = nf(ctx, ty.clone())?;
            Ok(Rc::new(Term::Decl { val, ty, name: name.clone() }))
        }
        Term::Block { stmts, ret } => {
            ctx.with_block(|ctx| {
                let stmts = stmts.iter().map(
                    |stmt| {
                        let res= nf(ctx, stmt.clone())?;
                        ctx.consider(&res);
                        Ok(res)
                    }).collect::<Result<Vec<_>>>()?;

                let ret = nf(ctx, ret.clone())?;

                Ok(Rc::new(Term::Block { stmts, ret }))
            })
        }
        Term::If { cond, then, else_ } => {
            let cond = nf(ctx, cond.clone())?;
            let then = nf(ctx, then.clone())?;
            let else_ = nf(ctx, else_.clone())?;
            if is_concrete(&cond) {
                if let Term::Bool(true) = &*cond {
                    return nf(ctx, then);
                } else if let Term::Bool(false) = &*cond {
                    return nf(ctx, else_);
                } else {
                    panic!("Non-boolean condition in if expression");
                }
            } else {
                Ok(Rc::new(Term::If { cond, then, else_ }))
            }
        }
        Term::Assign { lhs, rhs } => {
            let rhs = nf(ctx, rhs.clone())?;
            Ok(Rc::new(Term::Assign { lhs: lhs.clone(), rhs }))
        }
        Term::While { cond, body } => {
            let cond = nf(ctx, cond.clone())?;
            let body = nf(ctx, body.clone())?;
            Ok(Rc::new(Term::While { cond, body }))
        }

        Term::Undef => Ok(term),
        Term::Int(_) => Ok(term),
        Term::String(_) => Ok(term),
        Term::Bool(_) => Ok(term),
        Term::Unit => Ok(term),
        Term::IntTy => Ok(term),
        Term::StringTy => Ok(term),
        Term::BoolTy => Ok(term),
        Term::UnitTy => Ok(term),
        Term::Intrinsic(_) => Ok(term),
    }
}

fn substitute(ctx: &mut Ctx, term: Rc<Term>, name: SmolStr2, subs: Rc<Term>) -> Rc<Term> {
    match &*term {
        Term::Var(ident) if ident == &name => subs,
        Term::Lam { ident, ty, body } if ident != &name => {
            if free_vars(ctx, subs.clone()).contains(ident) {
                let new_name = ctx.fresh(ident.clone());
                let new_ident = Rc::new(Term::Var(new_name.clone()));
                let ty = substitute(ctx, ty.clone(), name.clone(), subs.clone());
                ctx.with_local(new_name.clone(), ctx.undef.clone(), ctx.undef.clone(), |ctx| {
                    let body = substitute(ctx, body.clone(), ident.clone(), new_ident.clone());
                    let body = substitute(ctx, body, name, subs);
                    Rc::new(Term::Lam { ident: new_name, ty, body })
                })
            } else {
                let ty = substitute(ctx, ty.clone(), name.clone(), subs.clone());
                ctx.with_local(ident.clone(), ctx.undef.clone(), ctx.undef.clone(), |ctx| {
                    let body = substitute(ctx, body.clone(), name, subs);
                    Rc::new(Term::Lam { ident: ident.clone(), ty, body })
                })
            }
        }
        Term::Pi { ident, ty, body } if ident != &name => {
            if free_vars(ctx, subs.clone()).contains(ident) {
                let new_name = ctx.fresh(ident.clone());
                let new_ident = Rc::new(Term::Var(new_name.clone()));
                let ty = substitute(ctx, ty.clone(), name.clone(), subs.clone());
                ctx.with_local(new_name.clone(), ctx.undef.clone(), ctx.undef.clone(), |ctx| {
                    let body = substitute(ctx, body.clone(), ident.clone(), new_ident.clone());
                    let body = substitute(ctx, body, name, subs);
                    Rc::new(Term::Pi { ident: new_name, ty, body })
                })
            } else {
                let ty = substitute(ctx, ty.clone(), name.clone(), subs.clone());
                ctx.with_local(ident.clone(), ctx.undef.clone(), ctx.undef.clone(), |ctx| {
                    let body = substitute(ctx, body.clone(), name, subs);
                    Rc::new(Term::Pi { ident: ident.clone(), ty, body })
                })
            }
        }
        Term::App { obj, arg } => {
            let obj = substitute(ctx, obj.clone(), name.clone(), subs.clone());
            let arg = substitute(ctx, arg.clone(), name, subs);
            Rc::new(Term::App { obj, arg })
        }
        Term::Match { obj, cases, default } => {
            let obj = substitute(ctx, obj.clone(), name.clone(), subs.clone());
            let cases = cases.iter().map(|(pat, body)| {
                let pat = substitute(ctx, pat.clone(), name.clone(), subs.clone());
                let body = substitute(ctx, body.clone(), name.clone(), subs.clone());
                (pat, body)
            }).collect::<Vec<_>>();
            let default = substitute(ctx, default.clone(), name, subs.clone());
            Rc::new(Term::Match { obj, cases, default })
        }
        Term::Fn { args, body } => {
            let args = args.iter().map(|(ident, ty)| {
                let ty = substitute(ctx, ty.clone(), name.clone(), subs.clone());
                (ident.clone(), ty)
            }).collect::<Vec<_>>();
            if args.iter().any(|(ident, _)| ident == &name) {
                Rc::new(Term::Fn { args, body: body.clone() })
            } else {
                let body = substitute(ctx, body.clone(), name, subs);
                Rc::new(Term::Fn { args, body })
            }
        }
        Term::FnTy { args, ret_ty } => {
            let args = args.iter().map(|ty| substitute(ctx, ty.clone(), name.clone(), subs.clone())).collect();
            let ret_ty = substitute(ctx, ret_ty.clone(), name, subs);
            Rc::new(Term::FnTy { args, ret_ty })
        }
        Term::Call { obj, args } => {
            let obj = substitute(ctx, obj.clone(), name.clone(), subs.clone());
            let args = args.iter().map(|arg| substitute(ctx, arg.clone(), name.clone(), subs.clone())).collect();
            Rc::new(Term::Call { obj, args })
        }
        Term::Source { term, .. } => substitute(ctx, term.clone(), name, subs),
        Term::Decl { val, ty, name: ident } => {
            let val = substitute(ctx, val.clone(), name.clone(), subs.clone());
            let ty = substitute(ctx, ty.clone(), name.clone(), subs);
            if ident == &name {
                ctx.stop_substitution_mark = true;
            }
            Rc::new(Term::Decl { name: ident.clone(), val, ty })
        }
        Term::Block { stmts, ret } => {
            ctx.with_block(|ctx| {
                let stmts = stmts.iter().map(|stmt| {
                    if !ctx.stop_substitution_mark {
                        let stmt = substitute(ctx, stmt.clone(), name.clone(), subs.clone());
                        ctx.consider(&stmt);
                        stmt
                    } else {
                        stmt.clone()
                    }
                }).collect();
                if !ctx.stop_substitution_mark {
                    let ret = substitute(ctx, ret.clone(), name, subs);
                    Rc::new(Term::Block { stmts, ret })
                } else {
                    Rc::new(Term::Block { stmts, ret: ret.clone() })
                }
            })
        },
        Term::If { cond, then, else_ } => {
            let cond = substitute(ctx, cond.clone(), name.clone(), subs.clone());
            let then = substitute(ctx, then.clone(), name.clone(), subs.clone());
            let else_ = substitute(ctx, else_.clone(), name, subs);
            Rc::new(Term::If { cond, then, else_ })
        }
        Term::Assign { lhs, rhs } => {
            let rhs = substitute(ctx, rhs.clone(), name, subs);
            Rc::new(Term::Assign { lhs: lhs.clone(), rhs })
        }
        _ => term,
    }
}

fn free_vars(ctx: &mut Ctx, term: Rc<Term>) -> Vec<SmolStr2> {
    match &*term {
        Term::Var(ident) => vec![ident.clone()],
        Term::Lam { ident, body, .. } => {
            let mut free = free_vars(ctx, body.clone());
            free.retain(|x| x != ident);
            free
        }
        Term::Pi { ident, body, .. } => {
            let mut free = free_vars(ctx, body.clone());
            free.retain(|x| x != ident);
            free
        }
        Term::App { obj, arg } => {
            let mut free = free_vars(ctx, obj.clone());
            free.append(&mut free_vars(ctx, arg.clone()));
            free
        }
        Term::Match { obj, cases, default } => {
            let mut free = free_vars(ctx, obj.clone());
            for (pat, body) in cases {
                free.append(&mut free_vars(ctx, pat.clone()));
                free.append(&mut free_vars(ctx, body.clone()));
            }
            free.append(&mut free_vars(ctx, default.clone()));
            free
        }
        Term::Source { term, .. } => free_vars(ctx, term.clone()),
        Term::Fn { args, body } => {
            let mut free = free_vars(ctx, body.clone());
            for (_, ty) in args {
                free.append(&mut free_vars(ctx, ty.clone()));
            }
            free.retain(|x| args.iter().all(|(ident, _)| ident != x));
            free
        }
        Term::FnTy { args, ret_ty } => {
            let mut free = free_vars(ctx, ret_ty.clone());
            for ty in args {
                free.append(&mut free_vars(ctx, ty.clone()));
            }
            free
        }
        Term::Call { obj, args } => {
            let mut free = free_vars(ctx, obj.clone());
            for arg in args {
                free.append(&mut free_vars(ctx, arg.clone()));
            }
            free
        }
        Term::Decl { val, ty, .. } => {
            let mut free = free_vars(ctx, val.clone());
            free.append(&mut free_vars(ctx, ty.clone()));
            free
        }
        Term::Block { stmts, ret } => {
            let mut free = stmts.iter().flat_map(|stmt| free_vars(ctx, stmt.clone())).collect::<Vec<_>>();
            free.append(&mut free_vars(ctx, ret.clone()));
            free
        }
        Term::If { cond, then, else_ } => {
            let mut free = free_vars(ctx, cond.clone());
            free.append(&mut free_vars(ctx, then.clone()));
            free.append(&mut free_vars(ctx, else_.clone()));
            free
        }
        Term::Assign { rhs, .. } => free_vars(ctx, rhs.clone()),

        _ => Vec::new(),
    }
}

thread_local! {
    static INDENT: std::cell::RefCell<usize> = const { std::cell::RefCell::new(0) };
}

impl Display for Term {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if let Some(width) = f.width() {
            INDENT.with(|i| *i.borrow_mut() = width);
        }
        let indent = INDENT.with(|i| *i.borrow());
        match self {
            Term::Var(name) => write!(f, "{}", &name[..]),
            Term::Meta(id) => write!(f, "'{}", id),
            Term::Undef => write!(f, "---"),
            Term::Int(x) => write!(f, "{}", x),
            Term::String(s) => write!(f, "\"{}\"", &s[..]),
            Term::Bool(b) => write!(f, "{}", b),
            Term::Unit => write!(f, "()"),
            Term::Type(indicators) => if indicators.is_empty() {
                write!(f, "type")
            } else {
                let indicators = indicators.iter().map(|i| format!("{}", i)).collect::<Vec<_>>();
                write!(f, "type[{}]", indicators.join(", "))
            },
            Term::IntTy => write!(f, "int"),
            Term::StringTy => write!(f, "string"),
            Term::BoolTy => write!(f, "bool"),
            Term::UnitTy => write!(f, "unit"),
            Term::Fn { args, body } => {
                write!(f, "fn(")?;
                for (ident, ty) in args {
                    write!(f, "{}: {},", ident, ty)?;
                }
                write!(f, ") {}", body)
            },
            Term::FnTy { args, ret_ty } => {
                write!(f, "Fn(")?;
                for ty in args {
                    write!(f, "{},", ty)?;
                }
                write!(f, ") -> {}", ret_ty)
            }
            Term::Call { obj, args } => {
                let args = args.iter().map(|i| format!("{}", i)).collect::<Vec<_>>();
                write!(f, "{}({})", obj, args.join(", "))
            }
            Term::Pi { ident, body, ty } => {
                write!(f, "∀{}: {}. {}", &ident[..], ty, body)
            }
            Term::Lam { ident, body, ty } => {
                write!(f, "λ{}: {}. {}", &ident[..], ty, body)
            }
            Term::App { obj, arg } => {
                write!(f, "({})[{}]", obj, arg)
            }
            Term::Match { obj, cases, default } => {
                INDENT.with(|i| *i.borrow_mut() += 4);
                writeln!(f, "match {} {{", obj)?;
                for (pat, body) in cases {
                    writeln!(f, "{1:0$}{2} => {3}", indent + 4, "", pat, body)?;
                }
                writeln!(f, "{1:0$}_ => {2}", indent + 4, "", default)?;
                INDENT.with(|i| *i.borrow_mut() -= 4);
                write!(f, "{1:0$}}}", indent, "")
            }
            Term::Source { term, .. } => write!(f, "{}", term),
            Term::Decl { name, val, ty } => write!(f, "{}: {} = {}", &name[..], ty, val),
            Term::Block { stmts, ret } => {
                writeln!(f, "{{")?;
                INDENT.with(|i| *i.borrow_mut() += 4);
                for stmt in stmts {
                    writeln!(f, "{1:0$}{2};", indent + 4, "", stmt)?;
                }
                writeln!(f, "{1:0$}{2}", indent + 4, "", ret)?;
                INDENT.with(|i| *i.borrow_mut() -= 4);
                write!(f, "{1:0$}}}", indent, "")
            }
            Term::Assign { lhs, rhs } => write!(f, "{} = {}", lhs, rhs),
            Term::Typed { term, ty } => write!(f, "({})::{}", term, ty),
            Term::Intrinsic(intrinsic) => write!(f, "<{}>", &intrinsic.name()[..]),
            Term::If { cond, then, else_ } => write!(f, "if {} then {} else {}", cond, then, else_),
            Term::While { cond, body } => write!(f, "while {} do {}", cond, body),
        }
    }
}

struct Builder {
    includes: Vec<SmolStr2>,
    structs: Vec<c::Struct>,
    functions: Vec<c::Function>,
    externals: Vec<c::ExternalFunction>,
    added_externals: HashMap<SmolStr2, usize>,
    function_cache: Vec<(Rc<Term>, usize)>,
}

struct FnBuilder {
    locals: Vec<c::CType>,
}

struct BlockBuilder {
    locals: HashMap<SmolStr2, usize>,
    statements: Vec<c::Statement>,
}

pub struct Extraction<'a> {
    pub ctx: &'a mut Ctx,
    b: &'a mut Builder,
    fb: &'a mut FnBuilder,
    bb: &'a mut Vec<BlockBuilder>,
}

impl<'a> Extraction<'a> {
    fn lookup_local(&self, name: &SmolStr2) -> Option<usize> {
        self.bb.iter().rev().find_map(|bb| bb.locals.get(name).copied())
    }

    pub fn add_external(&mut self, name: SmolStr2, init: impl FnOnce() -> (Vec<SmolStr2>, c::ExternalFunction)) -> usize {
        if let Some(id) = self.b.added_externals.get(&name) {
            *id
        } else {
            let (includes, external) = init();
            let id = self.b.externals.len();
            self.b.externals.push(external);
            self.b.added_externals.insert(name, id);
            for include in includes {
                if !self.b.includes.contains(&include) {
                    self.b.includes.push(include);
                }
            }

            id
        }
    }

    fn temp(&mut self, ty: c::CType) -> usize {
        let id = self.fb.locals.len();
        self.fb.locals.push(ty);
        id
    }

    fn new_local(&mut self, name: SmolStr2, ty: c::CType) -> usize {
        let id = self.temp(ty);
        self.bb.last_mut().unwrap().locals.insert(name, id);
        id
    }

    fn add_statement(&mut self, statement: c::Statement) {
        self.bb.last_mut().unwrap().statements.push(statement);
    }

    fn add_expr(&mut self, ty: c::CType, expr: Box<c::Expr>) -> c::Expr {
        let id = self.temp(ty);
        self.add_statement(c::Statement::Expression(c::Expr::Assign {
            lhs: id,
            rhs: expr,
        }));
        c::Expr::Local { id }
    }

    fn with_block<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        self.bb.push(BlockBuilder {
            locals: HashMap::new(),
            statements: vec![],
        });
        let res = f(self);
        let bb = self.bb.pop().unwrap();
        for stmt in bb.statements {
            self.add_statement(stmt);
        }
        res
    }

    fn build_block<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> (T, Vec<c::Statement>) {
        self.bb.push(BlockBuilder {
            locals: HashMap::new(),
            statements: vec![],
        });
        let res = f(self);
        let statements = self.bb.pop().unwrap().statements;
        (res, statements)
    }
}

pub fn extract_c(ctx: &mut Ctx) -> Result<c::Module> {
    let mut b = Builder {
        includes: vec![],
        structs: vec![],
        functions: vec![],
        externals: vec![],
        function_cache: vec![],
        added_externals: HashMap::new(),
    };

    let main = ctx.lookup("main".into()).ok_or_else(||
        vec![ElaborateError::Commented(
            ctx.span,
            "No main function found".into(),
        )]
    )?.0;

    let main_id = extract_fn(ctx, &mut b, main)?;

    Ok(c::Module {
        includes: b.includes,
        structs: b.structs,
        functions: b.functions,
        externals: b.externals,
        main: Some(main_id),
    })
}

fn extract_fn(ctx: &mut Ctx, b: &mut Builder, term: Rc<Term>) -> Result<usize> {
    let id = b.function_cache.iter().position(|(t, _)| **t == *term);
    if let Some(id) = id {
        return Ok(id);
    }

    let ty = type_of(ctx, term.clone())?;

    let Term::FnTy { ret_ty, .. } = &*ty else {
        return err(ElaborateError::Commented(
            ctx.span,
            format!("Expected function type, found `{}`", ty),
        ));
    };

    let id = b.functions.len();

    b.function_cache.push((term.clone(), id));

    let return_type = extract_c_type(ctx, b, &ret_ty)?;
    let zero_sized_return = return_type.is_none();
    let return_type = return_type.unwrap_or(c::CType::Void);

    let Term::Fn { args, body } = &*term else {
        return err(ElaborateError::Commented(
            ctx.span,
            format!("Expected function, found `{}`", term),
        ));
    };

    let locals = args.iter()
        .flat_map(|(_, ty)| extract_c_type(ctx, b, ty).transpose())
        .collect::<Result<Vec<_>>>()?;

    let parameters = (0..locals.len()).collect();

    b.functions.push(c::Function {
        parameters,
        body: vec![],
        locals: vec![],
        return_type,
    });

    let mut fb = FnBuilder {
        locals,
    };

    let mut bb = vec![BlockBuilder {
        locals: {
            let mut map = HashMap::new();
            for (i, (name, _)) in args.iter().enumerate() {
                map.insert(name.clone(), i);
            }
            map
        },
        statements: vec![],
    }];

    let mut e = Extraction {
        ctx,
        b,
        fb: &mut fb,
        bb: &mut bb,
    };

    let expr = extract_c_expr(&mut e, body.clone())?;

    if let Some(expr) = expr {
        if !zero_sized_return {
            e.add_statement(c::Statement::Return(expr));
        } else {
            e.add_statement(c::Statement::Expression(expr));
        }
    }

    assert_eq!(1, e.bb.len());
    b.functions[id].body = e.bb.pop().unwrap().statements;
    b.functions[id].locals = fb.locals;

    Ok(id)
}

fn extract_c_type(ctx: &mut Ctx, b: &mut Builder, term: &Term) -> Result<Option<c::CType>> {
    Ok(match term {
        Term::Undef => return err(ElaborateError::Commented(
            ctx.span,
            "Cannot extract C type from undefined term".into(),
        )),
        Term::Type(_) => None,
        Term::IntTy => Some(c::CType::Int),
        Term::StringTy => Some(c::CType::Pointer(Box::new(c::CType::Char))),
        Term::BoolTy => Some(c::CType::Int),
        Term::UnitTy => None,
        Term::Intrinsic(_) => None,
        Term::Source { term, .. } => return extract_c_type(ctx, b, term),
        Term::Decl { .. } => None,
        Term::Block { ret, .. } => return extract_c_type(ctx, b, ret),
        Term::Typed { term, .. } => return extract_c_type(ctx, b, term),
        _ => return err(ElaborateError::Commented(
            ctx.span,
            format!("Cannot extract C type from term `{}`", term),
        )),
    })
}

fn extract_c_expr(e: &mut Extraction, term: Rc<Term>) -> Result<Option<c::Expr>> {
    match &*term {
        Term::Var(name) => {
            if let Some(local) = e.lookup_local(name) {
                Ok(Some(c::Expr::Local {
                    id: local,
                }))
            } else {
                err(ElaborateError::Commented(
                    e.ctx.span,
                    format!("Unknown variable `{}` (if it is global, it must be normalized away)", name),
                ))
            }

        }
        Term::Meta(_) => err(ElaborateError::Commented(
            e.ctx.span,
            "Unsolved meta variables found upon C extraction (THIS IS ICE!)".into(),
        )),
        Term::Undef => Ok(None),
        Term::Int(x) => Ok(Some(c::Expr::Int { x: *x })),
        Term::String(s) => Ok(Some(c::Expr::String { s: s.into() })),
        Term::Bool(x) => Ok(Some(c::Expr::Int { x: if *x { 1 } else { 0 } })),
        Term::Unit => Ok(None),
        Term::Type(_) | Term::IntTy | Term::StringTy | Term::BoolTy
        | Term::UnitTy | Term::FnTy { .. }
            => err(ElaborateError::Commented(
                e.ctx.span,
                format!("Cannot extract C expression from type `{}`", term),
            )),
        Term::Intrinsic(_) | Term::Fn { .. } => err(ElaborateError::Commented(
            e.ctx.span,
            format!("Cannot extract C expression from non-called function `{}`", term),
        )),
        Term::Pi { .. } | Term::Lam { .. } | Term::App { .. } | Term::Match { .. } =>
            err(ElaborateError::Commented(
                e.ctx.span,
                format!("Cannot extract C expression from non-normalized meta-langauge item `{}`", term),
            )),
        Term::Source { term, .. } => {
            extract_c_expr(e, term.clone())
        },
        Term::Typed { term, .. } => {
            extract_c_expr(e, term.clone())
        },

        Term::Call { obj, args } => {
            let args = args.iter().map(|arg|
                extract_c_expr(e, arg.clone())
            ).collect::<Result<Vec<_>>>()?;

            match &**obj {
                Term::Intrinsic(intrinsic) => {
                    if !intrinsic.can_emit() {
                        return err(ElaborateError::Commented(
                            e.ctx.span,
                            format!("Cannot extract C expression from intrinsic `{}`", obj),
                        ));
                    }

                    let (stmts, expr) = intrinsic.emit(
                        e,
                        args.into_iter().flatten().collect()
                    );

                    for stmt in stmts {
                        e.add_statement(stmt);
                    }

                    Ok(expr)
                }

                Term::Fn { .. } => {
                    let function_id = extract_fn(e.ctx, e.b, obj.clone())?;

                    Ok(Some(c::Expr::Call {
                        f: Box::new(c::Expr::Global {
                            id: function_id,
                        }),
                        args: args.into_iter().flatten().collect(),
                    }))
                }

                _ => err(ElaborateError::Commented(
                    e.ctx.span,
                    format!("Cannot extract C expression from non-function `{}`", obj),
                )),
            }
        },

        Term::Decl { ty, val, name } => {
            let ty = extract_c_type(e.ctx, e.b, ty)?;
            let val = extract_c_expr(e, val.clone())?;
            if let Some(ty) = ty {
                let var = e.new_local(name.clone(), ty);
                e.add_statement(c::Statement::Expression(
                    c::Expr::Assign {
                        lhs: var,
                        rhs: Box::new(val.unwrap()),
                    }
                ));
            } else if let Some(expr) = val {
                e.add_statement(c::Statement::Expression(expr));
            }
            Ok(None)
        }
        Term::Block { stmts, ret } => {
            e.with_block(|e| {
                for stmt in stmts {
                    let expr = extract_c_expr(e, stmt.clone())?;

                    if let Some(expr) = expr {
                        e.add_statement(c::Statement::Expression(expr));
                    }
                }

                let ret_ty = type_of(e.ctx, ret.clone())?;
                if let Some(ty) = extract_c_type(e.ctx, e.b, &ret_ty)? {
                    let expr = extract_c_expr(e, ret.clone())?;
                    Ok(Some(e.add_expr(ty, Box::new(expr.unwrap()))))
                } else {
                    let expr = extract_c_expr(e, ret.clone())?;
                    if let Some(expr) = expr {
                        e.add_statement(c::Statement::Expression(expr));
                    }
                    Ok(None)
                }
            })
        }
        Term::If { cond, then, else_ } => {
            let cond = extract_c_expr(e, cond.clone())?.expect("ICE: If condition is not boolean");
            let cond = e.add_expr(c::CType::Int, Box::new(cond));

            let ty = type_of(e.ctx, then.clone())?;
            let ty = extract_c_type(e.ctx, e.b, &ty)?;
            let temp = ty.map(|ty| e.temp(ty));

            let (res, then) = e.build_block(|e| {
                let then = extract_c_expr(e, then.clone())?;
                if let Some(temp) = temp {
                    e.add_statement(c::Statement::Expression(c::Expr::Assign {
                        lhs: temp,
                        rhs: Box::new(then.unwrap()),
                    }));
                } else if let Some(then) = then {
                    e.add_statement(c::Statement::Expression(then));
                }
                Result::Ok(())
            });
            res?;

            let (res, else_) = e.build_block(|e| {
                let else_ = extract_c_expr(e, else_.clone())?;
                if let Some(temp) = temp {
                    e.add_statement(c::Statement::Expression(c::Expr::Assign {
                        lhs: temp,
                        rhs: Box::new(else_.unwrap()),
                    }));
                } else if let Some(else_) = else_ {
                    e.add_statement(c::Statement::Expression(else_));
                }
                Result::Ok(())
            });
            res?;

            e.add_statement(c::Statement::If {
                cond,
                then,
                else_,
            });

            Ok(temp.map(|temp| c::Expr::Local { id: temp }))
        }
        Term::Assign { lhs, rhs } => {
            let var = e.lookup_local(lhs);
            let rhs = extract_c_expr(e, rhs.clone())?;
            if let Some(var) = var {
                e.add_statement(c::Statement::Expression(c::Expr::Assign {
                    lhs: var,
                    rhs: Box::new(rhs.unwrap()),
                }));
            } else {
                e.add_statement(c::Statement::Expression(rhs.unwrap()));
            }
            Ok(None)
        }
        Term::While { cond, body } => {
            let cond = extract_c_expr(e, cond.clone())?
                .expect("ICE: While condition is not boolean");
            
            let (res, body) = e.build_block(|e| {
                let body = extract_c_expr(e, body.clone())?;
                if let Some(body) = body {
                    e.add_statement(c::Statement::Expression(body));
                }
                Result::Ok(())
            });
            res?;

            e.add_statement(c::Statement::While {
                cond,
                body,
            });

            Ok(None)
        }
    }
}
