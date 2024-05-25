use std::fmt::{Display, Formatter};
use std::rc::Rc;
use indexmap::IndexMap;
use serde::Serialize;
use crate::ast::{Ast, Expr, Item, Module, NodeExt, Param};
use crate::cmrg;
use crate::compound_result::{CompoundResult, err};
use crate::intrinsics::make_intrinsics;
use crate::smol_str2::SmolStr2;
use crate::span::Span;

#[derive(Debug, Clone)]
pub enum Term {
    Var(SmolStr2),
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
    Fn { args: Vec<(SmolStr2, Rc<Term>)>, body: Box<Code> },
    Call { obj: Rc<Term>, args: Vec<Rc<Term>> },
    Pi { ty: Rc<Term>, ident: SmolStr2, body: Rc<Term> },
    Lam { ty: Rc<Term>, ident: SmolStr2, body: Rc<Term> },
    App { obj: Rc<Term>, arg: Rc<Term> },
    Match { obj: Rc<Term>, cases: Vec<(Rc<Term>, Rc<Term>)>, default: Rc<Term> },
    Source { span: Span, term: Rc<Term> },
}

#[derive(Debug, Clone)]
pub enum Code {
    Var(SmolStr2),
    Value(Term),
    Call { obj: Box<Code>, name: SmolStr2, args: Vec<Code> },
    Decl { name: SmolStr2, ty: Rc<Term>, val: Box<Code> },
    Block { stmts: Vec<Code>, ret: Box<Code> },
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

struct Ctx {
    star: Rc<Term>,
    undef: Rc<Term>,
    anon: SmolStr2,
    anon_term: Rc<Term>,
    items: IndexMap<SmolStr2, (Rc<Term>, Rc<Term>)>,
    locals: Vec<(SmolStr2, (Rc<Term>, Rc<Term>))>,
    span: Span,
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

pub fn elaborate(module: &Module) -> Result<IndexMap<SmolStr2, (Rc<Term>, Rc<Term>)>> {
    let anon = SmolStr2::from("<anon>");
    let mut ctx = Ctx {
        star: Rc::new(Term::Type(Vec::new())),
        undef: Rc::new(Term::Undef),
        anon: anon.clone(),
        anon_term: Rc::new(Term::Var(anon.clone())),
        items: make_intrinsics(),
        locals: Vec::new(),
        span: Span(0, 0),
    };

    for item in &module.items {
        elaborate_item(&mut ctx, &item.node)?;
    }

    Ok(ctx.items)
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
                        })
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
            },
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
            },
        };

        Ok(result)
    }

    ctx.with_span(item.ident.span, |ctx| {
        let (new, new_ty) = elaborate_item_impl(ctx, item, 0)?;
        let new = nf(ctx, new)?;
        let new_ty = nf(ctx, new_ty)?;

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
        let (value, ty) = (value1, ty1);

        let value = nf(ctx, value)?;
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
            },
            Expr::String(value) => (Term::String(value.clone()), Term::StringTy),
            Expr::Bool(value) => (Term::Bool(*value), Term::BoolTy),
            Expr::Unit => (Term::Unit, Term::UnitTy),
            Expr::PlainType { .. } => (Term::Type(Vec::new()), Term::Type(Vec::new())),
            Expr::Ident(ident) => return ctx.lookup_ident(&ident.clone().spanned(expr.span)),
            Expr::Paren { expr, .. } => return infer(ctx, expr),

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
                return Ok((callee, callee_ty))
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
    eprintln!("Unifying `{}` and `{}`", a, b);
    use Term::*;

    let result = match (&*a, &*b) {
        (_, Undef) => return Ok(a.clone()),
        (Undef, _) => return Ok(b.clone()),
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
        },

        (Pi { ident: ident1, ty: ty1, body: body1 }, Pi { ident: ident2, ty: ty2, body: body2 }) => {
            let ident = fresh_common(ctx, ident1.clone(), ident2.clone());

            let ty = unify(ctx, ty1.clone(), ty2.clone())?;
            let body1 = substitute(ctx, body1.clone(), ident1.clone(), local(ident.clone()));
            let body2 = substitute(ctx, body2.clone(), ident2.clone(), local(ident.clone()));
            let body = unify(ctx, body1, body2)?;

            Pi { ident, ty, body }
        },

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
                default: unify(ctx, a, default.clone())?
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
                default: unify(ctx, default.clone(), b)?
            }
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
                            break
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
        Term::Fn { .. } => false,
        Term::Call { .. } => false,
        Term::Pi { body, ty, .. } => is_concrete(body) && is_concrete(ty),
        Term::Lam { body, ty, .. } => is_concrete(body) && is_concrete(ty),
        Term::App { obj, arg } => is_concrete(obj) && is_concrete(arg),
        Term::Match { .. } => false,
        Term::Source { term, .. } => is_concrete(term),
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
                Term::Match {
                    obj: obj2,
                    cases,
                    default
                } => {
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
        _ => Ok(term),
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
        },
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
        },
        Term::App { obj, arg } => {
            let obj = substitute(ctx, obj.clone(), name.clone(), subs.clone());
            let arg = substitute(ctx, arg.clone(), name, subs);
            Rc::new(Term::App { obj, arg })
        },
        Term::Match { obj, cases, default } => {
           let obj = substitute(ctx, obj.clone(), name.clone(), subs.clone());
           let cases = cases.iter().map(|(pat, body)| {
               let pat = substitute(ctx, pat.clone(), name.clone(), subs.clone());
               let body = substitute(ctx, body.clone(), name.clone(), subs.clone());
               (pat, body)
           }).collect::<Vec<_>>();
           let default = substitute(ctx, default.clone(), name, subs.clone());
           Rc::new(Term::Match { obj, cases, default })
        },
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
        },
        Term::Pi { ident, body, .. } => {
            let mut free = free_vars(ctx, body.clone());
            free.retain(|x| x != ident);
            free
        },
        Term::App { obj, arg } => {
            let mut free = free_vars(ctx, obj.clone());
            free.append(&mut free_vars(ctx, arg.clone()));
            free
        },
        Term::Match { obj, cases, default } => {
            let mut free = free_vars(ctx, obj.clone());
            for (pat, body) in cases {
                free.append(&mut free_vars(ctx, pat.clone()));
                free.append(&mut free_vars(ctx, body.clone()));
            }
            free.append(&mut free_vars(ctx, default.clone()));
            free
        }
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
            Term::Fn { .. } => write!(f, "<function>"),
            Term::Call { obj, args} => {
                let args = args.iter().map(|i| format!("{}", i)).collect::<Vec<_>>();
                write!(f, "{}[{}]", obj, args.join(", "))
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
        }
    }
}
