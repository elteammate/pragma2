use std::fmt::{Display, Formatter};
use std::rc::Rc;
use indexmap::IndexMap;
use serde::Serialize;
use crate::ast::{Ast, BinaryOp, Expr, Item, Module, NodeExt, Param};
use crate::cmrg;
use crate::compound_result::{CompoundResult, err};
use crate::intrinsics::{Intrinsic, make_intrinsics};
use crate::smol_str2::SmolStr2;
use crate::span::Span;

#[derive(Debug)]
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
    Intrinsic(Box<dyn Intrinsic>),
    Fn { args: Vec<(SmolStr2, Rc<Term>)>, body: Rc<Term> },
    FnTy { args: Vec<(Rc<Term>)>, ret_ty: Rc<Term> },
    Call { obj: Rc<Term>, args: Vec<Rc<Term>> },
    Pi { ty: Rc<Term>, ident: SmolStr2, body: Rc<Term> },
    Lam { ty: Rc<Term>, ident: SmolStr2, body: Rc<Term> },
    App { obj: Rc<Term>, arg: Rc<Term> },
    Match { obj: Rc<Term>, cases: Vec<(Rc<Term>, Rc<Term>)>, default: Rc<Term> },
    Source { span: Span, term: Rc<Term> },
    Decl { name: SmolStr2, ty: Rc<Term>, val: Rc<Term> },
    Block { stmts: Vec<Rc<Term>>, ret: Rc<Term> },
    Typed { term: Rc<Term>, ty: Rc<Term> },
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
    unit: Rc<Term>,
    unit_ty: Rc<Term>,
    items: IndexMap<SmolStr2, (Rc<Term>, Rc<Term>)>,
    locals: Vec<(SmolStr2, (Rc<Term>, Rc<Term>))>,
    span: Span,
    metas: Vec<(Option<Rc<Term>>, Rc<Term>)>,
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

fn type_of(ctx: &mut Ctx, term: Rc<Term>) -> Rc<Term> {
    match &*term {
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
            let obj_ty = type_of(ctx, obj.clone());
            let obj_ty = nf(ctx, obj_ty).expect("Typed code failed to normalize");
            match &*obj_ty {
                Term::FnTy { ret_ty, .. } => ret_ty.clone(),
                _ => panic!("Calling not-function in typed code"),
            }
        }
        Term::Pi { .. } => ctx.star.clone(), // TODO: maybe filter out indicators properly
        Term::Lam { ident, ty, body } => {
            ctx.with_local(ident.clone(), ctx.undef.clone(), ctx.undef.clone(), |ctx| {
                let body = type_of(ctx, body.clone());
                Rc::new(Term::Pi { ident: ident.clone(), ty: ty.clone(), body })
            })
        }
        Term::FnTy { .. } => ctx.star.clone(),
        Term::App { obj, arg } => {
            let obj_ty = type_of(ctx, obj.clone());
            Rc::new(Term::App {
                obj: obj_ty.clone(),
                arg: arg.clone(),
            })
        }
        Term::Match { obj, cases, default } => {
            let cases = cases.iter().map(|(pat, body)| {
                ctx.with_case(obj.clone(), pat.clone(), |ctx| {
                    let body_ty = type_of(ctx, body.clone());
                    (pat.clone(), body_ty)
                })
            }).collect();
            let default_ty = type_of(ctx, default.clone());
            Rc::new(Term::Match { obj: obj.clone(), cases, default: default_ty.clone() })
        }
        Term::Fn { args, body } => {
            let ret_ty = ctx.with_local_decls(args, |ctx| {
                type_of(ctx, body.clone())
            });
            Rc::new(Term::FnTy { args: args.iter().map(|(_, ty)| ty).cloned().collect(), ret_ty })
        }
        Term::Source { term, .. } => type_of(ctx, term.clone()),
        Term::Block { ret, .. } => {
            type_of(ctx, ret.clone())
        }
        Term::Decl { .. } => ctx.unit_ty.clone(),
        Term::Typed { ty, .. } => ty.clone(),
        Term::Intrinsic(intrinsic) => intrinsic.type_of(),
    }
}

pub fn elaborate(module: &Module) -> Result<IndexMap<SmolStr2, (Rc<Term>, Rc<Term>)>> {
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
        let new_ty = type_of(ctx, new.clone());
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
        let ty = type_of(ctx, value.clone());
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
                return Ok((meta.clone(), type_of(ctx, meta)));
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

            Expr::Binary { op, lhs, rhs } => {
                let (lhs, lhs_ty) = infer(ctx, lhs)?;
                let (rhs, rhs_ty) = infer(ctx, rhs)?;

                let item = SmolStr2::from(match op.node {
                    BinaryOp::Add => "add",
                    BinaryOp::Sub => "sub",
                    BinaryOp::Mul => "mul",
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

                let ty = type_of(ctx, term.clone());
                return Ok((term, ty));
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
        (Meta(id), _) => {
            let a_ty = type_of(ctx, a.clone());
            let b_ty = type_of(ctx, b.clone());
            unify(ctx, a_ty, b_ty.clone())?;
            if let Some(value) = &ctx.metas[*id].0 {
                return unify(ctx, value.clone(), b);
            }
            ctx.metas[*id] = (Some(b.clone()), b_ty);
            return Ok(b);
        }
        (_, Meta(id)) => {
            let a_ty = type_of(ctx, a.clone());
            let b_ty = type_of(ctx, b.clone());
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
        Term::Fn { .. } => todo!(),
        Term::Call { .. } => todo!(),
        Term::FnTy { .. } => todo!(),
        Term::Source { term, .. } => is_concrete(term),
        Term::Pi { body, ty, .. } => is_concrete(body) && is_concrete(ty),
        Term::Lam { body, ty, .. } => is_concrete(body) && is_concrete(ty),
        Term::App { obj, arg } => is_concrete(obj) && is_concrete(arg),
        Term::Match { .. } => false,
        Term::Decl { ty, val, .. } => is_concrete(ty) && is_concrete(val),
        Term::Block { stmts, ret } => stmts.iter().all(is_concrete) && is_concrete(ret),
        Term::Typed { ty, term } => is_concrete(ty) && is_concrete(term),
        Term::Intrinsic(_) => true,
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
        Term::Intrinsic(_) => false,
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
        Term::Call { obj, args } => {
            let obj = nf(ctx, obj.clone())?;
            let args = args.iter().map(|arg| nf(ctx, arg.clone())).collect::<Result<Vec<_>>>()?;
            match &*obj {
                Term::Fn { args: args_decl, body } => {
                    let locals = args.iter().zip(args_decl)
                        .map(|(val, (ident, ty))| {
                            (ident.clone(), (val.clone(), ty.clone()))
                        }).collect::<Vec<_>>();
                    ctx.with_locals(&locals, |ctx| {
                        nf(ctx, body.clone())
                    })
                }
                Term::Intrinsic(intrinsic) => {
                    intrinsic.call(args)
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
                _ => Ok(term),
            }
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
            Term::Fn { .. } => write!(f, "<function>"),
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
                writeln!(f, "{1:0$}{{", indent, "")?;
                INDENT.with(|i| *i.borrow_mut() += 4);
                for stmt in stmts {
                    writeln!(f, "{1:0$}{};", indent, stmt)?;
                }
                writeln!(f, "{1:0$}{}", indent, ret)?;
                INDENT.with(|i| *i.borrow_mut() -= 4);
                write!(f, "{1:0$}}}", indent, "")
            }
            Term::Typed { term, ty } => write!(f, "({})::{}", term, ty),
            Term::Intrinsic(intrinsic) => write!(f, "<{}>", &intrinsic.name()[..]),
        }
    }
}
