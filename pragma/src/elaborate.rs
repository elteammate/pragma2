use std::cell::Cell;
use std::fmt::{Display, Formatter};
use std::ops::Deref;
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
    fn with_comment(self, comment: impl std::fmt::Display) -> Self;
}

impl<T> ResultExt<T> for Result<T> {
    fn with_comment(self, comment: impl std::fmt::Display) -> Result<T> {
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

pub fn elaborate(module: &Module) -> Result<IndexMap<SmolStr2, (Rc<Term>, Rc<Term>)>> {
    let anon = SmolStr2::from("<anonymous>");
    let mut ctx = Ctx {
        star: Rc::new(Term::Type(Vec::new())),
        undef: Rc::new(Term::Undef),
        anon: anon.clone(),
        anon_term: Rc::new(Term::Var(anon.clone())),
        items: make_intrinsics(),
        locals: Vec::new(),
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

    let (new, new_ty) = elaborate_item_impl(ctx, item, 0)?;
    let new = nf(ctx, new);
    let new_ty = nf(ctx, new_ty);

    let (existing, existing_ty) = match ctx.items.get(&item.ident.node) {
        Some((existing, existing_ty)) => (existing.clone(), existing_ty.clone()),
        None => {
            ctx.items.insert(item.ident.node.clone(), (new.clone(), new_ty.clone()));
            return Ok(());
        }
    };

    let span = item.ident.span;
    let ty1 = unify(ctx, existing_ty, new_ty, span)
        .with_comment("Cannot unify the existing item with this because of type mismatch")?;
    let value1 = unify(ctx, existing, new, span)
        .with_comment("Cannot unify the existing item with this because of value mismatch")?;
    let (value, ty) = (value1, ty1);
    
    let value = nf(ctx, value);
    let ty = nf(ctx, ty);

    ctx.items.insert(item.ident.node.clone(), (value, ty));
    Ok(())
}

/// Infer the largest type of `expr` and return its value.
fn infer(ctx: &mut Ctx, expr: &Ast<Expr>) -> Result<(Rc<Term>, Rc<Term>)> {
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
                match &*callee_ty {
                    Term::Pi { ident, ty, body } => {
                        let param = &param.0;
                        let mut param_ty = ty.clone();
                        let param = check(ctx, param, &mut param_ty)?;
                        (callee, callee_ty) = ctx.with_local(ident.clone(), param.clone(), param_ty, |ctx| {
                            (
                                Rc::new(Term::App {
                                    obj: callee,
                                    arg: param,
                                }),
                                // TODO: a bit sus?.. why not nf(app(body, param_ty))?
                                nf(ctx, body.clone()),
                            )
                        })
                    }
                    _ => return err(ElaborateError::Commented(expr.span, "Cannot apply a non-function".to_string())),
                }
            }
            return Ok((callee, callee_ty))
        }

        _ => todo!("Can't infer type of expression yet {:#?}", expr.node),
    };

    Ok((Rc::new(val), Rc::new(ty)))
}


/// Check that `expr` has type `ty` and return the inferred value,
/// possibly strengthening the type in the process.
fn check(ctx: &mut Ctx, expr: &Ast<Expr>, ty: &mut Rc<Term>) -> Result<Rc<Term>> {
    match &expr.node {
        _ => {
            let (value, inferred) = infer(ctx, expr)?;
            *ty = unify(ctx, inferred.clone(), ty.clone(), expr.span)
                .with_comment("Typecheck failed")?;
            Ok(value)
        }
    }
}

/// Return the largest type that is a subtype of both `a` and `b`.
fn unify(ctx: &mut Ctx, a: Rc<Term>, b: Rc<Term>, span: Span) -> Result<Rc<Term>> {
    let a = nf(ctx, a);
    let b = nf(ctx, b);
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
        (Type(xs), Type(ys)) => {
            let result = Type(Vec::new());
            assert!(xs.is_empty() && ys.is_empty(), "TODO: Unify type parameters");
            result
        },

        _ => return err(ElaborateError::UnificationFailed(
            span, 
            format!("Cannot unify {} and {}", a, b)),
        )
    };

    Ok(Rc::new(result))
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
    }
}

fn nf(ctx: &mut Ctx, term: Rc<Term>) -> Rc<Term> {
    match &*term {
        Term::Var(ident) => {
            if let Some((value, _)) = ctx.lookup(ident.clone()) {
                if let Term::Var(ident2) = &*value {
                    if ident == ident2 {
                        return value;
                    }
                }
                nf(ctx, value)
            } else {
                term
            }
        }
        Term::Type(indicators) => {
            Rc::new(Term::Type(indicators.iter().map(|i| nf(ctx, i.clone())).collect()))
        }
        Term::Pi { ident, ty, body } => {
            let ty = nf(ctx, ty.clone());
            let body = nf(ctx, body.clone());
            Rc::new(Term::Pi { ident: ident.clone(), ty, body })
        }
        Term::Lam { ident, ty, body } => {
            let ty = nf(ctx, ty.clone());
            let body = nf(ctx, body.clone());
            Rc::new(Term::Lam { ident: ident.clone(), ty, body })
        }
        Term::App { obj, arg } => {
            let obj = nf(ctx, obj.clone());
            let arg = nf(ctx, arg.clone());
            match &*obj {
                Term::Lam { ident, ty: _, body } => {
                    let body = substitute(ctx, body.clone(), ident.clone(), arg.clone());
                    nf(ctx, body)
                }
                Term::Pi { ident, ty: _, body } => {
                    let body = substitute(ctx, body.clone(), ident.clone(), arg.clone());
                    nf(ctx, body)
                }
                _ => Rc::new(Term::App { obj, arg }),
            }
        }
        Term::Match { obj, cases, default } => {
            let obj = nf(ctx, obj.clone());
            let cases = cases.iter().map(|(pat, body)| {
                let pat = nf(ctx, pat.clone());
                let body = nf(ctx, body.clone());
                (pat, body)
            }).collect::<Vec<_>>();
            let default = nf(ctx, default.clone());
            let mut all_applicable = true;
            for (pat, body) in &cases {
                let applicable = is_concrete(&obj) && is_concrete(pat);
                if applicable {
                    if unify(ctx, obj.clone(), pat.clone(), Span(0, 0)).is_ok() {
                        return nf(ctx, body.clone());
                    }
                } else {
                    all_applicable = false;
                }
            }
            if all_applicable {
                default
            } else {
                Rc::new(Term::Match { obj, cases, default })
            }
        }
        _ => term,
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

impl Display for Term {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
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
                writeln!(f, "match {} {{", obj)?;
                for (pat, body) in cases {
                    writeln!(f, "  {} => {}", pat, body)?;
                }
                writeln!(f, "  _ => {}", default)?;
                write!(f, "}}")
            }
        }
    }
}
