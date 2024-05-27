use std::fmt::Debug;
use std::rc::Rc;
use indexmap::IndexMap;
use crate::c;
use crate::elaborate::{Term, Result, Extraction};
use crate::smol_str2::SmolStr2;

macro_rules! intrinsic {
    (@term *) => { Rc::new(Term::Type(vec![])) };
    (@term $t:expr) => { Rc::new($t) };
    ($i:ident($name:expr, $term:expr, $($ty:tt)+)) => {
        $i.insert(SmolStr2::from($name), (intrinsic!(@term $term), intrinsic!(@term $($ty)+)));
    };
    ($i:ident($intrinsic:expr)) => {
        let intrinsic = $intrinsic;
        let ty = intrinsic.type_of();
        $i.insert($intrinsic.name(), (Rc::new(Term::Intrinsic(Box::leak(Box::new(intrinsic)))), ty));
    }
}

pub trait Intrinsic: Debug {
    fn type_of(&self) -> Rc<Term>;
    fn can_call(&self) -> bool { true }
    fn can_emit(&self) -> bool { true }
    fn call(&self, args: Vec<Rc<Term>>) -> Result<Rc<Term>>;
    fn name(&self) -> SmolStr2;
    fn emit(&self, e: &mut Extraction, args: Vec<c::Expr>) -> (Vec<c::Statement>, Option<c::Expr>);
}

impl PartialEq for &dyn Intrinsic {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

#[derive(Debug)]
struct IntAdd;

impl Intrinsic for IntAdd {
    fn type_of(&self) -> Rc<Term> {
        Rc::new(Term::FnTy {
            args: vec![Rc::new(Term::IntTy), Rc::new(Term::IntTy)],
            ret_ty: Rc::new(Term::IntTy),
        })
    }

    fn call(&self, args: Vec<Rc<Term>>) -> Result<Rc<Term>> {
        assert_eq!(2, args.len());
        let Term::Int(a) = &*args[0] else { panic!("expected int") };
        let Term::Int(b) = &*args[1] else { panic!("expected int") };
        Ok(Rc::new(Term::Int(a + b)))
    }

    fn name(&self) -> SmolStr2 {
        SmolStr2::from("__intrinsic_int_add")
    }

    fn emit(&self, _: &mut Extraction, args: Vec<c::Expr>) -> (Vec<c::Statement>, Option<c::Expr>) {
        let mut iter = args.into_iter();
        let a = iter.next().unwrap();
        let b = iter.next().unwrap();
        assert!(iter.next().is_none());
        (vec![], Some(c::Expr::Plus {
            lhs: Box::new(a),
            rhs: Box::new(b),
        }))
    }
}

#[derive(Debug)]
struct IntSub;

impl Intrinsic for IntSub {
    fn type_of(&self) -> Rc<Term> {
        Rc::new(Term::FnTy {
            args: vec![Rc::new(Term::IntTy), Rc::new(Term::IntTy)],
            ret_ty: Rc::new(Term::IntTy),
        })
    }

    fn call(&self, args: Vec<Rc<Term>>) -> Result<Rc<Term>> {
        assert_eq!(2, args.len());
        let Term::Int(a) = &*args[0] else { panic!("expected int") };
        let Term::Int(b) = &*args[1] else { panic!("expected int") };
        Ok(Rc::new(Term::Int(a - b)))
    }

    fn name(&self) -> SmolStr2 {
        SmolStr2::from("__intrinsic_int_sub")
    }

    fn emit(&self, _: &mut Extraction, args: Vec<c::Expr>) -> (Vec<c::Statement>, Option<c::Expr>) {
        let mut iter = args.into_iter();
        let a = iter.next().unwrap();
        let b = iter.next().unwrap();
        assert!(iter.next().is_none());
        (vec![], Some(c::Expr::Minus {
            lhs: Box::new(a),
            rhs: Box::new(b),
        }))
    }
}

#[derive(Debug)]
struct IntMul;


impl Intrinsic for IntMul {
    fn type_of(&self) -> Rc<Term> {
        Rc::new(Term::FnTy {
            args: vec![Rc::new(Term::IntTy), Rc::new(Term::IntTy)],
            ret_ty: Rc::new(Term::IntTy),
        })
    }

    fn call(&self, args: Vec<Rc<Term>>) -> Result<Rc<Term>> {
        assert_eq!(2, args.len());
        let Term::Int(a) = &*args[0] else { panic!("expected int") };
        let Term::Int(b) = &*args[1] else { panic!("expected int") };
        Ok(Rc::new(Term::Int(a * b)))
    }

    fn name(&self) -> SmolStr2 {
        SmolStr2::from("__intrinsic_int_mul")
    }

    fn emit(&self, _: &mut Extraction, args: Vec<c::Expr>) -> (Vec<c::Statement>, Option<c::Expr>) {
        let mut iter = args.into_iter();
        let a = iter.next().unwrap();
        let b = iter.next().unwrap();
        assert!(iter.next().is_none());
        (vec![], Some(c::Expr::Multiply {
            lhs: Box::new(a),
            rhs: Box::new(b),
        }))
    }
}

#[derive(Debug)]
struct IntLt;

impl Intrinsic for IntLt {
    fn type_of(&self) -> Rc<Term> {
        Rc::new(Term::FnTy {
            args: vec![Rc::new(Term::IntTy), Rc::new(Term::IntTy)],
            ret_ty: Rc::new(Term::BoolTy),
        })
    }

    fn call(&self, args: Vec<Rc<Term>>) -> Result<Rc<Term>> {
        assert_eq!(2, args.len());
        let Term::Int(a) = &*args[0] else { panic!("expected int") };
        let Term::Int(b) = &*args[1] else { panic!("expected int") };
        Ok(Rc::new(Term::Bool(a < b)))
    }

    fn name(&self) -> SmolStr2 {
        SmolStr2::from("__intrinsic_int_lt")
    }

    fn emit(&self, _: &mut Extraction, args: Vec<c::Expr>) -> (Vec<c::Statement>, Option<c::Expr>) {
        let mut iter = args.into_iter();
        let a = iter.next().unwrap();
        let b = iter.next().unwrap();
        assert!(iter.next().is_none());
        (vec![], Some(c::Expr::Lt {
            lhs: Box::new(a),
            rhs: Box::new(b),
        }))
    }
}

#[derive(Debug)]
struct Puts;

impl Intrinsic for Puts {
    fn type_of(&self) -> Rc<Term> {
        Rc::new(Term::FnTy {
            args: vec![Rc::new(Term::StringTy)],
            ret_ty: Rc::new(Term::UnitTy),
        })
    }

    fn can_call(&self) -> bool { false }

    fn call(&self, _: Vec<Rc<Term>>) -> Result<Rc<Term>> {
        panic!("cannot call puts")
    }

    fn name(&self) -> SmolStr2 {
        SmolStr2::from("__intrinsic_puts")
    }

    fn emit(&self, e: &mut Extraction, args: Vec<c::Expr>) -> (Vec<c::Statement>, Option<c::Expr>) {
        let mut iter = args.into_iter();
        let s = iter.next().unwrap();
        assert!(iter.next().is_none());
        (vec![], Some(c::Expr::Call {
            f: Box::new(c::Expr::External { id: e.add_external("puts".into(), || {
                (
                    vec!["stdio.h".into()],
                    c::ExternalFunction {
                        name: "puts".into(),
                    },
                )
            }) }),
            args: vec![s],
        }))
    }
}

pub fn make_intrinsics() -> IndexMap<SmolStr2, (Rc<Term>, Rc<Term>)> {
    let mut i = IndexMap::new();

    intrinsic!(i("main", Term::Undef, Term::FnTy { args: vec![], ret_ty: Rc::new(Term::UnitTy) }));
    intrinsic!(i("int", Term::IntTy, *));
    intrinsic!(i("bool", Term::BoolTy, *));
    intrinsic!(i("unit", Term::UnitTy, *));
    intrinsic!(i("string", Term::StringTy, *));

    intrinsic!(i(IntAdd));
    intrinsic!(i(IntSub));
    intrinsic!(i(IntMul));
    intrinsic!(i(IntLt));
    intrinsic!(i(Puts));

    i
}