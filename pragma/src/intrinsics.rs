use std::fmt::Debug;
use std::rc::Rc;
use indexmap::IndexMap;
use crate::elaborate::{Term, Result};
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
        $i.insert($intrinsic.name(), (Rc::new(Term::Intrinsic(Box::new(intrinsic))), ty));
    }
}

pub trait Intrinsic: Debug {
    fn type_of(&self) -> Rc<Term>;
    fn call(&self, args: Vec<Rc<Term>>) -> Result<Rc<Term>>;
    fn name(&self) -> SmolStr2;
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
}


pub fn make_intrinsics() -> IndexMap<SmolStr2, (Rc<Term>, Rc<Term>)> {
    let mut i = IndexMap::new();

    intrinsic!(i("int", Term::IntTy, *));
    intrinsic!(i("bool", Term::BoolTy, *));
    intrinsic!(i("unit", Term::UnitTy, *));
    intrinsic!(i("string", Term::StringTy, *));

    intrinsic!(i(IntAdd));
    intrinsic!(i(IntSub));
    intrinsic!(i(IntMul));

    i
}