use std::rc::Rc;
use indexmap::IndexMap;
use crate::elaborate::Term;
use crate::smol_str2::SmolStr2;

macro_rules! intrinsic {
    (@term *) => { Rc::new(Term::Type(vec![])) };
    (@term $t:expr) => { Rc::new($t) };
    ($i:ident($name:expr, $term:expr, $($ty:tt)+)) => {
        $i.insert(SmolStr2::from($name), (intrinsic!(@term $term), intrinsic!(@term $($ty)+)));
    };
}

pub fn make_intrinsics() -> IndexMap<SmolStr2, (Rc<Term>, Rc<Term>)> {
    let mut i = IndexMap::new();

    intrinsic!(i("int", Term::IntTy, *));
    intrinsic!(i("bool", Term::BoolTy, *));
    intrinsic!(i("unit", Term::UnitTy, *));
    intrinsic!(i("string", Term::StringTy, *));

    i
}