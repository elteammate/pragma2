pub type CompoundResult<T, E> = Result<T, Vec<E>>;


#[macro_export]
macro_rules! cmrg {
    (@ $(,)?) => {
        Result::Ok(())
    };
    (@ $x:expr, $($rest:expr),* $(,)?) => {
        match ($x, cmrg!(@ $($rest,)*)) {
            (Ok(x), Ok(rest)) => Ok((x, rest)),
            (Err(e), Ok(_)) => {
                Err(e)
            }
            (Ok(_), Err(e)) => {
                Err(e)
            }
            (Err(mut e), Err(mut e2)) => {
                e.append(&mut e2);
                Err(e)
            }
        }
    };
    ($x:expr) => {
        match cmrg!(@ $x) {
            Ok((x, ())) => Ok(x),
            Err(e) => Err(e),
        }

    };
    ($x1:expr, $x2:expr) => {
        match cmrg!(@ $x1, $x2) {
            Ok((x1, (x2, ()))) => Ok((x1, x2)),
            Err(e) => Err(e),
        }
    };
    ($x1:expr, $x2:expr, $x3:expr) => {
        match cmrg!(@ $x1, $x2, $x3) {
            Ok((x1, (x2, (x3, ())))) => Ok((x1, x2, x3)),
            Err(e) => Err(e),
        }
    };
    ($x1:expr, $x2:expr, $x3:expr, $x4:expr) => {
        match cmrg!(@ $x1, $x2, $x3, $x4) {
            Ok((x1, (x2, (x3, (x4, ()))))) => Ok((x1, x2, x3, x4)),
            Err(e) => Err(e),
        }
    };
    ($x1:expr, $x2:expr, $x3:expr, $x4:expr, $x5:expr) => {
        match cmrg!(@ $x1, $x2, $x3, $x4, $x5) {
            Ok((x1, (x2, (x3, (x4, (x5, ())))))) => Ok((x1, x2, x3, x4, x5)),
            Err(e) => Err(e),
        }
    };
    ($x1:expr, $x2:expr, $x3:expr, $x4:expr, $x5:expr, $x6:expr) => {
        match cmrg!(@ $x1, $x2, $x3, $x4, $x5, $x6) {
            Ok((x1, (x2, (x3, (x4, (x5, (x6, ()))))))) => Ok((x1, x2, x3, x4, x5, x6)),
            Err(e) => Err(e),
        }
    };
    ($x1:expr, $x2:expr, $x3:expr, $x4:expr, $x5:expr, $x6:expr, $x7:expr) => {
        match cmrg!(@ $x1, $x2, $x3, $x4, $x5, $x6, $x7) {
            Ok((x1, (x2, (x3, (x4, (x5, (x6, (x7, ())))))))) => Ok((x1, x2, x3, x4, x5, x6, x7)),
            Err(e) => Err(e),
        }
    };
    ($x1:expr, $x2:expr, $x3:expr, $x4:expr, $x5:expr, $x6:expr, $x7:expr, $x8:expr) => {
        match cmrg!(@ $x1, $x2, $x3, $x4, $x5, $x6, $x7, $x8) {
            Ok((x1, (x2, (x3, (x4, (x5, (x6, (x7, (x8, ()))))))))) => Ok((x1, x2, x3, x4, x5, x6, x7, x8)),
            Err(e) => Err(e),
        }
    };
}

#[macro_export]
macro_rules! vmrg {
    ($x:expr) => {{
        let mut errors = Vec::new();
        let mut results = Vec::with_capacity($x.len());
        for x in $x {
            match x {
                Ok(x) => results.push(x),
                Err(mut e) => errors.append(&mut e),
            }
        }
        if errors.is_empty() {
            Ok(results)
        } else {
            Err(errors)
        }
    }};
}

pub trait ResultExt<T, E> {
    fn delegate<Q>(self, other: &mut CompoundResult<Q, E>) -> Option<T>;
}

impl<T, E> ResultExt<T, E> for CompoundResult<T, E> {
    fn delegate<Q>(self, other: &mut CompoundResult<Q, E>) -> Option<T> {
        match self {
            Ok(x) => Some(x),
            Err(mut e) => {
                match other {
                    Ok(_) => *other = Err(e),
                    Err(e2) => {
                        e2.append(&mut e);
                    }
                }
                None
            }
        }
    }
}

pub fn err<T, E>(e: E) -> CompoundResult<T, E> {
    Err(vec![e])
}
