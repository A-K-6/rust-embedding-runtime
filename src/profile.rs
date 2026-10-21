//! Opt-in phase timing for local profiling; absent from normal inference builds.
#[cfg(feature = "profile")]
use std::{cell::RefCell, collections::BTreeMap, time::Instant};

#[cfg(feature = "profile")]
thread_local! {
    static TIMES: RefCell<BTreeMap<&'static str, f64>> = RefCell::default();
}

pub(crate) struct Span {
    #[cfg(feature = "profile")]
    name: &'static str,
    #[cfg(feature = "profile")]
    start: Instant,
}

#[inline]
pub(crate) fn span(_name: &'static str) -> Span {
    Span {
        #[cfg(feature = "profile")]
        name: _name,
        #[cfg(feature = "profile")]
        start: Instant::now(),
    }
}

impl Drop for Span {
    #[inline]
    fn drop(&mut self) {
        #[cfg(feature = "profile")]
        TIMES.with(|times| {
            *times.borrow_mut().entry(self.name).or_default() +=
                self.start.elapsed().as_secs_f64() * 1000.;
        });
    }
}

#[cfg(feature = "profile")]
pub(crate) fn take() -> BTreeMap<&'static str, f64> {
    TIMES.with(|times| std::mem::take(&mut *times.borrow_mut()))
}
