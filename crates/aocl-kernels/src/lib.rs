//! Owned, checked persistent F32 weights for a pinned external AOCL-DLP build.
//!
//! FFI and its ownership invariant live here. Calls on one handle are serialized;
//! different handles may run independently. See README for scope and requirements.
#![deny(unsafe_op_in_unsafe_fn)]

use std::{fmt, sync::Mutex};

const MAX_DIMENSION: usize = 32_768;
const MAX_ROWS: usize = 1_024;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidInput(&'static str),
    Unsupported,
    Backend(i32),
    AllocationFailed,
    Poisoned,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid AOCL input: {message}"),
            Self::Unsupported => {
                f.write_str("AOCL F32 kernels require Linux x86_64 with AVX2 and FMA")
            }
            Self::Backend(code) => write!(f, "AOCL checked kernel failed with status {code}"),
            Self::AllocationFailed => f.write_str("AOCL output allocation failed"),
            Self::Poisoned => f.write_str("AOCL packed-weight lock is poisoned"),
        }
    }
}

impl std::error::Error for Error {}

/// Whether the compiled backend and required AVX2/FMA CPU features are available.
pub fn available() -> bool {
    backend::available()
}

/// A single owned packed F32 weight matrix with shape `[output, input]`.
///
/// Source weights may be released after construction. Packed bytes are specific
/// to this CPU/ISA context and are never exposed for persistence or mutation.
pub struct PackedLinear {
    handle: Mutex<backend::Handle>,
    input: usize,
    output: usize,
    storage_bytes: usize,
}

impl fmt::Debug for PackedLinear {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PackedLinear")
            .field("input", &self.input)
            .field("output", &self.output)
            .field("storage_bytes", &self.storage_bytes)
            .finish_non_exhaustive()
    }
}

impl PackedLinear {
    pub fn new(weights: &[f32], output: usize, input: usize, threads: usize) -> Result<Self> {
        validate_dimension(input)?;
        validate_dimension(output)?;
        if !(1..=64).contains(&threads) {
            return Err(Error::InvalidInput("thread count must be within 1..=64"));
        }
        if checked_elements(output, input)? != weights.len() {
            return Err(Error::InvalidInput(
                "weight length must equal output * input",
            ));
        }
        if !available() {
            return Err(Error::Unsupported);
        }
        let (handle, storage_bytes) = backend::create(weights, output, input, threads)?;
        Ok(Self {
            handle: Mutex::new(handle),
            input,
            output,
            storage_bytes,
        })
    }

    /// Compute contiguous `[rows,input] @ weights.T` into `[rows,output]`.
    pub fn forward(&self, input: &[f32], rows: usize) -> Result<Vec<f32>> {
        if !(1..=MAX_ROWS).contains(&rows) {
            return Err(Error::InvalidInput("row count must be within 1..=1024"));
        }
        if input.len() != checked_elements(rows, self.input)? {
            return Err(Error::InvalidInput(
                "input length must equal rows * input_size",
            ));
        }
        let output_len = checked_elements(rows, self.output)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(output_len)
            .map_err(|_| Error::AllocationFailed)?;
        output.resize(output_len, 0.0);
        let handle = self.handle.lock().map_err(|_| Error::Poisoned)?;
        backend::forward(&handle, input, rows, &mut output)?;
        Ok(output)
    }

    pub fn storage_bytes(&self) -> usize {
        self.storage_bytes
    }

    pub fn input_size(&self) -> usize {
        self.input
    }

    pub fn output_size(&self) -> usize {
        self.output
    }
}

fn validate_dimension(value: usize) -> Result<()> {
    if !(1..=MAX_DIMENSION).contains(&value) {
        return Err(Error::InvalidInput(
            "matrix dimensions must be within 1..=32768",
        ));
    }
    Ok(())
}

fn checked_elements(rows: usize, columns: usize) -> Result<usize> {
    let elements = rows
        .checked_mul(columns)
        .ok_or(Error::InvalidInput("matrix element count overflows"))?;
    let bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(Error::InvalidInput("matrix byte count overflows"))?;
    if bytes > isize::MAX as usize {
        return Err(Error::InvalidInput("matrix exceeds addressable slice size"));
    }
    Ok(elements)
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod backend {
    use super::{Error, Result};
    use std::ptr::NonNull;

    #[repr(C)]
    struct Opaque {
        _private: [u8; 0],
    }

    unsafe extern "C" {
        fn runtime_aocl_available() -> std::ffi::c_int;
        fn runtime_aocl_packed_create(
            weights: *const f32,
            weights_len: usize,
            output: i64,
            input: i64,
            threads: std::ffi::c_int,
            handle: *mut *mut Opaque,
            packed_bytes: *mut u64,
        ) -> std::ffi::c_int;
        fn runtime_aocl_packed_forward(
            handle: *const Opaque,
            input: *const f32,
            input_len: usize,
            rows: i64,
            output: *mut f32,
            output_len: usize,
        ) -> std::ffi::c_int;
        fn runtime_aocl_packed_destroy(handle: *mut Opaque);
    }

    pub(super) struct Handle(NonNull<Opaque>);

    // SAFETY: this value uniquely owns the opaque C allocation. Moving it does
    // not change the allocation; every call reapplies the caller's thread-local
    // AOCL thread count. PackedLinear's Mutex serializes calls and destruction.
    unsafe impl Send for Handle {}

    impl Drop for Handle {
        fn drop(&mut self) {
            // SAFETY: construction returned one owned, non-null handle. Drop
            // runs exactly once, after Mutex ownership excludes in-flight calls.
            unsafe { runtime_aocl_packed_destroy(self.0.as_ptr()) };
        }
    }

    pub(super) fn available() -> bool {
        // SAFETY: this function has no pointer arguments and performs CPU checks.
        unsafe { runtime_aocl_available() != 0 }
    }

    pub(super) fn create(
        weights: &[f32],
        output: usize,
        input: usize,
        threads: usize,
    ) -> Result<(Handle, usize)> {
        let mut raw = std::ptr::null_mut();
        let mut bytes = 0;
        // SAFETY: parent validates dimensions, thread count, and exact weight
        // length. All pointers live through this synchronous copy/reorder call;
        // raw/bytes are writable stack slots. The C shim repeats bounds checks.
        let status = unsafe {
            runtime_aocl_packed_create(
                weights.as_ptr(),
                weights.len(),
                output as i64,
                input as i64,
                threads as std::ffi::c_int,
                &mut raw,
                &mut bytes,
            )
        };
        if status != 0 {
            return Err(Error::Backend(status));
        }
        let handle = Handle(NonNull::new(raw).ok_or(Error::Backend(-3))?);
        let bytes = usize::try_from(bytes).map_err(|_| Error::Backend(-3))?;
        if bytes == 0 {
            return Err(Error::Backend(-3));
        }
        Ok((handle, bytes))
    }

    pub(super) fn forward(
        handle: &Handle,
        input: &[f32],
        rows: usize,
        output: &mut [f32],
    ) -> Result<()> {
        // SAFETY: caller owns the handle's mutex guard and validates exact
        // lengths/rows. Rust borrows ensure these slices cannot overlap; output
        // is separately allocated and initialized. C repeats all shape checks.
        let status = unsafe {
            runtime_aocl_packed_forward(
                handle.0.as_ptr(),
                input.as_ptr(),
                input.len(),
                rows as i64,
                output.as_mut_ptr(),
                output.len(),
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(Error::Backend(status))
        }
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
mod backend {
    use super::{Error, Result};
    pub(super) struct Handle;
    pub(super) fn available() -> bool {
        false
    }
    pub(super) fn create(_: &[f32], _: usize, _: usize, _: usize) -> Result<(Handle, usize)> {
        Err(Error::Unsupported)
    }
    pub(super) fn forward(_: &Handle, _: &[f32], _: usize, _: &mut [f32]) -> Result<()> {
        Err(Error::Unsupported)
    }
}

#[cfg(test)]
mod tests;
