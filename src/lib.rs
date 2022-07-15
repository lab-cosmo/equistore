#![warn(clippy::all, clippy::pedantic)]

// disable some style lints
#![allow(clippy::needless_return, clippy::must_use_candidate, clippy::comparison_chain)]
#![allow(clippy::redundant_field_names, clippy::redundant_closure_for_method_calls, clippy::redundant_else)]
#![allow(clippy::unreadable_literal, clippy::option_if_let_else, clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc, clippy::missing_safety_doc)]
#![allow(clippy::similar_names, clippy::borrow_as_ptr)]


mod utils;

pub mod labels;
pub use self::labels::{LabelsBuilder, LabelValue, Labels};

mod data;
pub use self::data::{eqs_array_t, eqs_sample_mapping_t, eqs_data_origin_t};
pub use self::data::{DataOrigin, register_data_origin, get_data_origin};
pub use self::data::{Array, EmptyArray};

mod blocks;
pub use self::blocks::{BasicBlock, TensorBlock, TensorBlockRefMut};

pub mod tensor;
pub use self::tensor::TensorMap;

#[doc(hidden)]
pub mod c_api;
use c_api::eqs_status_t;

#[cfg(feature = "serialization")]
pub mod io;

#[cfg(not(feature = "serialization"))]
#[allow(clippy::needless_pass_by_value)]
pub mod io {
    #![allow(unused_variables)]
    use super::{Error, TensorMap};

    pub fn load<R: std::io::Read + std::io::Seek>(reader: R) -> Result<TensorMap, Error> {
        return Err(Error::Serialization("serialization was not enabled in equistore".into()))
    }

    pub fn save<W: std::io::Write + std::io::Seek>(writer: W, tensor: &TensorMap) -> Result<(), Error> {
        return Err(Error::Serialization("serialization was not enabled in equistore".into()))
    }
}

/// The possible sources of error in equistore
#[derive(Debug)]
pub enum Error {
    /// A function got an invalid parameter
    InvalidParameter(String),
    /// A buffer passed to a C API function does not have the right size
    BufferSize(String),
    /// I/O error when loading/writing `TensorMap` to a file
    Io(std::io::Error),
    /// Serialization format error when loading/writing `TensorMap` to a file
    Serialization(String),
    /// External error, coming from a function used as a callback in `eqs_array_t`
    External {
        status: eqs_status_t,
        context: String,
    },
    /// Any other internal error, usually these are internal bugs.
    Internal(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidParameter(e) => write!(f, "invalid parameter: {}", e),
            Error::Io(e) => write!(f, "io error: {}", e),
            Error::Serialization(e) => write!(f, "serialization format error: {}", e),
            Error::BufferSize(e) => write!(f, "buffer is not big enough: {}", e),
            Error::External { status, context } => write!(f, "external error: {} (status {})", context, status.as_i32()),
            Error::Internal(e) => write!(f, "internal error: {}", e),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::InvalidParameter(_) |
            Error::Serialization(_) |
            Error::Internal(_) |
            Error::BufferSize(_) |
            Error::External {..} => None,
            Error::Io(e) => Some(e),
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Error::Io(error)
    }
}

#[cfg(feature = "serialization")]
impl From<(String, zip::result::ZipError)> for Error {
    fn from((path, error): (String, zip::result::ZipError)) -> Self {
        match error {
            zip::result::ZipError::Io(e) => Error::Io(e),
            error => Error::Serialization(format!("{}: {}", error, path)),
        }
    }
}

// Box<dyn Any + Send + 'static> is the error type in std::panic::catch_unwind
impl From<Box<dyn std::any::Any + Send + 'static>> for Error {
    fn from(error: Box<dyn std::any::Any + Send + 'static>) -> Error {
        let message = if let Some(message) = error.downcast_ref::<String>() {
            message.clone()
        } else if let Some(message) = error.downcast_ref::<&str>() {
            (*message).to_owned()
        } else {
            panic!("panic message is not a string, something is very wrong")
        };

        Error::Internal(message)
    }
}
