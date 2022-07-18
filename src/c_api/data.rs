use std::os::raw::c_char;
use std::ffi::CStr;

use crate::{eqs_data_origin_t, eqs_array_t};

use super::{eqs_status_t, catch_unwind};
use super::utils::copy_str_to_c;


/// Register a new data origin with the given `name`. Calling this function
/// multiple times with the same name will give the same `eqs_data_origin_t`.
///
/// @param name name of the data origin as an UTF-8 encoded NULL-terminated string
/// @param origin pointer to an `eqs_data_origin_t` where the origin will be stored
///
/// @returns The status code of this operation. If the status is not
///          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn eqs_register_data_origin(
    name: *const c_char,
    origin: *mut eqs_data_origin_t,
) -> eqs_status_t {
    catch_unwind(|| {
        check_pointers!(name, origin);

        let name = CStr::from_ptr(name).to_str().unwrap();
        *origin = crate::register_data_origin(name.into());

        Ok(())
    })
}


/// Get the name used to register a given data `origin` in the given `buffer`
///
/// @param origin pre-registered data origin
/// @param buffer buffer to be filled with the data origin name. The origin name
///               will be written  as an UTF-8 encoded, NULL-terminated string
/// @param buffer_size size of the buffer
///
/// @returns The status code of this operation. If the status is not
///          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn eqs_get_data_origin(
    origin: eqs_data_origin_t,
    buffer: *mut c_char,
    buffer_size: usize,
) -> eqs_status_t {
    catch_unwind(|| {
        check_pointers!(buffer);

        let origin = crate::get_data_origin(origin);
        return copy_str_to_c(&origin, buffer, buffer_size);
    })
}

/// Access the data stored inside a rust ndarray.
///
/// After a successful call, the `*data` will point to the first element of a
/// row-major array, the shape of which will be stored in `*shape`.
///
/// @param array `eqs_array_t` wrapping a rust ndarray
/// @param data pointer to be filled with a pointer to the first element of the
///             data
/// @param shape pointer to be filled with the an array containing the shape of
///              the data
/// @param shape_count pointer to be filled with the number of entries in the
///                   `shape` array
///
/// @returns The status code of this operation. If the status is not
///          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn eqs_get_rust_array(
    array: *const eqs_array_t,
    data: *mut *const f64,
    shape: *mut *const usize,
    shape_count: *mut usize,
) -> eqs_status_t {
    catch_unwind(|| {
        check_pointers!(array, data, shape, shape_count);

        #[cfg(feature = "ndarray")]
        {
            let array = (*array).as_array();
            assert!(array.is_standard_layout());
            *data = array.as_ptr();
            *shape = array.shape().as_ptr();
            *shape_count = array.shape().len();

            return Ok(());
        }

        #[cfg(not(feature = "ndarray"))]
        {
            return Err(crate::Error::Internal(
                "support for rust ndarray is disabled, please recompile the code \
                to activate it".into()
            ));
        }
    })
}
