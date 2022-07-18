use std::os::raw::{c_char, c_void};
use std::ffi::CStr;

use crate::{LabelValue, Labels, LabelsBuilder, Error};
use super::status::{eqs_status_t, catch_unwind};

/// A set of labels used to carry metadata associated with a tensor map.
///
/// This is similar to a list of `count` named tuples, but stored as a 2D array
/// of shape `(count, size)`, with a set of names associated with the columns of
/// this array (often called *variables*). Each row/entry in this array is
/// unique, and they are often (but not always) sorted in lexicographic order.
#[repr(C)]
pub struct eqs_labels_t {
    /// internal: pointer to the rust `Labels` struct if any, null otherwise
    pub labels_ptr: *const c_void,

    /// Names of the variables composing this set of labels. There are `size`
    /// elements in this array, each being a NULL terminated UTF-8 string.
    pub names: *const *const c_char,
    /// Pointer to the first element of a 2D row-major array of 32-bit signed
    /// integer containing the values taken by the different variables in
    /// `names`. Each row has `size` elements, and there are `count` rows in
    /// total.
    pub values: *const i32,
    /// Number of variables/size of a single entry in the set of labels
    pub size: usize,
    /// Number entries in the set of labels
    pub count: usize,
}

impl std::convert::TryFrom<&eqs_labels_t> for Labels {
    type Error = Error;

    fn try_from(labels: &eqs_labels_t) -> Result<Labels, Self::Error> {
        if labels.names.is_null() {
            return Err(Error::InvalidParameter("labels.names can not be NULL in eqs_labels_t".into()))
        }

        if labels.values.is_null() && labels.count > 0 {
            return Err(Error::InvalidParameter("labels.values is NULL but labels.count is >0 in eqs_labels_t".into()))
        }

        let mut names = Vec::new();
        unsafe {
            for i in 0..labels.size {
                let name = CStr::from_ptr(*(labels.names.add(i)));
                names.push(name.to_str().expect("invalid UTF8 name"));
            }
        }

        let mut builder = LabelsBuilder::new(names);

        unsafe {
            let slice = std::slice::from_raw_parts(labels.values.cast::<LabelValue>(), labels.count * labels.size);
            if !slice.is_empty() {
                for chunk in slice.chunks_exact(labels.size) {
                    builder.add(chunk);
                }
            }
        }

        return Ok(builder.finish());
    }
}


impl std::convert::TryFrom<&Labels> for eqs_labels_t {
    type Error = Error;

    fn try_from(rust_labels: &Labels) -> Result<eqs_labels_t, Self::Error> {
        let size = rust_labels.size();
        let count = rust_labels.count();

        let values = if rust_labels.count() == 0 || rust_labels.size() == 0 {
            std::ptr::null()
        } else {
            (&rust_labels[0][0] as *const LabelValue).cast()
        };

        let names = if rust_labels.size() == 0 {
            std::ptr::null()
        } else {
            rust_labels.c_names().as_ptr().cast()
        };

        // this is a bit sketchy & rely on the fact that the containing
        // `eqs_block_t` or `eqs_tensormap_t` has a fixed address since it is
        // boxed. This also valid only for as long as the block/tensor map is
        // not modified.
        // TODO: could we use Pin/Pin projection here?
        let labels_ptr = (rust_labels as *const Labels).cast();

        Ok(eqs_labels_t {
            labels_ptr, names, values, size, count
        })
    }
}


/// Get the position of the entry defined by the `values` array in the given set
/// of `labels`. This operation is only available if the labels correspond to a
/// set of Rust Labels (i.e. `labels.labels_ptr` is not NULL).
///
/// @param labels set of labels coming from an `eqs_block_t` or an `eqs_tensormap_t`
/// @param values array containing the label to lookup
/// @param values_count size of the values array
/// @param result position of the values in the labels or -1 if the values
///               were not found
///
/// @returns The status code of this operation. If the status is not
///          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn eqs_labels_position(
    labels: eqs_labels_t,
    values: *const i32,
    values_count: usize,
    result: *mut i64
) -> eqs_status_t {
    catch_unwind(|| {
        if labels.labels_ptr.is_null() {
            return Err(Error::InvalidParameter(
                "these labels do not support calling eqs_labels_position".into()
            ));
        }

        let labels = &(*labels.labels_ptr.cast::<Labels>());
        if values_count != labels.size() {
            return Err(Error::InvalidParameter(format!(
                "expected label of size {} in eqs_labels_position, got size {}",
                (*labels).size(), values_count
            )));
        }

        let label = std::slice::from_raw_parts(values.cast(), values_count);
        *result = labels.position(label).map_or(-1, |p| p as i64);

        Ok(())
    })
}
