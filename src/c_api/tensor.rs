use std::os::raw::c_char;
use std::ffi::CStr;
use std::convert::{TryFrom, TryInto};
use std::collections::BTreeSet;

use crate::{TensorMap, Labels, TensorBlock, Error};

use super::labels::eqs_labels_t;
use super::blocks::eqs_block_t;
use super::status::{eqs_status_t, catch_unwind};

/// Opaque type representing a `TensorMap`.
#[allow(non_camel_case_types)]
pub struct eqs_tensormap_t(TensorMap);

impl eqs_tensormap_t {
    /// Create a raw pointer to `eqs_tensormap_t` using a rust Box
    pub fn into_boxed_raw(tensor: TensorMap) -> *mut eqs_tensormap_t {
        let boxed = Box::new(eqs_tensormap_t(tensor));
        return Box::into_raw(boxed);
    }

    /// Take a raw pointer created by `eqs_tensormap_t::into_boxed_raw` and
    /// extract the TensorMap. The pointer is consumed by this function and no
    /// longer valid.
    pub unsafe fn from_boxed_raw(tensor: *mut eqs_tensormap_t) -> TensorMap {
        return Box::from_raw(tensor).0;
    }
}

impl std::ops::Deref for eqs_tensormap_t {
    type Target = TensorMap;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for eqs_tensormap_t {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}


/// Create a new `eqs_tensormap_t` with the given `keys` and `blocks`.
/// `blocks_count` must be set to the number of entries in the blocks array.
///
/// The new tensor map takes ownership of the blocks, which should not be
/// released separately.
///
/// The memory allocated by this function and the blocks should be released
/// using `eqs_tensormap_free`.
///
/// @param keys labels containing the keys associated with each block
/// @param blocks pointer to the first element of an array of blocks
/// @param blocks_count number of elements in the `blocks` array
///
/// @returns A pointer to the newly allocated tensor map, or a `NULL` pointer in
///          case of error. In case of error, you can use `eqs_last_error()`
///          to get the error message.
#[no_mangle]
pub unsafe extern fn eqs_tensormap(
    keys: eqs_labels_t,
    blocks: *mut *mut eqs_block_t,
    blocks_count: usize,
) -> *mut eqs_tensormap_t {
    let mut result = std::ptr::null_mut();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);
    let status = catch_unwind(move || {
        let keys = Labels::try_from(&keys)?;

        let blocks_slice = std::slice::from_raw_parts_mut(blocks, blocks_count);
        // check for uniqueness of the pointers: we don't want to move out
        // the same value twice
        if blocks_slice.iter().collect::<BTreeSet<_>>().len() != blocks_slice.len() {
            return Err(Error::InvalidParameter(
                "got the same block more than once when constructing a tensor map".into()
            ));
        }

        let blocks_vec = blocks_slice.iter_mut().map(|ptr| {
            // move out of the blocks pointers
            let block = Box::from_raw(*ptr).block();
            *ptr = std::ptr::null_mut();
            return block;
        }).collect();

        let tensor = TensorMap::new(keys, blocks_vec)?;

        // force the closure to capture the full unwind_wrapper, not just
        // unwind_wrapper.0
        let _ = &unwind_wrapper;
        *(unwind_wrapper.0) = eqs_tensormap_t::into_boxed_raw(tensor);
        Ok(())
    });

    if !status.is_success() {
        return std::ptr::null_mut();
    }

    return result;
}


/// Free the memory associated with a `tensor` previously created with
/// `eqs_tensormap`.
///
/// If `tensor` is `NULL`, this function does nothing.
///
/// @param tensor pointer to an existing tensor map, or `NULL`
///
/// @returns The status code of this operation. If the status is not
///          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn eqs_tensormap_free(tensor: *mut eqs_tensormap_t) -> eqs_status_t {
    catch_unwind(|| {
        if !tensor.is_null() {
            std::mem::drop(eqs_tensormap_t::from_boxed_raw(tensor));
        }

        Ok(())
    })
}


/// Get the keys for the given `tensor` map. After a successful call to this
/// function, `keys.values` contains a pointer to memory inside the
/// `tensor` which is invalidated when the tensor map is freed with
/// `eqs_tensormap_free` or the set of keys is modified by calling one
/// of the `eqs_tensormap_keys_to_XXX` function.

/// @param tensor pointer to an existing tensor map
/// @param keys pointer to be filled with the keys of the tensor map
///
/// @returns The status code of this operation. If the status is not
///          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn eqs_tensormap_keys(
    tensor: *const eqs_tensormap_t,
    keys: *mut eqs_labels_t,
) -> eqs_status_t {
    catch_unwind(|| {
        check_pointers!(tensor, keys);

        *keys = (*tensor).keys().try_into()?;
        Ok(())
    })
}


/// Get a pointer to the `index`-th block in this tensor map.
///
/// The block memory is still managed by the tensor map, this block should not
/// be freed. The block is invalidated when the tensor map is freed with
/// `eqs_tensormap_free` or the set of keys is modified by calling one
/// of the `eqs_tensormap_keys_to_XXX` function.
///
/// @param tensor pointer to an existing tensor map
/// @param block pointer to be filled with a block
/// @param index index of the block to get
///
/// @returns The status code of this operation. If the status is not
///          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn eqs_tensormap_block_by_id(
    tensor: *const eqs_tensormap_t,
    block: *mut *const eqs_block_t,
    index: usize,
) -> eqs_status_t {
    catch_unwind(|| {
        check_pointers!(tensor, block);

        (*block) = (&(*tensor).blocks()[index] as *const TensorBlock).cast();

        Ok(())
    })
}


/// Get indices of the blocks in this `tensor` corresponding to the given
/// `selection`. The `selection` should have a subset of the names/variables of
/// the keys for this tensor map, and only one entry, describing the requested
/// blocks.
///
/// When calling this function, `*count` should contain the number of entries in
/// `block_indexes`. When the function returns successfully, `*count` will
/// contain the number of blocks matching the selection, i.e. how many values
/// were written to `block_indexes`.
///
/// @param tensor pointer to an existing tensor map
/// @param block_indexes array to be filled with indexes of blocks in the tensor
///                      map matching the `selection`
/// @param count number of entries in `block_indexes`
/// @param selection labels with a single entry describing which blocks are requested
///
/// @returns The status code of this operation. If the status is not
///          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn eqs_tensormap_blocks_matching(
    tensor: *const eqs_tensormap_t,
    block_indexes: *mut usize,
	count: *mut usize,
    selection: eqs_labels_t,
) -> eqs_status_t {
    catch_unwind(|| {
        check_pointers!(tensor, block_indexes, count);

        if *count != (*tensor).keys().count() {
            return Err(Error::InvalidParameter(format!(
                "expected an array of {} elements as input to eqs_tensormap_blocks_matching, got {} elements",
                (*tensor).keys().count(), *count
            )));
        }

        let selection = Labels::try_from(&selection)?;
        let rust_blocks = (*tensor).blocks_matching(&selection)?;
        let block_indexes = std::slice::from_raw_parts_mut(block_indexes, *count);
		*count = rust_blocks.len();
		for (idx,block) in rust_blocks.into_iter().enumerate() {
            block_indexes[idx] = block;
		}

        Ok(())
    })
}


/// Merge blocks with the same value for selected keys variables along the
/// property axis.
///
/// The variables (names) of `keys_to_move` will be moved from the keys to
/// the property labels, and blocks with the same remaining keys variables
/// will be merged together along the property axis.
///
/// If `keys_to_move` does not contains any entries (`keys_to_move.count
/// == 0`), then the new property labels will contain entries corresponding
/// to the merged blocks only. For example, merging a block with key `a=0`
/// and properties `p=1, 2` with a block with key `a=2` and properties `p=1,
/// 3` will produce a block with properties `a, p = (0, 1), (0, 2), (2, 1),
/// (2, 3)`.
///
/// If `keys_to_move` contains entries, then the property labels must be the
/// same for all the merged blocks. In that case, the merged property labels
/// will contains each of the entries of `keys_to_move` and then the current
/// property labels. For example, using `a=2, 3` in `keys_to_move`, and
/// blocks with properties `p=1, 2` will result in `a, p = (2, 1), (2, 2),
/// (3, 1), (3, 2)`.
///
/// The new sample labels will contains all of the merged blocks sample
/// labels. The order of the samples is controlled by `sort_samples`. If
/// `sort_samples` is true, samples are re-ordered to keep them
/// lexicographically sorted. Otherwise they are kept in the order in which
/// they appear in the blocks.
///
/// @param tensor pointer to an existing tensor map
/// @param keys_to_move description of the keys to move
/// @param sort_samples whether to sort the samples lexicographically after
///                     merging blocks
///
/// @returns The status code of this operation. If the status is not
///          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn eqs_tensormap_keys_to_properties(
    tensor: *mut eqs_tensormap_t,
    keys_to_move: eqs_labels_t,
    sort_samples: bool,
) -> eqs_status_t {
    catch_unwind(|| {
        check_pointers!(tensor);

        let keys_to_move = Labels::try_from(&keys_to_move)?;
        (*tensor).keys_to_properties(&keys_to_move, sort_samples)?;

        Ok(())
    })
}


/// Move the given variables from the component labels to the property labels
/// for each block in this tensor map.
///
/// `variables` must be an array of `variables_count` NULL-terminated strings,
/// encoded as UTF-8.
///
/// @param tensor pointer to an existing tensor map
/// @param variables names of the key variables to move to the properties
/// @param variables_count number of entries in the `variables` array
///
/// @returns The status code of this operation. If the status is not
///          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn eqs_tensormap_components_to_properties(
    tensor: *mut eqs_tensormap_t,
    variables: *const *const c_char,
    variables_count: usize,
) -> eqs_status_t {
    catch_unwind(|| {
        check_pointers!(tensor, variables);

        let mut rust_variables = Vec::new();
        for &variable in std::slice::from_raw_parts(variables, variables_count) {
            check_pointers!(variable);
            let variable = CStr::from_ptr(variable).to_str().expect("invalid utf8");
            rust_variables.push(variable);
        }

        (*tensor).components_to_properties(&rust_variables)?;

        Ok(())
    })
}

/// Merge blocks with the same value for selected keys variables along the
/// samples axis.
///
/// The variables (names) of `keys_to_move` will be moved from the keys to
/// the sample labels, and blocks with the same remaining keys variables
/// will be merged together along the sample axis.
///
/// `keys_to_move` must be empty (`keys_to_move.count == 0`), and the new
/// sample labels will contain entries corresponding to the merged blocks'
/// keys.
///
/// The new sample labels will contains all of the merged blocks sample
/// labels. The order of the samples is controlled by `sort_samples`. If
/// `sort_samples` is true, samples are re-ordered to keep them
/// lexicographically sorted. Otherwise they are kept in the order in which
/// they appear in the blocks.
///
/// This function is only implemented if all merged block have the same
/// property labels.
///
/// @param tensor pointer to an existing tensor map
/// @param keys_to_move description of the keys to move
/// @param sort_samples whether to sort the samples lexicographically after
///                     merging blocks or not
///
/// @returns The status code of this operation. If the status is not
///          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn eqs_tensormap_keys_to_samples(
    tensor: *mut eqs_tensormap_t,
    keys_to_move: eqs_labels_t,
    sort_samples: bool,
) -> eqs_status_t {
    catch_unwind(|| {
        check_pointers!(tensor);

        let keys_to_move = Labels::try_from(&keys_to_move)?;
        (*tensor).keys_to_samples(&keys_to_move, sort_samples)?;

        Ok(())
    })
}
