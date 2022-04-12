/* ============    Automatically generated file, DOT NOT EDIT.    ============ *
 *                                                                             *
 *    This file is automatically generated from the aml-storage sources,       *
 *    using cbindgen. If you want to make change to this file (including       *
 *    documentation), make the corresponding changes in the rust sources.      *
 * =========================================================================== */

#ifndef AML_STORAGE_H
#define AML_STORAGE_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Status code used when a function succeeded
 */
#define AML_SUCCESS 0

/**
 * Status code used when a function got an invalid parameter
 */
#define AML_INVALID_PARAMETER_ERROR 1

/**
 * Status code used when a memory buffer is too small to fit the requested data
 */
#define AML_BUFFER_SIZE_ERROR 254

/**
 * Status code used when there was an internal error, i.e. there is a bug
 * inside AML itself
 */
#define AML_INTERNAL_ERROR 255

/**
 * Basic building block for tensor map. A single block contains a n-dimensional
 * `aml_array_t`, and n sets of `aml_labels_t` (one for each dimension).
 *
 * A block can also contain gradients of the values with respect to a variety
 * of parameters. In this case, each gradient has a separate set of sample
 * and component labels but share the property labels with the values.
 */
typedef struct aml_block_t aml_block_t;

/**
 * Opaque type representing a `TensorMap`.
 */
typedef struct aml_tensormap_t aml_tensormap_t;

/**
 * Status type returned by all functions in the C API.
 *
 * The value 0 (`AML_SUCCESS`) is used to indicate successful operations,
 * positive values are used by this library to indicate errors, while negative
 * values are reserved for users of this library to indicate their own errors
 * in callbacks.
 */
typedef int32_t aml_status_t;

/**
 * A set of labels used to carry metadata associated with a tensor map.
 *
 * This is similar to a list of `count` named tuples, but stored as a 2D array
 * of shape `(count, size)`, with a set of names associated with the columns of
 * this array (often called *variables*). Each row/entry in this array is
 * unique, and they are often (but not always) sorted in lexicographic order.
 */
typedef struct aml_labels_t {
  /**
   * internal: pointer to the rust `Labels` struct if any, null otherwise
   */
  const void *labels_ptr;
  /**
   * Names of the variables composing this set of labels. There are `size`
   * elements in this array, each being a NULL terminated UTF-8 string.
   */
  const char *const *names;
  /**
   * Pointer to the first element of a 2D row-major array of 32-bit signed
   * integer containing the values taken by the different variables in
   * `names`. Each row has `size` elements, and there are `count` rows in
   * total.
   */
  const int32_t *values;
  /**
   * Number of variables/size of a single entry in the set of labels
   */
  uintptr_t size;
  /**
   * Number entries in the set of labels
   */
  uintptr_t count;
} aml_labels_t;

/**
 * A single 64-bit integer representing a data origin (numpy ndarray, rust
 * ndarray, torch tensor, fortran array, ...).
 */
typedef uint64_t aml_data_origin_t;

/**
 * `aml_array_t` manages 3D arrays the be used as data in a block/tensor map.
 * The array itself if opaque to this library and can come from multiple
 * sources: Rust program, a C/C++ program, a Fortran program, Python with numpy
 * or torch. The data does not have to live on CPU, or even on the same machine
 * where this code is executed.
 *
 * This struct contains a C-compatible manual implementation of a virtual table
 * (vtable, i.e. trait in Rust, pure virtual class in C++); allowing
 * manipulation of the array in an opaque way.
 */
typedef struct aml_array_t {
  /**
   * User-provided data should be stored here, it will be passed as the
   * first parameter to all function pointers below.
   */
  void *ptr;
  /**
   * This function needs to store the "data origin" for this array in
   * `origin`. Users of `aml_array_t` should register a single data
   * origin with `register_data_origin`, and use it for all compatible
   * arrays.
   */
  aml_status_t (*origin)(const void *array, aml_data_origin_t *origin);
  /**
   * Get the shape of the array managed by this `aml_array_t` in the `*shape`
   * pointer, and the number of dimension (size of the `*shape` array) in
   * `*shape_count`.
   */
  aml_status_t (*shape)(const void *array, const uintptr_t **shape, uintptr_t *shape_count);
  /**
   * Change the shape of the array managed by this `aml_array_t` to the given
   * `shape`. `shape_count` must contain the number of elements in the
   * `shape` array
   */
  aml_status_t (*reshape)(void *array, const uintptr_t *shape, uintptr_t shape_count);
  /**
   * Swap the axes `axis_1` and `axis_2` in this `array`.
   */
  aml_status_t (*swap_axes)(void *array, uintptr_t axis_1, uintptr_t axis_2);
  /**
   * Create a new array with the same options as the current one (data type,
   * data location, etc.) and the requested `shape`; and store it in
   * `new_array`. The number of elements in the `shape` array should be given
   * in `shape_count`.
   *
   * The new array should be filled with zeros.
   */
  aml_status_t (*create)(const void *array, const uintptr_t *shape, uintptr_t shape_count, struct aml_array_t *new_array);
  /**
   * Make a copy of this `array` and return the new array in `new_array`
   */
  aml_status_t (*copy)(const void *array, struct aml_array_t *new_array);
  /**
   * Remove this array and free the associated memory. This function can be
   * set to `NULL` is there is no memory management to do.
   */
  void (*destroy)(void *array);
  /**
   * Set entries in this array taking data from the `other_array`. This array
   * is guaranteed to be created by calling `aml_array_t::create` with one of
   * the arrays in the same block or tensor map as this `array`.
   *
   * This function should copy data from `other_array[other_sample, ..., :]` to
   * `array[sample, ..., property_start:property_end]`. All indexes are 0-based.
   */
  aml_status_t (*move_sample)(void *array, uint64_t sample, uint64_t property_start, uint64_t property_end, const void *other_array, uint64_t other_sample);
} aml_array_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Get the last error message that was created on the current thread.
 *
 * @returns the last error message, as a NULL-terminated string
 */
const char *aml_last_error(void);

/**
 * Get the position of the entry defined by the `values` array in the given set
 * of `labels`. This operation is only available if the labels correspond to a
 * set of Rust Labels (i.e. `labels.labels_ptr` is not NULL).
 *
 * @param labels set of labels coming from an `aml_block_t` or an `aml_tensormap_t`
 * @param values array containing the label to lookup
 * @param count size of the values array
 * @param result position of the values in the labels or -1 if the values
 *               were not found
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_labels_position(struct aml_labels_t labels,
                                 const int32_t *values,
                                 uint64_t count,
                                 int64_t *result);

/**
 * Register a new data origin with the given `name`. Calling this function
 * multiple times with the same name will give the same `aml_data_origin_t`.
 *
 * @param name name of the data origin as an UTF-8 encoded NULL-terminated string
 * @param origin pointer to an `aml_data_origin_t` where the origin will be stored
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_register_data_origin(const char *name, aml_data_origin_t *origin);

/**
 * Get the name used to register a given data `origin` in the given `buffer`
 *
 * @param origin pre-registered data origin
 * @param buffer buffer to be filled with the data origin name. The origin name
 *               will be written  as an UTF-8 encoded, NULL-terminated string
 * @param buffer_size size of the buffer
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_get_data_origin(aml_data_origin_t origin, char *buffer, uint64_t buffer_size);

/**
 * Create a new `aml_block_t` with the given `data` and `samples`, `components`
 * and `properties` labels.
 *
 * The memory allocated by this function and the blocks should be released
 * using `aml_block_free`, or moved into a tensor map using `aml_tensormap`.
 *
 * @param data array handle containing the data for this block. The block takes
 *             ownership of the array, and will release it with
 *             `array.destroy(array.ptr)` when it no longer needs it.
 * @param samples sample labels corresponding to the first dimension of the data
 * @param components array of component labels corresponding to intermediary
 *                   dimensions of the data
 * @param components_count number of entries in the `components` array
 * @param properties property labels corresponding to the last dimension of the data
 *
 * @returns A pointer to the newly allocated block, or a `NULL` pointer in
 *          case of error. In case of error, you can use `aml_last_error()`
 *          to get the error message.
 */
struct aml_block_t *aml_block(struct aml_array_t data,
                              struct aml_labels_t samples,
                              const struct aml_labels_t *components,
                              uintptr_t components_count,
                              struct aml_labels_t properties);

/**
 * Free the memory associated with a `block` previously created with
 * `aml_block`.
 *
 * If `block` is `NULL`, this function does nothing.
 *
 * @param block pointer to an existing block, or `NULL`
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_block_free(struct aml_block_t *block);

/**
 * Make a copy of an `aml_block_t`.
 *
 * The memory allocated by this function and the blocks should be released
 * using `aml_block_free`, or moved into a tensor map using `aml_tensormap`.
 *
 * @param block existing block to copy
 *
 * @returns A pointer to the newly allocated block, or a `NULL` pointer in
 *          case of error. In case of error, you can use `aml_last_error()`
 *          to get the error message.
 */
struct aml_block_t *aml_block_copy(const struct aml_block_t *block);

/**
 * Get the set of labels of the requested `kind` from this `block`.
 *
 * The `values_gradients` parameter controls whether this function looks up
 * labels for `"values"` or one of the gradients in this block.
 *
 * The resulting `labels.values` points inside memory owned by the block, and
 * as such is only valid until the block is destroyed with `aml_block_free`, or
 * the containing tensor map is modified with one of the
 * `aml_tensormap_keys_to_xxx` function.
 *
 * @param block pointer to an existing block
 * @param values_gradients either `"values"` or the name of gradients to lookup
 * @param axis axis/dimension of the data array for which you need the labels
 * @param labels pointer to an empty `aml_labels_t` that will be set to the
 *               requested labels
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_block_labels(const struct aml_block_t *block,
                              const char *values_gradients,
                              uintptr_t axis,
                              struct aml_labels_t *labels);

/**
 * Get the array handle for either values or one of the gradient in this `block`.
 *
 * The `values_gradients` parameter controls whether this function looks up
 * labels for `"values"` or one of the gradients in this block.
 *
 * @param block pointer to an existing block
 * @param values_gradients either `"values"` or the name of gradients to lookup
 * @param data pointer to an empty `aml_array_t` that will be set to the
 *             requested array
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_block_data(const struct aml_block_t *block,
                            const char *values_gradients,
                            struct aml_array_t *data);

/**
 * Add a new gradient to this `block` with the given `name`.
 *
 * @param block pointer to an existing block
 * @param data array containing the gradient data. The block takes
 *                 ownership of the array, and will release it with
 *                 `array.destroy(array.ptr)` when it no longer needs it.
 * @param parameter name of the gradient as a NULL-terminated UTF-8 string.
 *                  This is usually the parameter used when taking derivatives
 *                  (e.g. `"positions"`, `"cell"`, etc.)
 * @param samples sample labels for the gradient array. The components and
 *                property labels are supposed to match the values in this block
 * @param components array of component labels corresponding to intermediary
 *                   dimensions of the data
 * @param components_count number of entries in the `components` array
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_block_add_gradient(struct aml_block_t *block,
                                    const char *parameter,
                                    struct aml_array_t data,
                                    struct aml_labels_t samples,
                                    const struct aml_labels_t *components,
                                    uintptr_t components_count);

/**
 * Get a list of all gradients defined in this `block` in the `parameters` array.
 *
 * @param block pointer to an existing block
 * @param parameters will be set to the first element of an array of
 *                   NULL-terminated UTF-8 strings containing all the
 *                   parameters for which a gradient exists in the block
 * @param count will be set to the number of elements in `parameters`
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_block_gradients_list(struct aml_block_t *block,
                                      const char *const **parameters,
                                      uint64_t *count);

/**
 * Create a new `aml_tensormap_t` with the given `keys` and `blocks`.
 * `blocks_count` must be set to the number of entries in the blocks array.
 *
 * The new tensor map takes ownership of the blocks, which should not be
 * released separately.
 *
 * The memory allocated by this function and the blocks should be released
 * using `aml_tensormap_free`.
 *
 * @param keys labels containing the keys associated with each block
 * @param blocks pointer to the first element of an array of blocks
 * @param blocks_count number of elements in the `blocks` array
 *
 * @returns A pointer to the newly allocated tensor map, or a `NULL` pointer in
 *          case of error. In case of error, you can use `aml_last_error()`
 *          to get the error message.
 */
struct aml_tensormap_t *aml_tensormap(struct aml_labels_t keys,
                                      struct aml_block_t **blocks,
                                      uint64_t blocks_count);

/**
 * Free the memory associated with a `tensor` previously created with
 * `aml_tensormap`.
 *
 * If `tensor` is `NULL`, this function does nothing.
 *
 * @param tensor pointer to an existing tensor map, or `NULL`
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_tensormap_free(struct aml_tensormap_t *tensor);

/**
 * Get the keys for the given `tensor` map. After a successful call to this
 * function, `keys.values` contains a pointer to memory inside the
 * `tensor` which is invalidated when the tensor map is freed with
 * `aml_tensormap_free` or the set of keys is modified by calling one
 * of the `aml_tensormap_keys_to_XXX` function.
 * @param tensor pointer to an existing tensor map
 * @param keys pointer to be filled with the keys of the tensor map
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_tensormap_keys(const struct aml_tensormap_t *tensor, struct aml_labels_t *keys);

/**
 * Get a pointer to the `index`-th block in this tensor map.
 *
 * The block memory is still managed by the tensor map, this block should not
 * be freed. The block is invalidated when the tensor map is freed with
 * `aml_tensormap_free` or the set of keys is modified by calling one
 * of the `aml_tensormap_keys_to_XXX` function.
 *
 * @param tensor pointer to an existing tensor map
 * @param block pointer to be filled with a block
 * @param index index of the block to get
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_tensormap_block_by_id(const struct aml_tensormap_t *tensor,
                                       const struct aml_block_t **block,
                                       uint64_t index);

/**
 * Get a pointer to the `block` in this `tensor` corresponding to the given
 * `selection`. The `selection` should have the same names/variables as the
 * keys for this tensor map, and only one entry, describing the
 * requested block.
 *
 * The block memory is still managed by the tensor map, this block should not
 * be freed. The block is invalidated when the tensor map is freed with
 * `aml_tensormap_free` or the set of keys is modified by calling one
 * of the `aml_tensormap_keys_to_XXX` function.
 *
 * @param tensor pointer to an existing tensor map
 * @param block pointer to be filled with a block
 * @param selection labels with a single entry describing which block is requested
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_tensormap_block_selection(const struct aml_tensormap_t *tensor,
                                           const struct aml_block_t **block,
                                           struct aml_labels_t selection);

/**
 * Move the given `variables` from the keys to the property labels of the
 * blocks.
 *
 * Blocks containing the same values in the keys for the `variables` will
 * be merged together. The resulting merged blocks will have `variables` as
 * the first property variables, followed by the current properties. The
 * new sample labels will contains all of the merged blocks sample labels,
 * re-ordered to keep them lexicographically sorted.
 *
 * `variables` must be an array of `variables_count` NULL-terminated strings,
 * encoded as UTF-8.
 *
 * @param tensor pointer to an existing tensor map
 * @param variables names of the key variables to move to the properties
 * @param variables_count number of entries in the `variables` array
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_tensormap_keys_to_properties(struct aml_tensormap_t *tensor,
                                              const char *const *variables,
                                              uint64_t variables_count);

/**
 * Move the given variables from the component labels to the property labels
 * for each block in this tensor map.
 *
 * `variables` must be an array of `variables_count` NULL-terminated strings,
 * encoded as UTF-8.
 *
 * @param tensor pointer to an existing tensor map
 * @param variables names of the key variables to move to the properties
 * @param variables_count number of entries in the `variables` array
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_tensormap_components_to_properties(struct aml_tensormap_t *tensor,
                                                    const char *const *variables,
                                                    uint64_t variables_count);

/**
 * Move the given `variables` from the keys to the sample labels of the
 * blocks.
 *
 * Blocks containing the same values in the keys for the `variables` will
 * be merged together. The resulting merged blocks will have `variables` as
 * the last sample variables, preceded by the current samples.
 *
 * This function is only implemented if all merged block have the same
 * property labels.
 *
 * `variables` must be an array of `variables_count` NULL-terminated strings,
 * encoded as UTF-8.
 *
 * @param tensor pointer to an existing tensor map
 * @param variables names of the key variables to move to the samples
 * @param variables_count number of entries in the `variables` array
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_tensormap_keys_to_samples(struct aml_tensormap_t *tensor,
                                           const char *const *variables,
                                           uint64_t variables_count);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif /* AML_STORAGE_H */
