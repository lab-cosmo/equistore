use std::ops::Range;
use std::os::raw::c_void;
use std::sync::Mutex;

use once_cell::sync::Lazy;

use crate::c_api::{eqs_status_t, catch_unwind};
use crate::{Error, check_pointers};

/// A single 64-bit integer representing a data origin (numpy ndarray, rust
/// ndarray, torch tensor, fortran array, ...).
#[repr(transparent)]
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct eqs_data_origin_t(pub u64);

pub type DataOrigin = eqs_data_origin_t;

static REGISTERED_DATA_ORIGIN: Lazy<Mutex<Vec<String>>> = Lazy::new(|| {
    // start the registered origins at 1, this allow using 0 as a marker for
    // "unknown data origin"
    Mutex::new(vec!["unregistered origin".into()])
});

/// Register a new data origin with the given `name`, or get the
/// `DataOrigin` corresponding to this name if it was already registered.
pub fn register_data_origin(name: String) -> DataOrigin {
    let mut registered_origins = REGISTERED_DATA_ORIGIN.lock().expect("mutex got poisoned");

    for (i, registered) in registered_origins.iter().enumerate() {
        if registered == &name {
            return eqs_data_origin_t(i as u64);
        }
    }

    // could not find the origin, register a new one
    registered_origins.push(name);

    return eqs_data_origin_t((registered_origins.len() - 1) as u64);
}

/// Get the name of the given (pre-registered) origin
#[allow(clippy::cast_possible_truncation)]
pub fn get_data_origin(origin: DataOrigin) -> String {
    let registered_origins = REGISTERED_DATA_ORIGIN.lock().expect("mutex got poisoned");
    let id = origin.0 as usize;
    if id < registered_origins.len() {
        return registered_origins[id].clone();
    } else {
        return registered_origins[0].clone();
    }
}

// SAFETY: this should be checked by the user/implementor of `eqs_array_t`. On
// the rust side, the `Array` trait requires Send + Sync
unsafe impl Sync for eqs_array_t {}
unsafe impl Send for eqs_array_t {}

/// `eqs_array_t` manages n-dimensional arrays used as data in a block or tensor
/// map. The array itself if opaque to this library and can come from multiple
/// sources: Rust program, a C/C++ program, a Fortran program, Python with numpy
/// or torch. The data does not have to live on CPU, or even on the same machine
/// where this code is executed.
///
/// This struct contains a C-compatible manual implementation of a virtual table
/// (vtable, i.e. trait in Rust, pure virtual class in C++); allowing
/// manipulation of the array in an opaque way.
///
/// **WARNING**: all function implementations **MUST** be thread-safe, and can
/// be called from multiple threads at the same time. The `eqs_array_t` itself
/// might be moved from one thread to another.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct eqs_array_t {
    /// User-provided data should be stored here, it will be passed as the
    /// first parameter to all function pointers below.
    pub ptr: *mut c_void,

    /// This function needs to store the "data origin" for this array in
    /// `origin`. Users of `eqs_array_t` should register a single data
    /// origin with `eqs_register_data_origin`, and use it for all compatible
    /// arrays.
    origin: Option<unsafe extern fn(
        array: *const c_void,
        origin: *mut eqs_data_origin_t
    ) -> eqs_status_t>,

    /// Get a pointer to the underlying data storage.
    ///
    /// This function is allowed to fail if the data is not accessible in RAM,
    /// not stored as 64-bit floating point values, or not stored as a
    /// C-contiguous array.
    data: Option<unsafe extern fn(
        array: *const c_void,
        data: *mut *const f64,
    ) -> eqs_status_t>,

    /// Get the shape of the array managed by this `eqs_array_t` in the `*shape`
    /// pointer, and the number of dimension (size of the `*shape` array) in
    /// `*shape_count`.
    shape: Option<unsafe extern fn(
        array: *const c_void,
        shape: *mut *const usize,
        shape_count: *mut usize,
    ) -> eqs_status_t>,

    /// Change the shape of the array managed by this `eqs_array_t` to the given
    /// `shape`. `shape_count` must contain the number of elements in the
    /// `shape` array
    reshape: Option<unsafe extern fn(
        array: *mut c_void,
        shape: *const usize,
        shape_count: usize,
    ) -> eqs_status_t>,

    /// Swap the axes `axis_1` and `axis_2` in this `array`.
    swap_axes: Option<unsafe extern fn(
        array: *mut c_void,
        axis_1: usize,
        axis_2: usize,
    ) -> eqs_status_t>,

    /// Create a new array with the same options as the current one (data type,
    /// data location, etc.) and the requested `shape`; and store it in
    /// `new_array`. The number of elements in the `shape` array should be given
    /// in `shape_count`.
    ///
    /// The new array should be filled with zeros.
    create: Option<unsafe extern fn(
        array: *const c_void,
        shape: *const usize,
        shape_count: usize,
        new_array: *mut eqs_array_t,
    ) -> eqs_status_t>,

    /// Make a copy of this `array` and return the new array in `new_array`.
    ///
    /// The new array is expected to have the same data origin and parameters
    /// (data type, data location, etc.)
    copy: Option<unsafe extern fn(
        array: *const c_void,
        new_array: *mut eqs_array_t,
    ) -> eqs_status_t>,

    /// Remove this array and free the associated memory. This function can be
    /// set to `NULL` is there is no memory management to do.
    destroy: Option<unsafe extern fn(array: *mut c_void)>,

    /// Set entries in the `output` array (the current array) taking data from
    /// the `input` array. The `output` array is guaranteed to be created by
    /// calling `eqs_array_t::create` with one of the arrays in the same block
    /// or tensor map as the `input`.
    ///
    /// The `samples` array of size `samples_count` indicate where the data
    /// should be moved from `input` to `output`.
    ///
    /// This function should copy data from `input[samples[i].input, ..., :]` to
    /// `array[samples[i].output, ..., property_start:property_end]` for `i` up
    /// to `samples_count`. All indexes are 0-based.
    move_samples_from: Option<unsafe extern fn(
        output: *mut c_void,
        input: *const c_void,
        samples: *const eqs_sample_mapping_t,
        samples_count: usize,
        property_start: usize,
        property_end: usize,
    ) -> eqs_status_t>,
}

/// Representation of a single sample moved from an array to another one
#[derive(Debug, Clone)]
#[repr(C)]
pub struct eqs_sample_mapping_t {
    /// index of the moved sample in the input array
    pub input: usize,
    /// index of the moved sample in the output array
    pub output: usize,
}

impl std::fmt::Debug for eqs_array_t {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut origin = None;
        let mut shape = None;
        if !self.ptr.is_null() {
            origin = self.origin().ok().map(get_data_origin);
            shape = self.shape().ok();
        }

        f.debug_struct("eqs_array_t")
            .field("ptr", &self.ptr)
            .field("origin", &origin)
            .field("shape", &shape)
            .finish()
    }
}

impl Drop for eqs_array_t {
    fn drop(&mut self) {
        if let Some(function) = self.destroy {
            unsafe { function(self.ptr) }
        }
    }
}

impl Clone for eqs_array_t {
    fn clone(&self) -> eqs_array_t {
        if let Some(function) = self.copy {
            let mut new_array = eqs_array_t::null();
            let status  = unsafe { function(self.ptr, &mut new_array) };
            assert!(status.is_success(), "calling eqs_array_t.copy failed");
            return new_array;
        } else {
            panic!("function eqs_array_t.copy is not set")
        }
    }
}

impl eqs_array_t {
    /// Create an `eqs_array_t` from a Rust implementation of the `Array`
    /// trait.
    pub fn new(value: Box<dyn Array>) -> eqs_array_t {
        // We need to box the box to make sure the pointer is a normal 1-word
        // pointer (`Box<dyn Trait>` contains a 2-words, *fat* pointer which can
        // not be casted to `*mut c_void`)
        let array = Box::new(value);
        eqs_array_t {
            ptr: Box::into_raw(array).cast(),
            origin: Some(rust_array_origin),
            data: Some(rust_array_data),
            shape: Some(rust_array_shape),
            reshape: Some(rust_array_reshape),
            swap_axes: Some(rust_array_swap_axes),
            create: Some(rust_array_create),
            copy: Some(rust_array_copy),
            destroy: Some(rust_array_destroy),
            move_samples_from: Some(rust_array_move_samples_from),
        }
    }

    /// make a raw (member by member) copy of the array. Contrary to
    /// `eqs_array_t::clone`, the returned array refers to the same
    /// `eqs_array_t` instance, and as such should not be freed.
    pub(crate) fn raw_copy(&self) -> eqs_array_t {
        eqs_array_t {
            ptr: self.ptr,
            origin: self.origin,
            data: self.data,
            shape: self.shape,
            reshape: self.reshape,
            swap_axes: self.swap_axes,
            create: self.create,
            copy: self.copy,
            destroy: None,
            move_samples_from: self.move_samples_from,
        }
    }

    /// Create an `eqs_array_t` with all fields set to null pointers.
    fn null() -> eqs_array_t {
        eqs_array_t {
            ptr: std::ptr::null_mut(),
            origin: None,
            data: None,
            shape: None,
            reshape: None,
            swap_axes: None,
            create: None,
            copy: None,
            destroy: None,
            move_samples_from: None,
        }
    }

    /// Get the origin of this array
    pub fn origin(&self) -> Result<eqs_data_origin_t, Error> {
        let function = self.origin.expect("eqs_array_t.origin function is NULL");

        let mut origin = eqs_data_origin_t(0);
        let status = unsafe {
            function(self.ptr, &mut origin)
        };

        if !status.is_success() {
            return Err(Error::External {
                status, context: "calling eqs_array_t.origin failed".into()
            });
        }

        return Ok(origin);
    }

    /// Get the underlying data for this array.
    pub fn data(&self) -> Result<&[f64], Error> {
        let shape = self.shape()?;
        let mut len = 1;
        for s in shape {
            len *= s;
        }

        let function = self.data.expect("eqs_array_t.data function is NULL");

        let mut data_ptr = std::ptr::null();

        let status = unsafe {
            function(
                self.ptr,
                &mut data_ptr,
            )
        };

        if !status.is_success() {
            return Err(Error::External {
                status, context: "calling eqs_array_t.data failed".into()
            });
        }

        let data = unsafe {
            std::slice::from_raw_parts(data_ptr, len)
        };

        return Ok(data);
    }

    /// Get the shape of this array
    #[allow(clippy::cast_possible_truncation)]
    pub fn shape(&self) -> Result<&[usize], Error> {
        let function = self.shape.expect("eqs_array_t.shape function is NULL");

        let mut shape = std::ptr::null();
        let mut shape_count: usize = 0;

        let status = unsafe {
            function(
                self.ptr,
                &mut shape,
                &mut shape_count,
            )
        };

        if !status.is_success() {
            return Err(Error::External {
                status, context: "calling eqs_array_t.shape failed".into()
            });
        }

        let shape = unsafe {
            std::slice::from_raw_parts(shape, shape_count)
        };

        return Ok(shape);
    }

    /// Set the shape of this array to the given new `shape`
    pub fn reshape(&mut self, shape: &[usize]) -> Result<(), Error> {
        let function = self.reshape.expect("eqs_array_t.reshape function is NULL");

        let status = unsafe {
            function(
                self.ptr,
                shape.as_ptr(),
                shape.len(),
            )
        };

        if !status.is_success() {
            return Err(Error::External {
                status, context: "calling eqs_array_t.reshape failed".into()
            });
        }

        return Ok(());
    }

    /// Swap the axes `axis_1` and `axis_2` in the dimensions of this array.
    pub fn swap_axes(&mut self, axis_1: usize, axis_2: usize) -> Result<(), Error> {
        let function = self.swap_axes.expect("eqs_array_t.swap_axes function is NULL");

        let status = unsafe {
            function(
                self.ptr,
                axis_1,
                axis_2,
            )
        };

        if !status.is_success() {
            return Err(Error::External {
                status, context: "calling eqs_array_t.swap_axes failed".into()
            });
        }

        return Ok(());
    }

    /// Create a new array with the same settings as this one and the given `shape`
    pub fn create(&self, shape: &[usize]) -> Result<eqs_array_t, Error> {
        let function = self.create.expect("eqs_array_t.create function is NULL");

        let mut data_storage = eqs_array_t::null();
        let status = unsafe {
            function(
                self.ptr,
                shape.as_ptr(),
                shape.len(),
                &mut data_storage
            )
        };

        if !status.is_success() {
            return Err(Error::External {
                status, context: "calling eqs_array_t.create failed".into()
            });
        }

        return Ok(data_storage);
    }

    /// Set entries in `self` (the current array) taking data from the `input`
    /// array. The `self` array is guaranteed to be created by calling
    /// `Array::create` with one of the arrays in the same block or tensor
    /// map as the `input`.
    ///
    /// The `samples` array indicate where the data should be moved from `input`
    /// to `output`.
    ///
    /// This function should copy data from `input[sample.input, ..., :]` to
    /// `array[sample.output, ..., properties]` for all `sample` in `samples`.
    /// All indexes are 0-based.
    pub fn move_samples_from(
        &mut self,
        input: &eqs_array_t,
        samples: &[eqs_sample_mapping_t],
        properties: Range<usize>,
    ) -> Result<(), Error> {
        let function = self.move_samples_from.expect("eqs_array_t.move_samples_from function is NULL");

        let status = unsafe {
            function(
                self.ptr,
                input.ptr,
                samples.as_ptr(),
                samples.len(),
                properties.start,
                properties.end,
            )
        };

        if !status.is_success() {
            return Err(Error::External {
                status, context: "calling eqs_array_t.move_samples_from failed".into()
            });
        }

        return Ok(());
    }

    /// Get the data in this `eqs_array_t` as a `ndarray::ArrayD`. This function
    /// will panic if the data in this `eqs_array_t` is not a `ndarray::ArrayD`.
    #[cfg(feature = "ndarray")]
    pub fn as_array(&self) -> &ndarray::ArrayD<f64> {
        assert_eq!(
            self.origin().unwrap_or(eqs_data_origin_t(0)), *NDARRAY_DATA_ORIGIN,
            "this array was not created by rust ndarray"
        );

        let array = self.ptr.cast::<Box<dyn Array>>();
        unsafe {
            (*array).as_any().downcast_ref().expect("invalid array type")
        }
    }

    #[cfg(feature = "ndarray")]
    pub fn as_array_mut(&mut self) -> &mut ndarray::ArrayD<f64> {
        assert_eq!(
            self.origin().unwrap_or(eqs_data_origin_t(0)), *NDARRAY_DATA_ORIGIN,
            "this array was not created by rust ndarray"
        );

        let array = self.ptr.cast::<Box<dyn Array>>();
        unsafe {
            (*array).as_any_mut().downcast_mut().expect("invalid array type")
        }
    }
}

impl<A: Array> From<A> for eqs_array_t {
    fn from(array: A) -> Self {
        eqs_array_t::new(Box::new(array))
    }
}

/// A rust trait with the same interface as `eqs_array_t`, see this struct for
/// documentation on all methods.
pub trait Array: std::any::Any + Send + Sync {
    // TODO: add docs to this trait directly

    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    fn origin(&self) -> DataOrigin;

    fn create(&self, shape: &[usize]) -> Box<dyn Array>;

    fn copy(&self) -> Box<dyn Array>;

    fn data(&self) -> &[f64];

    fn shape(&self) -> &[usize];
    fn reshape(&mut self, shape: &[usize]);
    fn swap_axes(&mut self, axis_1: usize, axis_2: usize);

    fn move_samples_from(
        &mut self,
        input: &dyn Array,
        samples: &[eqs_sample_mapping_t],
        properties: Range<usize>,
    );
}

/// Implementation of `eqs_array_t.origin` using `Box<dyn Array>`
unsafe extern fn rust_array_origin(
    array: *const c_void,
    origin: *mut eqs_data_origin_t
) -> eqs_status_t {
    catch_unwind(|| {
        check_pointers!(array, origin);
        let array = array.cast::<Box<dyn Array>>();

        *origin = (*array).origin();
        Ok(())
    })
}

/// Implementation of `eqs_array_t.shape` using `Box<dyn Array>`
unsafe extern fn rust_array_shape(
    array: *const c_void,
    shape: *mut *const usize,
    shape_count: *mut usize,
) -> eqs_status_t {
    catch_unwind(|| {
        check_pointers!(array, shape, shape_count);
        let array = array.cast::<Box<dyn Array>>();
        let rust_shape = (*array).shape();

        *shape = rust_shape.as_ptr();
        *shape_count = rust_shape.len();

        Ok(())
    })
}

/// Implementation of `eqs_array_t.reshape` using `Box<dyn Array>`
#[allow(clippy::cast_possible_truncation)]
unsafe extern fn rust_array_reshape(
    array: *mut c_void,
    shape: *const usize,
    shape_count: usize,
) -> eqs_status_t {
    catch_unwind(|| {
        check_pointers!(array);
        let array = array.cast::<Box<dyn Array>>();
        let shape = std::slice::from_raw_parts(shape, shape_count);
        (*array).reshape(shape);
        Ok(())
    })
}

/// Implementation of `eqs_array_t.swap_axes` using `Box<dyn Array>`
#[allow(clippy::cast_possible_truncation)]
unsafe extern fn rust_array_swap_axes(
    array: *mut c_void,
    axis_1: usize,
    axis_2: usize,
) -> eqs_status_t {
    catch_unwind(|| {
        check_pointers!(array);
        let array = array.cast::<Box<dyn Array>>();
        (*array).swap_axes(axis_1, axis_2);
        Ok(())
    })
}

/// Implementation of `eqs_array_t.create` using `Box<dyn Array>`
#[allow(clippy::cast_possible_truncation)]
unsafe extern fn rust_array_create(
    array: *const c_void,
    shape: *const usize,
    shape_count: usize,
    array_storage: *mut eqs_array_t,
) -> eqs_status_t {
    catch_unwind(|| {
        check_pointers!(array, array_storage);
        let array = array.cast::<Box<dyn Array>>();

        let shape = std::slice::from_raw_parts(shape, shape_count);
        let new_array = (*array).create(shape);

        *array_storage = eqs_array_t::new(new_array);

        Ok(())
    })
}

/// Implementation of `eqs_array_t.data` for `Box<dyn Array>`
unsafe extern fn rust_array_data(
    array: *const c_void,
    data: *mut *const f64,
) -> eqs_status_t {
    catch_unwind(|| {
        check_pointers!(array, data);
        let array = array.cast::<Box<dyn Array>>();
        *data = (*array).data().as_ptr();
        Ok(())
    })
}


/// Implementation of `eqs_array_t.copy` using `Box<dyn Array>`
unsafe extern fn rust_array_copy(
    array: *const c_void,
    array_storage: *mut eqs_array_t,
) -> eqs_status_t {
    catch_unwind(|| {
        check_pointers!(array, array_storage);
        let array = array.cast::<Box<dyn Array>>();
        *array_storage = eqs_array_t::new((*array).copy());

        Ok(())
    })
}

/// Implementation of `eqs_array_t.destroy` for `Box<dyn Array>`
unsafe extern fn rust_array_destroy(
    array: *mut c_void,
) {
    if !array.is_null() {
        let array = array.cast::<Box<dyn Array>>();
        let boxed = Box::from_raw(array);
        std::mem::drop(boxed);
    }
}

/// Implementation of `eqs_array_t.move_sample` using `Box<dyn Array>`
#[allow(clippy::cast_possible_truncation)]
unsafe extern fn rust_array_move_samples_from(
    output: *mut c_void,
    input: *const c_void,
    samples: *const eqs_sample_mapping_t,
    samples_count: usize,
    property_start: usize,
    property_end: usize,
) -> eqs_status_t {
    catch_unwind(|| {
        check_pointers!(output, input);
        let output = output.cast::<Box<dyn Array>>();
        let input = input.cast::<Box<dyn Array>>();

        let samples = std::slice::from_raw_parts(samples, samples_count);
        (*output).move_samples_from(&**input, samples, property_start..property_end);

        Ok(())
    })
}

/******************************************************************************/

#[cfg(feature = "ndarray")]
static NDARRAY_DATA_ORIGIN: Lazy<DataOrigin> = Lazy::new(|| {
    register_data_origin("rust.ndarray".into())
});


#[cfg(feature = "ndarray")]
impl Array for ndarray::ArrayD<f64> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn origin(&self) -> DataOrigin {
        return *NDARRAY_DATA_ORIGIN;
    }

    fn create(&self, shape: &[usize]) -> Box<dyn Array> {
        return Box::new(ndarray::Array::from_elem(shape, 0.0));
    }

    fn copy(&self) -> Box<dyn Array> {
        return Box::new(self.clone());
    }

    fn data(&self) -> &[f64] {
        return self.as_slice().expect("array is not contiguous")
    }

    fn shape(&self) -> &[usize] {
        return self.shape();
    }

    fn reshape(&mut self, shape: &[usize]) {
        let mut array = std::mem::take(self);
        array = array.to_shape(shape).expect("invalid shape").to_owned();
        std::mem::swap(self, &mut array);
    }

    fn swap_axes(&mut self, axis_1: usize, axis_2: usize) {
        self.swap_axes(axis_1, axis_2);
    }

    fn move_samples_from(
        &mut self,
        input: &dyn Array,
        samples: &[eqs_sample_mapping_t],
        property: Range<usize>,
    ) {
        use ndarray::{Axis, Slice};

        // -2 since we also remove one axis with `index_axis_mut` below
        let property_axis = self.shape().len() - 2;

        let input = input.as_any().downcast_ref::<ndarray::ArrayD<f64>>().expect("input must be a ndarray");
        for sample in samples {
            let value = input.index_axis(Axis(0), sample.input);

            let mut output_location = self.index_axis_mut(Axis(0), sample.output);
            let mut output_location = output_location.slice_axis_mut(
                Axis(property_axis), Slice::from(property.clone())
            );

            output_location.assign(&value);
        }
    }
}

/******************************************************************************/

static DUMMY_DATA_ORIGIN: Lazy<DataOrigin> = Lazy::new(|| {
    register_data_origin("rust.EmptyArray".into())
});

/// An implementation of the `Array` trait without any data. This only
/// tracks the shape of the array.
pub struct EmptyArray {
    shape: Vec<usize>,
}

impl EmptyArray {
    /// Create a new `EmptyArray` with the given shape.
    pub fn new(shape: Vec<usize>) -> EmptyArray {
        EmptyArray { shape }
    }
}

impl Array for EmptyArray {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn origin(&self) -> crate::DataOrigin {
        *DUMMY_DATA_ORIGIN
    }

    fn data(&self) -> &[f64] {
        panic!("can not call Array::data() for EmptyArray");
    }

    fn create(&self, shape: &[usize]) -> Box<dyn Array> {
        Box::new(EmptyArray { shape: shape.to_vec() })
    }

    fn copy(&self) -> Box<dyn Array> {
        Box::new(EmptyArray { shape: self.shape.clone() })
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn reshape(&mut self, shape: &[usize]) {
        self.shape = shape.to_vec();
    }

    fn swap_axes(&mut self, axis_1: usize, axis_2: usize) {
        self.shape.swap(axis_1, axis_2);
    }

    fn move_samples_from(&mut self, _: &dyn Array, _: &[eqs_sample_mapping_t], _: Range<usize>) {
        panic!("can not call Array::move_samples_from() for EmptyArray");
    }
}

/******************************************************************************/

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_origin() {
        assert_eq!(get_data_origin(eqs_data_origin_t(0)), "unregistered origin");
        assert_eq!(get_data_origin(eqs_data_origin_t(10000)), "unregistered origin");

        let origin = register_data_origin("test origin".into());
        assert_eq!(get_data_origin(origin), "test origin");
    }

    #[test]
    fn debug() {
        let data: eqs_array_t = EmptyArray::new(vec![3, 4, 5]).into();

        let debug_format = format!("{:?}", data);
        assert_eq!(debug_format, format!(
            "eqs_array_t {{ ptr: {:?}, origin: Some(\"rust.EmptyArray\"), shape: Some([3, 4, 5]) }}",
            data.ptr
        ));
    }

    #[cfg(feature = "ndarray")]
    mod ndarray {
        use ndarray::ArrayD;

        use crate::{eqs_array_t, get_data_origin, eqs_sample_mapping_t};

        #[test]
        fn shape() {
            let mut data: eqs_array_t = ArrayD::from_elem(vec![3, 4, 2], 1.0).into();

            assert_eq!(data.shape().unwrap(), [3, 4, 2]);
            data.reshape(&[12, 2]).unwrap();
            assert_eq!(data.shape().unwrap(), [12, 2]);
            assert_eq!(data.as_array(), ArrayD::from_elem(vec![12, 2], 1.0));

            data.swap_axes(0, 1).unwrap();
            assert_eq!(data.shape().unwrap(), [2, 12]);
        }

        #[test]
        fn create() {
            let data: eqs_array_t = ArrayD::from_elem(vec![4, 2], 1.0).into();

            assert_eq!(get_data_origin(data.origin().unwrap()), "rust.ndarray");
            assert_eq!(data.as_array(), ArrayD::from_elem(vec![4, 2], 1.0));

            let other = data.create(&[5, 3, 7, 12]).unwrap();
            assert_eq!(other.shape().unwrap(), [5, 3, 7, 12]);
            assert_eq!(get_data_origin(other.origin().unwrap()), "rust.ndarray");
            assert_eq!(other.as_array(), ArrayD::from_elem(vec![5, 3, 7, 12], 0.0));
        }

        #[test]
        fn move_samples_from() {
            let data: eqs_array_t = ArrayD::from_elem(vec![3, 2, 2, 4], 1.0).into();

            let mut other = data.create(&[1, 2, 2, 8]).unwrap();
            assert_eq!(other.as_array(), ArrayD::from_elem(vec![1, 2, 2, 8], 0.0));

            let mapping = eqs_sample_mapping_t {
                output: 0,
                input: 1,
            };
            other.move_samples_from(&data, &[mapping], 2..6).unwrap();
            let expected = ArrayD::from_shape_vec(vec![1, 2, 2, 8], vec![
                 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
            ]).unwrap();
            assert_eq!(other.as_array(), expected);
        }
    }
}
