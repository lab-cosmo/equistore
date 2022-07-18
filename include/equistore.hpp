#ifndef EQUISTORE_HPP
#define EQUISTORE_HPP

#include <array>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <initializer_list>

#include <cassert>
#include <cstdint>
#include <cstring>

#include "equistore.h"

/// This file contains the C++ API to equistore, manually built on top of the C
/// API defined in `equistore.h`. This API uses the standard C++ library where
/// convenient, but also allow to drop back to the C API if required, by
/// providing functions to extract the C API handles (named `as_eqs_XXX`).

namespace equistore {
class Labels;
class TensorBlock;
class TensorMap;


/// Exception class used for all errors in equistore
class Error: public std::runtime_error {
public:
    /// Create a new Error with the given `message`
    Error(std::string message): std::runtime_error(std::move(message)) {}
};

namespace details {
    /// Check if a return status from the C API indicates an error, and if it is
    /// the case, throw an exception of type `equistore::Error` with the last
    /// error message from the library.
    inline void check_status(eqs_status_t status) {
        if (status == EQS_SUCCESS) {
            return;
        } else if (status > 0) {
            throw Error(eqs_last_error());
        } else if (status < 0) {
            throw Error("error in callback");
        } else {

        }
    }

    /// Check if a pointer allocated by the C API is null, and if it is the
    /// case, throw an exception of type `equistore::Error` with the last error
    /// message from the library.
    inline void check_pointer(const void* pointer) {
        if (pointer == nullptr) {
            throw Error(eqs_last_error());
        }
    }

    /// Compute the product of all values in the `shape` vector
    inline size_t product(const std::vector<size_t>& shape) {
        size_t result = 1;
        for (auto size: shape) {
            result *= size;
        }
        return result;
    }

    /// Get the N-dimensional index corresponding to the given linear `index`
    /// and array `shape`
    inline std::vector<size_t> cartesian_index(const std::vector<size_t>& shape, size_t index) {
        auto result = std::vector<size_t>(shape.size(), 0);
        for (size_t i=0; i<shape.size(); i++) {
            result[i] = index % shape[i];
            index = index / shape[i];
        }
        assert(index == 0);
        return result;
    }

    /// Get the linear index corresponding to the N-dimensional
    /// `index[index_size]`, according to the given array `shape`
    inline size_t linear_index(const std::vector<size_t>& shape, const size_t* index, size_t index_size) {
        assert(index_size != 0);
        assert(index_size == shape.size());

        if (index_size == 1) {
            assert(index[0] < shape[0] && "out of bounds");
            return index[0];
        } else {
            assert(index[0] < shape[0]);
            auto linear_index = index[0];
            for (size_t i=1; i<index_size; i++) {
                assert(index[i] < shape[i] && "out of bounds");
                linear_index *= shape[i];
                linear_index += index[i];
            }

            return linear_index;
        }
    }

    template<size_t N>
    size_t linear_index(const std::vector<size_t>& shape, const std::array<size_t, N>& index) {
        return linear_index(shape, index.data(), index.size());
    }

    inline size_t linear_index(const std::vector<size_t>& shape, const std::vector<size_t>& index) {
        return linear_index(shape, index.data(), index.size());
    }
}


/// Simple N-dimensional array interface
///
/// This class can either be a non-owning view inside some existing memory (for
/// example memory allocated by Rust); or own its memory (in the form of an
/// `std::vector<double>`). If the array does not own its memory, accessing it
/// is only valid for as long as the memory is kept alive.
///
/// The API of this class is very intentionally minimal to keep equistore as
/// simple as possible. Feel free to wrap the corresponding data inside types
/// with richer API such as Eigen, Boost, etc.
template<typename T>
class NDArray {
public:
    /// Create a new empty `NDArray`, with shape `[0, 0]`.
    NDArray(): NDArray(nullptr, {0, 0}, true) {}

    /// Create a new `NDArray` using a non-owning view in `const` memory with
    /// the given `shape`.
    ///
    /// `data` must point to contiguous memory containing the right number of
    /// elements as described by the `shape`, which will be interpreted as an
    /// N-dimensional array in row-major order. The resulting `NDArray` is
    /// only valid for as long as `data` is.
    NDArray(const T* data, std::vector<size_t> shape):
        NDArray(data, std::move(shape), /*is_const*/ true) {}

    /// Create a new `NDArray` using a non-owning view in non-`const` memory
    /// with the given `shape`.
    ///
    /// `data` must point to contiguous memory containing the right number of
    /// elements as described by the `shape`, which will be interpreted as an
    /// N-dimensional array in row-major order. The resulting `NDArray` is
    /// only valid for as long as `data` is.
    NDArray(T* data, std::vector<size_t> shape):
        NDArray(data, std::move(shape), /*is_const*/ false) {}

    /// Create a new `NDArray` *owning* its `data` with the given `shape`.
    NDArray(std::vector<T> data, std::vector<size_t> shape):
        NDArray(data.data(), std::move(shape), /*is_const*/ false)
    {
        using vector_t = std::vector<T>;

        owned_data_ = reinterpret_cast<void*>(new vector_t(std::move(data)));
        deleter_ = [](void* data){
            auto data_vector = reinterpret_cast<vector_t*>(data);
            delete data_vector;
        };
    }

    ~NDArray() {
        deleter_(this->owned_data_);
    }

    /// NDArray is not copy-constructible
    NDArray(const NDArray&) = delete;
    /// NDArray can not be copy-assigned
    NDArray& operator=(const NDArray& other) = delete;

    /// NDArray is move-constructible
    NDArray(NDArray&& other) noexcept: NDArray() {
        *this = std::move(other);
    }

    /// NDArray can be move-assigned
    NDArray& operator=(NDArray&& other) noexcept {
        this->deleter_(this->owned_data_);

        this->data_ = std::move(other.data_);
        this->shape_ = std::move(other.shape_);
        this->is_const_ = std::move(other.is_const_);
        this->owned_data_ = std::move(other.owned_data_);
        this->deleter_ = std::move(other.deleter_);

        other.data_ = nullptr;
        other.owned_data_ = nullptr;
        other.deleter_ = [](void*){};

        return *this;
    }

    /// Is this NDArray a view into external data?
    bool is_view() const {
        return owned_data_ == nullptr;
    }

    /// Get the value inside this `NDArray` at the given index
    ///
    /// ```
    /// auto array = NDArray(...);
    ///
    /// double value = array(2, 3, 1);
    /// ```
    template<typename ...Args>
    T operator()(Args... args) const {
        auto index = std::array<size_t, sizeof... (Args)>{static_cast<size_t>(args)...};
        if (index.size() != shape_.size()) {
            throw Error(
                "expected " + std::to_string(shape_.size()) +
                " indexes in Labels::operator(), got " + std::to_string(index.size())
            );
        }
        return data_[details::linear_index(shape_, index)];
    }

    /// Get a reference to the value inside this `NDArray` at the given index
    ///
    /// ```
    /// auto array = NDArray(...);
    ///
    /// array(2, 3, 1) = 5.2;
    /// ```
    template<typename ...Args>
    T& operator()(Args... args) {
        if (is_const_) {
            throw Error("This NDArray is const, can not get non const access to it");
        }

        auto index = std::array<size_t, sizeof... (Args)>{static_cast<size_t>(args)...};
        if (index.size() != shape_.size()) {
            throw Error(
                "expected " + std::to_string(shape_.size()) +
                " indexes in Labels::operator(), got " + std::to_string(index.size())
            );
        }
        return data_[details::linear_index(shape_, index)];
    }

    /// Get the data pointer for this array, i.e. the pointer to the first
    /// element.
    const T* data() const {
        return data_;
    }

    /// Get the data pointer for this array, i.e. the pointer to the first
    /// element.
    T* data() {
        if (is_const_) {
            throw Error("This NDArray is const, can not get non const access to it");
        }
        return data_;
    }

    /// Get the shape of this array
    const std::vector<size_t>& shape() const {
        return shape_;
    }

    /// Check if this array is empty, i.e. if at least one of the shape element
    /// is 0.
    bool is_empty() const {
        for (auto s: shape_) {
            if (s == 0) {
                return true;
            }
        }
        return false;
    }

private:
    /// Create an `NDArray` from a pointer to the (row-major) data & shape.
    ///
    /// The `is_const` parameter controls whether this class should allow
    /// non-const access to the data.
    NDArray(const T* data, std::vector<size_t> shape, bool is_const):
        data_(const_cast<T*>(data)),
        shape_(std::move(shape)),
        is_const_(is_const),
        owned_data_(nullptr),
        deleter_([](void*){})
    {
        validate();
    }

    /// Create a 2D NDArray from a vector of initializer lists. All the inner
    /// lists must have the same `size`.
    ///
    /// This allows creating an array (and in particular a set of Labels) with
    /// `NDArray({{1, 2}, {3, 4}, {5, 6}}, 2)`
    NDArray(std::vector<std::initializer_list<T>> data, size_t size):
        data_(nullptr),
        shape_({data.size(), size}),
        is_const_(false),
        deleter_([](void*){})
    {
        using vector_t = std::vector<T>;
        auto vector = std::vector<T>();
        vector.reserve(data.size() * size);
        for (auto row: std::move(data)) {
            if (row.size() != size) {
                throw Error(
                    "invalid size for row: expected " + std::to_string(size) +
                    " got " + std::to_string(row.size())
                );
            }

            for (auto entry: row) {
                vector.emplace_back(entry);
            }
        }

        data_ = vector.data();
        owned_data_ = reinterpret_cast<void*>(new vector_t(std::move(vector)));
        deleter_ = [](void* data){
            auto data_vector = reinterpret_cast<vector_t*>(data);
            delete data_vector;
        };
        validate();
    }

    friend class Labels;

    void validate() const {
        static_assert(
            std::is_arithmetic<T>::value,
            "NDArray only works with integers and floating points"
        );

        if (shape_.size() == 0) {
            throw Error("invalid parameters to NDArray, shape should contain at least one element");
        }

        size_t size = 1;
        for (auto s: shape_) {
            size *= s;
        }

        if (size != 0 && data_ == nullptr) {
            throw Error("invalid parameters to NDArray, got null data pointer and non zero size");
        }
    }

    /// Pointer to the data used by this array
    T* data_ = nullptr;
    /// Full shape of this array
    std::vector<size_t> shape_ = {0, 0};
    /// Is this array const? This will dynamically prevent calling non-const
    /// function on it.
    bool is_const_ = true;
    /// Type-erased owned data for this array. This is a `nullptr` if this array
    /// is a view.
    void* owned_data_ = nullptr;
    /// Custom delete function for the `owned_data_`.
    std::function<void(void*)> deleter_;
};


/// Compare this `NDArray` with another `NDarray`. The array are equal if
/// and only if both the shape and data are equal.
template<typename T>
bool operator==(const NDArray<T>& lhs, const NDArray<T>& rhs) {
    if (lhs.shape() != rhs.shape()) {
        return false;
    }
    return std::memcmp(lhs.data(), rhs.data(), details::product(lhs.shape())) == 0;
}

/// Compare this `NDArray` with another `NDarray`. The array are equal if
/// and only if both the shape and data are equal.
template<typename T>
bool operator!=(const NDArray<T>& lhs, const NDArray<T>& rhs) {
    return !(lhs == rhs);
}


/// A set of labels used to carry metadata associated with a tensor map.
///
/// This is similar to an array of named tuples, but stored as a 2D array
/// of shape `(count, size)`, with a set of names associated with the columns of
/// this array (often called *variables*). Each row/entry in this array is
/// unique, and they are often (but not always) sorted in lexicographic order.
class Labels final: public NDArray<int32_t> {
public:
    /// Create a new set of Labels from the given `names` and `values`.
    ///
    /// Each entry in the values must contain `names.size()` elements.
    ///
    /// ```
    /// auto labels = Labels({"first", "second"}, {
    ///    {0, 1},
    ///    {1, 4},
    ///    {2, 1},
    ///    {2, 3},
    /// });
    /// ```
    Labels(const std::vector<std::string>& names, std::vector<std::initializer_list<int32_t>> values):
        NDArray(std::move(values), names.size()),
        names_()
    {
        std::memset(&labels_, 0, sizeof(labels_));

        assert(this->shape().size() == 2);

        set_names_from_cxx(names);
    }


    /// Create an empty set of Labels with the given names
    Labels(const std::vector<std::string>& names):
        NDArray(static_cast<const int32_t*>(nullptr), {0, names.size()}),
        names_()
    {
        std::memset(&labels_, 0, sizeof(labels_));

        assert(this->shape().size() == 2);
        set_names_from_cxx(names);
    }

    ~Labels() {
        for (auto variable: this->names_) {
            free(const_cast<char*>(variable));
        }
    }

    /// Labels is not copy-constructible
    Labels(const Labels&) = delete;
    /// Labels can not be copy-assigned
    Labels& operator=(const Labels& other) = delete;

    /// Labels is move-constructible
    Labels(Labels&& other) noexcept: Labels({}, {{}}) {
        *this = std::move(other);
    }

    /// Labels can be move-assigned
    Labels& operator=(Labels&& other) noexcept {
        NDArray<int32_t>::operator=(std::move(other));

        for (auto variable: this->names_) {
            free(const_cast<char*>(variable));
        }

        this->names_ = std::move(other.names_);
        return *this;
    }

    /// Get the names of the variables used in these `Labels`.
    const std::vector<const char*>& names() const {
        return names_;
    }

    /// Get the number of entries in this set of Labels.
    ///
    /// This is the same as `shape()[0]` for the corresponding values array
    size_t count() const {
        return this->shape()[0];
    }

    /// Get the number of variables in this set of Labels.
    ///
    /// This is the same as `shape()[1]` for the corresponding values array
    size_t size() const {
        return this->names_.size();
    }

    /// Create the C++ `Labels` from a C `eqs_labels_t`.
    ///
    /// The resulting Labels are only valid for as long as the data in
    /// `labels.values` is.
    static Labels unsafe_from_eqs_labels(eqs_labels_t labels) {
        return Labels(labels);
    }

    /// Convert from this set of Labels to the C `eqs_labels_t`
    eqs_labels_t as_eqs_labels_t() const {
        if (labels_.labels_ptr != nullptr) {
            return labels_;
        } else {
            eqs_labels_t labels;
            std::memset(&labels, 0, sizeof(labels));

            labels.names = names_.data();
            labels.size = names_.size();
            labels.count = this->shape()[0];
            labels.values = this->data();
            return labels;
        }
    }

    /// Get the position of the entry defined by the `label` array in these
    /// Labels.
    ///
    /// sThis operation is only available if the labels correspond to a set of
    /// Rust Labels (i.e. `labels.labels_ptr` is not NULL).
    int64_t position(std::initializer_list<int32_t> label) const {
        int64_t result = 0;
        details::check_status(eqs_labels_position(labels_, label.begin(), label.size(), &result));
        return result;
    }

    /// Variant of `Labels::position` taking a fixed-size array as input
    template<size_t N>
    int64_t position(std::array<int32_t, N> label) const {
        int64_t result = 0;
        details::check_status(eqs_labels_position(labels_, label.data(), label.size(), &result));
        return result;
    }

private:
    Labels(eqs_labels_t labels):
        NDArray(labels.values, {labels.count, labels.size}),
        labels_(labels)
    {
        for (size_t i=0; i<labels.size; i++) {
            auto size = std::strlen(labels.names[i]) + 1;
            auto new_name = static_cast<char*>(std::calloc(1, size));
            std::strncpy(new_name, labels.names[i], size);
            names_.push_back(new_name);
        }
    }

    void set_names_from_cxx(const std::vector<std::string>& names) {
        if (names.size() != this->shape()[1]) {
            throw Error("expected as many names as there are columns in the values");
        }

        names_.resize(names.size());
        for (size_t i=0; i<names.size(); i++) {
            // allocate 1 extra char to NULL-terminate the string
            auto size = names[i].size() + 1;
            auto new_name = static_cast<char*>(std::calloc(1, size));
            std::strncpy(new_name, names[i].data(), size);
            names_[i] = new_name;
        }
    }

    std::vector<const char*> names_;
    eqs_labels_t labels_;

    friend bool operator==(const Labels& lhs, const Labels& rhs);
};


/// Two Labels compare equal only if they have the same names and values in the
/// same order.
inline bool operator==(const Labels& lhs, const Labels& rhs) {
    if (lhs.names_.size() != rhs.names_.size()) {
        return false;
    }

    for (size_t i=0; i<lhs.names_.size(); i++) {
        if (std::strcmp(lhs.names_[i], rhs.names_[i]) != 0) {
            return false;
        }
    }

    return static_cast<const NDArray<int32_t>&>(lhs) == static_cast<const NDArray<int32_t>&>(rhs);
}

/// Two Labels compare equal only if they have the same names and values in the
/// same order.
inline bool operator!=(const Labels& lhs, const Labels& rhs) {
    return !(lhs == rhs);
}

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/


/// `DataArrayBase` manages n-dimensional arrays used as data in a block or
/// tensor map. The array itself if opaque to this library and can come from
/// multiple sources: Rust program, a C/C++ program, a Fortran program, Python
/// with numpy or torch. The data does not have to live on CPU, or even on the
/// same machine where this code is executed.
///
/// **WARNING**: all function implementations **MUST** be thread-safe, and can
/// be called from multiple threads at the same time. The `DataArrayBase` itself
/// might be moved from one thread to another.
class DataArrayBase {
public:
    DataArrayBase() {}
    virtual ~DataArrayBase() = default;

    /// DataArrayBase can be copy-constructed
    DataArrayBase(const DataArrayBase&) = default;
    /// DataArrayBase can be copy-assigned
    DataArrayBase& operator=(const DataArrayBase&) = default;
    /// DataArrayBase can be move-constructed
    DataArrayBase(DataArrayBase&&) = default;
    /// DataArrayBase can be move-assigned
    DataArrayBase& operator=(DataArrayBase&&) = default;

    /// Convert a concrete `DataArrayBase` to a C-compatible `eqs_array_t`
    static eqs_array_t to_eqs_array_t(std::unique_ptr<DataArrayBase> data) {
        eqs_array_t array;
        std::memset(&array, 0, sizeof(array));

        array.ptr = data.release();

        array.destroy = [](void* array) {
            auto ptr = std::unique_ptr<DataArrayBase>(static_cast<DataArrayBase*>(array));
            // let ptr go out of scope
        };

        array.origin = [](const void* array, eqs_data_origin_t* origin) {
            try {
                auto cxx_array = static_cast<const DataArrayBase*>(array);
                *origin = cxx_array->origin();
                return EQS_SUCCESS;
            } catch (const std::exception&) {
                return -1;
            } catch (...) {
                return -128;
            }
        };

        array.copy = [](const void* array, eqs_array_t* new_array) {
            try {
                auto cxx_array = static_cast<const DataArrayBase*>(array);
                auto copy = cxx_array->copy();
                *new_array = DataArrayBase::to_eqs_array_t(std::move(copy));
                return EQS_SUCCESS;
            } catch (const std::exception&) {
                return -1;
            } catch (...) {
                return -128;
            }
        };

        array.create = [](const void* array, const uintptr_t* shape, uintptr_t shape_count, eqs_array_t* new_array) {
            try {
                auto cxx_array = static_cast<const DataArrayBase*>(array);
                auto cxx_shape = std::vector<size_t>();
                for (size_t i=0; i<static_cast<size_t>(shape_count); i++) {
                    cxx_shape.push_back(static_cast<size_t>(shape[i]));
                }
                auto copy = cxx_array->create(std::move(cxx_shape));
                *new_array = DataArrayBase::to_eqs_array_t(std::move(copy));
                return EQS_SUCCESS;
            } catch (const std::exception&) {
                return -1;
            } catch (...) {
                return -128;
            }
        };


        array.data = [](const void* array, const double** data) {
            try {
                auto cxx_array = static_cast<const DataArrayBase*>(array);
                *data = cxx_array->data();
                return EQS_SUCCESS;
            } catch (const std::exception&) {
                return -1;
            } catch (...) {
                return -128;
            }
        };

        array.shape = [](const void* array, const uintptr_t** shape, uintptr_t* shape_count) {
            try {
                auto cxx_array = static_cast<const DataArrayBase*>(array);
                const auto& cxx_shape = cxx_array->shape();
                *shape = cxx_shape.data();
                *shape_count = static_cast<uintptr_t>(cxx_shape.size());
                return EQS_SUCCESS;
            } catch (const std::exception&) {
                return -1;
            } catch (...) {
                return -128;
            }
        };

        array.reshape = [](void* array, const uintptr_t* shape, uintptr_t shape_count) {
            try {
                auto cxx_array = static_cast<DataArrayBase*>(array);
                auto cxx_shape = std::vector<uintptr_t>(shape, shape + shape_count);
                cxx_array->reshape(std::move(cxx_shape));
                return EQS_SUCCESS;
            } catch (const std::exception&) {
                return -1;
            } catch (...) {
                return -128;
            }
        };
        array.swap_axes = [](void* array, uintptr_t axis_1, uintptr_t axis_2) {
            try {
                auto cxx_array = static_cast<DataArrayBase*>(array);
                cxx_array->swap_axes(axis_1, axis_2);
                return EQS_SUCCESS;
            } catch (const std::exception&) {
                return -1;
            } catch (...) {
                return -128;
            }
        };


        array.move_samples_from = [](
            void* array,
            const void* input,
            const eqs_sample_mapping_t* samples,
            uintptr_t samples_count,
            uintptr_t property_start,
            uintptr_t property_end
        ) {
            try {
                auto cxx_array = static_cast<DataArrayBase*>(array);
                auto cxx_input = static_cast<const DataArrayBase*>(input);
                auto cxx_samples = std::vector<eqs_sample_mapping_t>(samples, samples + samples_count);

                cxx_array->move_samples_from(*cxx_input, cxx_samples, property_start, property_end);
                return EQS_SUCCESS;
            } catch (const std::exception&) {
                return -1;
            } catch (...) {
                return -128;
            }
        };

        return array;
    }

    /// Get "data origin" for this array in.
    ///
    /// Users of `DataArrayBase` should register a single data
    /// origin with `eqs_register_data_origin`, and use it for all compatible
    /// arrays.
    virtual eqs_data_origin_t origin() const = 0;


    /// Make a copy of this DataArrayBase and return the new array. The new
    /// array is expected to have the same data origin and parameters (data
    /// type, data location, etc.)
    virtual std::unique_ptr<DataArrayBase> copy() const = 0;

    /// Create a new array with the same options as the current one (data type,
    /// data location, etc.) and the requested `shape`.
    ///
    /// The new array should be filled with zeros.
    virtual std::unique_ptr<DataArrayBase> create(std::vector<uintptr_t> shape) const = 0;


    /// Get a pointer to the underlying data storage.
    ///
    /// This function is allowed to fail if the data is not accessible in RAM,
    /// not stored as 64-bit floating point values, or not stored as a
    /// C-contiguous array.
    virtual const double* data() const = 0;

    /// Get the shape of this array
    virtual const std::vector<uintptr_t>& shape() const = 0;

    /// Set the shape of this array to the given `shape`
    virtual void reshape(std::vector<uintptr_t> shape) = 0;

    /// Swap the axes `axis_1` and `axis_2` in this `array`.
    virtual void swap_axes(uintptr_t axis_1, uintptr_t axis_2) = 0;


    /// Set entries in the current array taking data from the `input` array.
    ///
    /// This array is guaranteed to be created by calling `eqs_array_t::create`
    /// with one of the arrays in the same block or tensor map as the `input`.
    ///
    /// The `samples` indicate where the data should be moved from `input` to
    /// the current DataArrayBase.
    ///
    /// This function should copy data from `input[samples[i].input, ..., :]` to
    /// `array[samples[i].output, ..., property_start:property_end]` for `i` up
    /// to `samples_count`. All indexes are 0-based.
    virtual void move_samples_from(
        const DataArrayBase& input,
        std::vector<eqs_sample_mapping_t> samples,
        uintptr_t property_start,
        uintptr_t property_end
    ) = 0;

};


/// Very basic implementation of DataArrayBase in C++.
///
/// This is included as an example implementation of DataArrayBase, and to make
/// equistore usable without additional dependencies. For other uses cases, it
/// might be better to implement DataArrayBase on your data, using
/// functionalities from `Eigen`, `Boost.Array`, etc.
class SimpleDataArray: public equistore::DataArrayBase {
public:
    /// Create a SimpleDataArray with the given `shape`, and all elements set to
    /// `value`
    SimpleDataArray(std::vector<uintptr_t> shape, double value = 0.0):
        shape_(std::move(shape)), data_(details::product(shape_), value) {}

    /// Create a SimpleDataArray with the given `shape` and `data`.
    ///
    /// The data is interpreted as a row-major n-dimensional array.
    SimpleDataArray(std::vector<uintptr_t> shape, std::vector<double> data):
        shape_(std::move(shape)),
        data_(std::move(data))
    {
        if (data_.size() != details::product(shape_)) {
            throw Error("the shape and size of the data don't match in SimpleDataArray");
        }
    }

    ~SimpleDataArray() override = default;

    /// SimpleDataArray can be copy-constructed
    SimpleDataArray(const SimpleDataArray&) = default;
    /// SimpleDataArray can be copy-assigned
    SimpleDataArray& operator=(const SimpleDataArray&) = default;
    /// SimpleDataArray can be move-constructed
    SimpleDataArray(SimpleDataArray&&) = default;
    /// SimpleDataArray can be move-assigned
    SimpleDataArray& operator=(SimpleDataArray&&) = default;

    eqs_data_origin_t origin() const override {
        eqs_data_origin_t origin = 0;
        eqs_register_data_origin("equistore::SimpleDataArray", &origin);
        return origin;
    }

    const double* data() const override {
        return data_.data();
    }

    const std::vector<uintptr_t>& shape() const override {
        return shape_;
    }

    void reshape(std::vector<uintptr_t> shape) override {
        if (details::product(shape_) != details::product(shape)) {
            throw equistore::Error("invalid shape in reshape");
        }
        shape_ = std::move(shape);
    }

    void swap_axes(uintptr_t axis_1, uintptr_t axis_2) override {
        auto new_data = std::vector<double>(details::product(shape_), 0.0);
        auto new_shape = shape_;
        std::swap(new_shape[axis_1], new_shape[axis_2]);

        for (size_t i=0; i<details::product(shape_); i++) {
            auto index = details::cartesian_index(shape_, i);
            std::swap(index[axis_1], index[axis_2]);

            new_data[details::linear_index(new_shape, index)] = data_[i];
        }

        shape_ = std::move(new_shape);
        data_ = std::move(new_data);
    }

    std::unique_ptr<DataArrayBase> copy() const override {
        return std::unique_ptr<DataArrayBase>(new SimpleDataArray(*this));
    }

    std::unique_ptr<DataArrayBase> create(std::vector<uintptr_t> shape) const override {
        return std::unique_ptr<DataArrayBase>(new SimpleDataArray(std::move(shape)));
    }

    void move_samples_from(
        const DataArrayBase& input,
        std::vector<eqs_sample_mapping_t> samples,
        uintptr_t property_start,
        uintptr_t property_end
    ) override {
        const auto& input_array = dynamic_cast<const SimpleDataArray&>(input);
        assert(input_array.shape_.size() == this->shape_.size());

        size_t property_count = property_end - property_start;
        size_t property_dim = shape_.size() - 1;
        assert(input_array.shape_[property_dim] == property_count);

        auto input_index = std::vector<size_t>(shape_.size(), 0);
        auto output_index = std::vector<size_t>(shape_.size(), 0);

        for (const auto& sample: samples) {
            input_index[0] = sample.input;
            output_index[0] = sample.output;

            if (property_dim == 1) {
                // no components
                for (size_t property_i=0; property_i<property_count; property_i++) {
                    input_index[property_dim] = property_i;
                    output_index[property_dim] = property_i + property_start;

                    auto value = input_array.data_[details::linear_index(input_array.shape_, input_index)];
                    this->data_[details::linear_index(shape_, output_index)] = value;
                }
            } else {
                auto last_component_dim = shape_.size() - 2;
                for (size_t component_i=1; component_i<shape_.size() - 1; component_i++) {
                    input_index[component_i] = 0;
                }

                bool done = false;
                while (!done) {
                    for (size_t component_i=1; component_i<shape_.size() - 1; component_i++) {
                        output_index[component_i] = input_index[component_i];
                    }

                    for (size_t property_i=0; property_i<property_count; property_i++) {
                        input_index[property_dim] = property_i;
                        output_index[property_dim] = property_i + property_start;

                        auto value = input_array.data_[details::linear_index(input_array.shape_, input_index)];
                        this->data_[details::linear_index(shape_, output_index)] = value;
                    }

                    input_index[last_component_dim] += 1;
                    for (size_t component_i=last_component_dim; component_i>2; component_i--) {
                        if (input_index[component_i] >= shape_[component_i]) {
                            input_index[component_i] = 0;
                            input_index[component_i - 1] += 1;
                        }
                    }

                    if (input_index[1] >= shape_[1]) {
                        done = true;
                    }
                }
            }
        }
    }

    /// Get a const view of the data managed by this SimpleDataArray
    NDArray<double> view() const {
        return NDArray<double>(data_.data(), shape_);
    }

    /// Get a mutable view of the data managed by this SimpleDataArray
    NDArray<double> view() {
        return NDArray<double>(data_.data(), shape_);
    }

    /// Extract a reference to SimpleDataArray out of an `eqs_array_t`.
    ///
    /// This function fails if the `eqs_array_t` does not contain a
    /// SimpleDataArray.
    static SimpleDataArray& from_eqs_array(eqs_array_t& array) {
        eqs_data_origin_t origin = 0;
        auto status = array.origin(array.ptr, &origin);
        if (status != EQS_SUCCESS) {
            throw Error("failed to get data origin");
        }

        char buffer[64] = {0};
        status = eqs_get_data_origin(origin, buffer, 64);
        if (status != EQS_SUCCESS || std::string(buffer) != "equistore::SimpleDataArray") {
            throw Error("this array is not an equistore::SimpleDataArray");
        }

        auto* base = static_cast<DataArrayBase*>(array.ptr);
        return dynamic_cast<SimpleDataArray&>(*base);
    }

    /// Extract a const reference to SimpleDataArray out of an `eqs_array_t`.
    ///
    /// This function fails if the `eqs_array_t` does not contain a
    /// SimpleDataArray.
    static const SimpleDataArray& from_eqs_array(const eqs_array_t& array) {
        eqs_data_origin_t origin = 0;
        auto status = array.origin(array.ptr, &origin);
        if (status != EQS_SUCCESS) {
            throw Error("failed to get data origin");
        }

        char buffer[64] = {0};
        status = eqs_get_data_origin(origin, buffer, 64);
        if (status != EQS_SUCCESS || std::string(buffer) != "equistore::SimpleDataArray") {
            throw Error("this array is not an equistore::SimpleDataArray");
        }

        const auto* base = static_cast<const DataArrayBase*>(array.ptr);
        return dynamic_cast<const SimpleDataArray&>(*base);
    }

private:
    std::vector<uintptr_t> shape_;
    std::vector<double> data_;

    friend bool operator==(const SimpleDataArray& lhs, const SimpleDataArray& rhs);
};

/// Two SimpleDataArray compare as equal if they have the exact same shape and
/// data.
inline bool operator==(const SimpleDataArray& lhs, const SimpleDataArray& rhs) {
    return lhs.shape_ == rhs.shape_ && lhs.data_ == rhs.data_;
}

/// Two SimpleDataArray compare as equal if they have the exact same shape and
/// data.
inline bool operator!=(const SimpleDataArray& lhs, const SimpleDataArray& rhs) {
    return !(lhs == rhs);
}


/******************************************************************************/
/******************************************************************************/
/******************************************************************************/


/// This is a proxy class allowing to access the information associated with a
/// gradient inside a `TensorBlock`.
///
/// Tha data accessible through this proxy is only alive for as long as the
/// corresponding `TensorBlock` is.
class GradientProxy {
public:
    ~GradientProxy() = default;

    /// GradientProxy can be copy-constructed
    GradientProxy(const GradientProxy&) = default;
    /// GradientProxy can be copy-assigned
    GradientProxy& operator=(const GradientProxy&) = default;
    /// GradientProxy can be move-constructed
    GradientProxy(GradientProxy&&) = default;
    /// GradientProxy can be move-assigned
    GradientProxy& operator=(GradientProxy&&) = default;

    /// Get a const view of the data for this gradient
    NDArray<double> data() const {
        auto array = this->const_eqs_array();
        const double* data = nullptr;
        details::check_status(array.data(array.ptr, &data));

        return NDArray<double>(data, this->data_shape());
    }

    /// Get the `eqs_array_t` corresponding to the data of this gradient
    eqs_array_t eqs_array() {
        eqs_array_t array;
        std::memset(&array, 0, sizeof(array));

        details::check_status(
            eqs_block_data(block_, parameter_.c_str(), &array)
        );
        return array;
    }

    /// Access the sample `Labels` for this gradient.
    ///
    /// The entries in these labels describe the first dimension of the `data()`
    /// array.
    Labels samples() const {
        return this->labels(0);
    }

    /// Access the component `Labels` for this gradient.
    ///
    /// The entries in these labels describe intermediate dimensions of the
    /// `data()` array.
    std::vector<Labels> components() const {
        auto shape = this->data_shape();

        auto result = std::vector<Labels>();
        for (size_t i=1; i<shape.size() - 1; i++) {
            result.emplace_back(this->labels(i));
        }

        return result;
    }

    /// Access the property `Labels` for this gradient.
    ///
    /// The entries in these labels describe the last dimension of the `data()`
    /// array. The properties are guaranteed to be the same for values and
    /// gradients in the same block.
    Labels properties() const {
        auto shape = this->data_shape();
        return this->labels(shape.size() - 1);
    }


private:
    /// Get the labels for the given axis
    Labels labels(uintptr_t axis) const {
        eqs_labels_t labels;
        std::memset(&labels, 0, sizeof(labels));
        details::check_status(eqs_block_labels(
            block_, parameter_.c_str(), axis, &labels
        ));

        return Labels::unsafe_from_eqs_labels(labels);
    }

    /// extract the shape of the data for these gradients
    std::vector<uintptr_t> data_shape() const {
        auto array = this->const_eqs_array();

        const uintptr_t* shape = nullptr;
        uintptr_t shape_count = 0;
        details::check_status(array.shape(array.ptr, &shape, &shape_count));
        assert(shape_count >= 2);

        return {shape, shape + shape_count};
    }

    /// Get the `eqs_array_t` corresponding to the data of this gradient. The
    /// returned `eqs_array_t` should only be used in a const context
    eqs_array_t const_eqs_array() const {
        eqs_array_t array;
        std::memset(&array, 0, sizeof(array));

        details::check_status(
            eqs_block_data(block_, parameter_.c_str(), &array)
        );
        return array;
    }

    /// Create a gradient proxy for the given parameter and block
    GradientProxy(eqs_block_t* block, std::string parameter):
        block_(block), parameter_(std::move(parameter)) {}

    friend class TensorBlock;

    eqs_block_t* block_;
    std::string parameter_;
};


/// Basic building block for a tensor map.
///
/// A single block contains a n-dimensional `eqs_array_t` (or `DataArrayBase`),
/// and n sets of `Labels` (one for each dimension). The first dimension is the
/// *samples* dimension, the last dimension is the *properties* dimension. Any
/// intermediate dimension is called a *component* dimension.
///
/// Samples should be used to describe *what* we are representing, while
/// properties should contain information about *how* we are representing it.
/// Finally, components should be used to describe vectorial or tensorial
/// components of the data.
///
/// A block can also contain gradients of the values with respect to a variety
/// of parameters. In this case, each gradient has a separate set of samples,
/// and possibly components but share the same property labels as the values.
class TensorBlock final {
public:
    /// Create a new TensorBlock containing the given `values` array.
    ///
    /// The different dimensions of the values are described by `samples`,
    /// `components` and `properties` `Labels`
    TensorBlock(
        std::unique_ptr<DataArrayBase> values,
        const Labels& samples,
        const std::vector<Labels>& components,
        const Labels& properties
    ):
        block_(nullptr),
        is_view_(false)
    {
        auto c_components = std::vector<eqs_labels_t>();
        for (const auto& component: components) {
            c_components.push_back(component.as_eqs_labels_t());
        }
        block_ = eqs_block(
            DataArrayBase::to_eqs_array_t(std::move(values)),
            samples.as_eqs_labels_t(),
            c_components.data(),
            c_components.size(),
            properties.as_eqs_labels_t()
        );

        details::check_pointer(block_);
    }

    ~TensorBlock() {
        if (!is_view_) {
            eqs_block_free(block_);
        }
    }

    /// TensorBlock can be copy constructed
    TensorBlock(const TensorBlock& other): TensorBlock() {
        *this = other;
    }

    /// TensorBlock can be copy assigned
    TensorBlock& operator=(const TensorBlock& other) {
        if (!is_view_) {
            eqs_block_free(block_);
        }

        block_ = eqs_block_copy(other.block_);
        is_view_ = false;
        details::check_pointer(block_);

        return *this;
    }

    /// TensorBlock can be move constructed
    TensorBlock(TensorBlock&& other): TensorBlock() {
        *this = std::move(other);
    }

    /// TensorBlock can be moved assigned
    TensorBlock& operator=(TensorBlock&& other) {
        if (!is_view_) {
            eqs_block_free(block_);
        }

        this->block_ = other.block_;
        this->is_view_ = other.is_view_;
        other.block_ = nullptr;
        other.is_view_ = true;

        return *this;
    }

    /// Get a const view in the values in this block
    NDArray<double> values() const {
        auto array = this->const_eqs_array("values");
        const double* data = nullptr;
        details::check_status(array.data(array.ptr, &data));

        return NDArray<double>(data, this->values_shape());
    }

    /// Access the sample `Labels` for this block.
    ///
    /// The entries in these labels describe the first dimension of the
    /// `values()` array.
    Labels samples() const {
        return this->labels("values", 0);
    }

    /// Access the component `Labels` for this block.
    ///
    /// The entries in these labels describe intermediate dimensions of the
    /// `values()` array.
    std::vector<Labels> components() const {
        auto shape = this->values_shape();

        auto result = std::vector<Labels>();
        for (size_t i=1; i<shape.size() - 1; i++) {
            result.emplace_back(this->labels("values", i));
        }

        return result;
    }

    /// Access the property `Labels` for this block.
    ///
    /// The entries in these labels describe the last dimension of the
    /// `values()` array. The properties are guaranteed to be the same for
    /// values and gradients in the same block.
    Labels properties() const {
        auto shape = this->values_shape();
        return this->labels("values", shape.size() - 1);
    }

    /// Add a set of gradients with respect to `parameters` in this block.
    ///
    /// @param parameter add gradients with respect to this `parameter` (e.g.
    ///                 `"positions"`, `"cell"`, ...)
    /// @param data the gradient array, of shape `(gradient_samples, components,
    ///             properties)`, where the properties labels are the same as
    ///             the values' properties labels.
    /// @param samples labels describing the gradient samples
    /// @param components labels describing the gradient components
    void add_gradient(
        const std::string& parameter,
        std::unique_ptr<DataArrayBase> data,
        const Labels& samples,
        const std::vector<Labels>& components
    ) {
        if (is_view_) {
            throw Error(
                "can not call TensorBlock::add_gradient on this block since "
                "it is a view inside a TensorMap"
            );
        }

        auto c_components = std::vector<eqs_labels_t>();
        for (const auto& component: components) {
            c_components.push_back(component.as_eqs_labels_t());
        }
        details::check_status(eqs_block_add_gradient(
            block_,
            parameter.c_str(),
            DataArrayBase::to_eqs_array_t(std::move(data)),
            samples.as_eqs_labels_t(),
            c_components.data(),
            c_components.size()
        ));
    }

    /// Get a list of all gradients defined in this block.
    std::vector<std::string> gradients_list() const {
        const char*const * parameters = nullptr;
        uintptr_t count = 0;
        details::check_status(eqs_block_gradients_list(
            block_,
            &parameters,
            &count
        ));

        auto result = std::vector<std::string>();
        for (uint64_t i=0; i<count; i++) {
            result.push_back(std::string(parameters[i]));
        }

        return result;
    }

    /// Get the gradient of the `values()` in this block with respect to
    /// the given `parameter`.
    ///
    /// @param parameter check for gradients with respect to this `parameter`
    ///                  (e.g. `"positions"`, `"cell"`, ...)
    GradientProxy gradient(std::string parameter) const {
        return GradientProxy(block_, std::move(parameter));
    }

    /// Get the `eqs_block_t` pointer corresponding to this block.
    ///
    /// The block pointer is still managed by the current `TensorBlock`
    eqs_block_t* as_eqs_block_t() {
        if (is_view_) {
            throw Error(
                "can not call non-const TensorBlock::as_eqs_block_t on this "
                "block since it is a view inside a TensorMap"
            );
        }
        return block_;
    }

    /// const version of `as_eqs_block_t`
    const eqs_block_t* as_eqs_block_t() const {
        return block_;
    }

    /// Release the `eqs_block_t` pointer corresponding to this `TensorBlock`.
    ///
    /// The block pointer is **no longer** managed by the current `TensorBlock`,
    /// and should manually be freed when no longer required.
    eqs_block_t* release() {
         if (is_view_) {
            throw Error(
                "can not call non-const TensorBlock::release on this "
                "block since it is a view inside a TensorMap"
            );
        }
        auto ptr = block_;
        block_ = nullptr;
        is_view_ = false;
        return ptr;
    }

    /// Create a new TensorBlock taking ownership of a raw `eqs_block_t` pointer.
    static TensorBlock unsafe_from_ptr(eqs_block_t* ptr) {
        auto block = TensorBlock();
        block.block_ = ptr;
        block.is_view_ = false;
        return block;
    }

    /// Create a new TensorBlock which is a view corresponding to a raw
    /// `eqs_block_t` pointer.
    static TensorBlock unsafe_from_const_ptr(const eqs_block_t* ptr) {
        auto block = TensorBlock();
        // this const_cast is fine since we dynamically check on `is_view_`
        // before calling any non-const function
        block.block_ = const_cast<eqs_block_t*>(ptr);
        block.is_view_ = true;
        return block;
    }

    /// Get a raw `eqs_array_t` corresponding to either the values or one of the
    /// gradients in this block.
    ///
    /// `values_gradients` should be `"values"` to get the values, or the
    /// gradient parameter to get a gradient.
    eqs_array_t eqs_array(const char* values_gradients) {
        eqs_array_t array;
        std::memset(&array, 0, sizeof(array));

        details::check_status(
            eqs_block_data(block_, values_gradients, &array)
        );
        return array;
    }

    /// Get the labels in this block associated with either `"values"` or one
    /// gradient (by setting `values_gradients` to the gradient parameter); in
    /// the given `axis`.
    Labels labels(const char* values_gradients, uintptr_t axis) const {
        eqs_labels_t labels;
        std::memset(&labels, 0, sizeof(labels));
        details::check_status(eqs_block_labels(
            block_, values_gradients, axis, &labels
        ));

        return Labels::unsafe_from_eqs_labels(labels);
    }

private:
    /// Constructor of a TensorBlock not associated with anything
    TensorBlock(): block_(nullptr), is_view_(true) {}

    /// Get the shape of the value array for this block
    std::vector<uintptr_t> values_shape() const {
        auto array = this->const_eqs_array("values");

        const uintptr_t* shape = nullptr;
        uintptr_t shape_count = 0;
        details::check_status(array.shape(array.ptr, &shape, &shape_count));
        assert(shape_count >= 2);

        return {shape, shape + shape_count};
    }

    /// Get one of the `eqs_array_t` for this block, either the `"values"` or
    /// one of the gradients
    ///
    /// The returned `eqs_array_t` should only be used in a const context
    eqs_array_t const_eqs_array(const char* values_gradients) const {
        eqs_array_t array;
        std::memset(&array, 0, sizeof(array));

        details::check_status(
            eqs_block_data(block_, values_gradients, &array)
        );
        return array;
    }

    friend class TensorMap;

    eqs_block_t* block_;
    bool is_view_;
};


/******************************************************************************/
/******************************************************************************/
/******************************************************************************/


/// A TensorMap is the main user-facing class of this library, and can store any
/// kind of data used in atomistic machine learning.
///
/// A tensor map contains a list of `TensorBlock`, each one associated with a
/// key. Users can access the blocks either one by one with the `block_by_id()`
/// function.
///
/// A tensor map provides functions to move some of these keys to the samples or
/// properties labels of the blocks, moving from a sparse representation of the
/// data to a dense one.
class TensorMap final {
public:
    /// Create a new TensorMap with the given `keys` and `blocks`
    TensorMap(Labels keys, std::vector<TensorBlock> blocks) {
        auto c_blocks = std::vector<eqs_block_t*>();
        for (auto& block: blocks) {
            c_blocks.push_back(block.as_eqs_block_t());
            // We will move the data inside the new map, let's put the
            // TensorBLock in its moved-from state now
            block.block_ = nullptr;
        }

        tensor_ = eqs_tensormap(
            keys.as_eqs_labels_t(),
            c_blocks.data(),
            c_blocks.size()
        );

        details::check_pointer(tensor_);
    }

    ~TensorMap() {
        eqs_tensormap_free(tensor_);
    }

    /// TensorMap can not be copy constructed
    TensorMap(const TensorMap&) = delete;
    /// TensorMap can not be copy assigned
    TensorMap& operator=(const TensorMap&) = delete;

    /// TensorMap can be move constructed
    TensorMap(TensorMap&& other): TensorMap(nullptr) {
        *this = std::move(other);
    }

    /// TensorMap can be move assigned
    TensorMap& operator=(TensorMap&& other) {
        details::check_status(eqs_tensormap_free(tensor_));

        this->tensor_ = other.tensor_;
        other.tensor_ = nullptr;

        return *this;
    }

    /// Get the set of keys labeling the blocks in this tensor map
    Labels keys() const {
        eqs_labels_t keys;
        details::check_status(eqs_tensormap_keys(tensor_, &keys));
        return Labels::unsafe_from_eqs_labels(keys);
    }

    /// Get a (possibly empty) list of block indexes matching the `selection`
    std::vector<uintptr_t> blocks_matching(const Labels& selection) const {
        auto matching = std::vector<uintptr_t>(this->keys().count());
        uintptr_t count = matching.size();

        details::check_status(eqs_tensormap_blocks_matching(
            tensor_,
            matching.data(),
            &count,
            selection.as_eqs_labels_t()
        ));

        assert(count <= matching.size());
        matching.resize(count);
        return matching;
    }

    /// Get a block inside this TensorMap by it's index/the index of the
    /// corresponding key.
    ///
    /// The returned `TensorBlock` is a view inside memory owned by this
    /// `TensorMap`, and is only valid as long as the `TensorMap` is kept alive.
    TensorBlock block_by_id(uintptr_t index) const {
        const eqs_block_t* block = nullptr;
        details::check_status(eqs_tensormap_block_by_id(tensor_, &block, index));
        details::check_pointer(block);

        return TensorBlock::unsafe_from_const_ptr(block);
    }

    /// Merge blocks with the same value for selected keys variables along the
    /// property axis.
    ///
    /// The variables (names) of `keys_to_move` will be moved from the keys to
    /// the property labels, and blocks with the same remaining keys variables
    /// will be merged together along the property axis.
    ///
    /// If `keys_to_move` does not contains any entries (i.e.
    /// `keys_to_move.count() == 0`), then the new property labels will contain
    /// entries corresponding to the merged blocks only. For example, merging a
    /// block with key `a=0` and properties `p=1, 2` with a block with key `a=2`
    /// and properties `p=1, 3` will produce a block with properties `a, p = (0,
    /// 1), (0, 2), (2, 1), (2, 3)`.
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
    /// @param keys_to_move description of the keys to move
    /// @param sort_samples whether to sort the merged samples or keep them in
    ///                     the order in which they appear in the original blocks
    void keys_to_properties(const Labels& keys_to_move, bool sort_samples = true) {
        details::check_status(eqs_tensormap_keys_to_properties(
            tensor_,
            keys_to_move.as_eqs_labels_t(),
            sort_samples
        ));
    }

    /// This function calls `keys_to_properties` with an empty set of `Labels`
    /// with the variables defined in `keys_to_move`
    void keys_to_properties(const std::vector<std::string>& keys_to_move, bool sort_samples = true) {
        keys_to_properties(Labels(keys_to_move), sort_samples);
    }

    /// This function calls `keys_to_properties` with an empty set of `Labels`
    /// with a single variable: `key_to_move`
    void keys_to_properties(const std::string& key_to_move, bool sort_samples = true) {
        keys_to_properties(std::vector<std::string>{key_to_move}, sort_samples);
    }

    /// Merge blocks with the same value for selected keys variables along the
    /// samples axis.
    ///
    /// The variables (names) of `keys_to_move` will be moved from the keys to
    /// the sample labels, and blocks with the same remaining keys variables
    /// will be merged together along the sample axis.
    ///
    /// If `keys_to_move` must be an empty set of `Labels`
    /// (`keys_to_move.count() == 0`). The new sample labels will contain
    /// entries corresponding to the merged blocks' keys.
    ///
    /// The order of the samples is controlled by `sort_samples`. If
    /// `sort_samples` is true, samples are re-ordered to keep them
    /// lexicographically sorted. Otherwise they are kept in the order in which
    /// they appear in the blocks.
    ///
    /// This function is only implemented if all merged block have the same
    /// property labels.
    ///
    /// @param keys_to_move description of the keys to move
    /// @param sort_samples whether to sort the merged samples or keep them in
    ///                     the order in which they appear in the original blocks
    void keys_to_samples(const Labels& keys_to_move, bool sort_samples = true) {
        details::check_status(eqs_tensormap_keys_to_samples(
            tensor_,
            keys_to_move.as_eqs_labels_t(),
            sort_samples
        ));
    }

    /// This function calls `keys_to_properties` with an empty set of `Labels`
    /// with the variables defined in `keys_to_move`
    void keys_to_samples(const std::vector<std::string>& keys_to_move, bool sort_samples = true) {
        keys_to_samples(Labels(keys_to_move), sort_samples);
    }

    /// This function calls `keys_to_properties` with an empty set of `Labels`
    /// with a single variable: `key_to_move`
    void keys_to_samples(const std::string& key_to_move, bool sort_samples = true) {
        keys_to_samples(std::vector<std::string>{key_to_move}, sort_samples);
    }

    /// Move the given `variables` from the component labels to the property
    /// labels for each block.
    ///
    /// @param variables name of the component variables to move to the
    ///                  properties
    void components_to_properties(const std::vector<std::string>& variables) {
        auto c_variables = std::vector<const char*>();
        for (const auto& v: variables) {
            c_variables.push_back(v.c_str());
        }

        details::check_status(eqs_tensormap_components_to_properties(
            tensor_,
            c_variables.data(),
            c_variables.size()
        ));
    }

    /// Call `components_to_properties` with a single variable
    void components_to_properties(const std::string& variable) {
        const char* c_str = variable.c_str();
        details::check_status(eqs_tensormap_components_to_properties(
            tensor_,
            &c_str,
            1
        ));
    }

    /// Load a previously saved `TensorMap` from the given path.
    ///
    /// `TensorMap` are serialized using numpy's `.npz` format, i.e. a ZIP
    /// file without compression (storage method is `STORED`), where each file
    /// is stored as a `.npy` array. See the C API documentation for more
    /// information on the format.
    static TensorMap load(const std::string& path) {
        auto ptr = eqs_tensormap_load(path.c_str());
        details::check_pointer(ptr);
        return TensorMap(ptr);
    }

    /// "Save the given `TensorMap` to a file at `path`.
    ///
    /// `TensorMap` are serialized using numpy's `.npz` format, i.e. a ZIP
    /// file without compression (storage method is `STORED`), where each file
    /// is stored as a `.npy` array. See the C API documentation for more
    /// information on the format.
    static void save(const std::string& path, const TensorMap& tensor) {
        details::check_status(eqs_tensormap_save(path.c_str(), tensor.tensor_));
    }

    /// Get the `eqs_tensormap_t` pointer corresponding to this `TensorMap`.
    ///
    /// The tensor map pointer is still managed by the current `TensorMap`
    eqs_tensormap_t* as_eqs_tensormap_t() {
        return tensor_;
    }

    /// Release the `eqs_tensormap_t` pointer corresponding to this `TensorMap`.
    ///
    /// The tensor map pointer is **no longer** managed by the current
    /// `TensorMap`, and should manually be freed when no longer required.
    eqs_tensormap_t* release() {
        auto ptr = tensor_;
        tensor_ = nullptr;
        return ptr;
    }

    /// Create a C++ TensorMap from a C `eqs_tensormap_t` pointer. The C++
    /// tensor map takes ownership of the C pointer.
    TensorMap(eqs_tensormap_t* tensor): tensor_(tensor) {}

private:
    eqs_tensormap_t* tensor_;
};


}

#endif /* EQUISTORE_HPP */
