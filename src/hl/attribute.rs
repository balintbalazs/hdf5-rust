use std::fmt::{self, Debug};
use std::ops::Deref;

use hdf5_sys::{
    h5a::{ H5Acreate2, 
    },
};

use crate::internal_prelude::*;

/// Represents the HDF5 attribute object.
#[repr(transparent)]
#[derive(Clone)]
pub struct Attribute(Handle);

impl ObjectClass for Attribute {
    const NAME: &'static str = "attribute";
    const VALID_TYPES: &'static [H5I_type_t] = &[H5I_ATTR];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }

    // TODO: short_repr()
}

impl Debug for Attribute {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.debug_fmt(f)
    }
}

impl Deref for Attribute {
    type Target = Container;

    fn deref(&self) -> &Container {
        unsafe { self.transmute() }
    }
}

impl Attribute {

}

#[derive(Clone)]
pub struct AttributeBuilder<T> {
    packed: bool,
    filters: Filters,
    parent: Result<Handle>,
    track_times: bool,
    phantom: std::marker::PhantomData<T>,
}

impl<T: H5Type> AttributeBuilder<T> {
    /// Create a new dataset builder and bind it to the parent container.
    pub fn new(parent: &Group) -> Self {
        h5lock!({
            // Store the reference to the parent handle and try to increase its reference count.
            let handle = Handle::try_new(parent.id());
            if let Ok(ref handle) = handle {
                handle.incref();
            }

            Self {
                packed: false,
                filters: Filters::default(),
                parent: handle,
                track_times: false,
                phantom: std::marker::PhantomData,
            }
        })
    }

    /// Create a new dataset builder and bind it to the parent container.
    pub fn new_from_dataset(parent: &Dataset) -> Self {
        h5lock!({
            // Store the reference to the parent handle and try to increase its reference count.
            let handle = Handle::try_new(parent.id());
            if let Ok(ref handle) = handle {
                handle.incref();
            }

            Self {
                packed: false,
                filters: Filters::default(),
                parent: handle,
                track_times: false,
                phantom: std::marker::PhantomData,
            }
        })
    }

    pub fn packed(&mut self, packed: bool) -> &mut Self {
        self.packed = packed;
        self
    }

    /// Enable or disable tracking object modification time (disabled by default).
    pub fn track_times(&mut self, track_times: bool) -> &mut Self {
        self.track_times = track_times;
        self
    }

    fn finalize<S: Into<Extents>>(&self, name: &str, extents: S) -> Result<Attribute> {
        let type_descriptor = if self.packed {
            <T as H5Type>::type_descriptor().to_packed_repr()
        } else {
            <T as H5Type>::type_descriptor().to_c_repr()
        };
        let extents = extents.into();

        h5lock!({
            let datatype = Datatype::from_descriptor(&type_descriptor)?;
            let parent = try_ref_clone!(self.parent);

            let dataspace = Dataspace::try_new(extents)?;

            let name = to_cstring(name)?;
            Attribute::from_id(h5try!(H5Acreate2(
                parent.id(),
                name.as_ptr(),
                datatype.id(),
                dataspace.id(),
                H5P_DEFAULT,
                H5P_DEFAULT,
            )))
        })
    }

    /// Create the dataset and link it into the file structure.
    pub fn create<S: Into<Extents>>(&self, name: &str, shape: S) -> Result<Attribute> {
        self.finalize(name, shape)
    }
}

#[cfg(test)]
pub mod attribute_tests {
    use crate::internal_prelude::*;
    use ndarray::{arr2, Array2};

    #[test]
    pub fn test_shape_ndim_size() {
        with_tmp_file(|file| {
            let d = file.new_attribute::<f32>().create("name1", (2, 3)).unwrap();
            assert_eq!(d.shape(), vec![2, 3]);
            assert_eq!(d.size(), 6);
            assert_eq!(d.ndim(), 2);
            assert_eq!(d.is_scalar(), false);

            let d = file.new_attribute::<u8>().create("name2", ()).unwrap();
            assert_eq!(d.shape(), vec![]);
            assert_eq!(d.size(), 1);
            assert_eq!(d.ndim(), 0);
            assert_eq!(d.is_scalar(), true);
        })
    }

    #[test]
    pub fn test_datatype() {
        with_tmp_file(|file| {
            assert_eq!(
                file.new_attribute::<f32>().create("name", 1).unwrap().dtype().unwrap(),
                Datatype::from_type::<f32>().unwrap()
            );
        })
    }

    #[test]
    pub fn test_read_write() {
        with_tmp_file(|file| {

            let arr = arr2(&[[1, 2, 3], [4, 5, 6]]);

            let attr = file.new_attribute::<f32>().create("foo", (2, 3)).unwrap();
            attr.as_writer().write(&arr).unwrap();

            let read_attr = file.attribute("foo").unwrap();
            assert_eq!(read_attr.shape(), vec![2, 3]);

            let arr_dyn: Array2<_> = read_attr.as_reader().read().unwrap();

            assert_eq!(arr, arr_dyn.into_dimensionality().unwrap());
        })
    }

    #[test]
    pub fn test_create() {
        with_tmp_file(|file| {
            let attr = file.new_attribute::<u32>().create("foo", (1, 2)).unwrap();
            assert!(attr.is_valid());
            assert_eq!(attr.shape(), vec![1, 2]);
            // FIXME - attr.name() returns "/" here, which is the name the attribute is connected to,
            // not the name of the attribute.
            //assert_eq!(attr.name(), "foo");
            assert_eq!(file.attribute("foo").unwrap().shape(), vec![1, 2]);
        })
    }
}
