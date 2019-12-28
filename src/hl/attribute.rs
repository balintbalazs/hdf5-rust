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
pub mod tests {
    use std::fs;
    use std::io::Read;

    use hdf5_sys::{h5d::H5Dwrite, h5s::H5S_ALL};

    use crate::internal_prelude::*;

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
    pub fn test_filters() {
        with_tmp_file(|file| {
            assert_eq!(
                file.new_dataset::<u32>().create_anon(100).unwrap().filters(),
                Filters::default()
            );
            assert_eq!(
                file.new_dataset::<u32>()
                    .shuffle(true)
                    .create_anon(100)
                    .unwrap()
                    .filters()
                    .get_shuffle(),
                true
            );
            assert_eq!(
                file.new_dataset::<u32>()
                    .fletcher32(true)
                    .create_anon(100)
                    .unwrap()
                    .filters()
                    .get_fletcher32(),
                true
            );
            assert_eq!(
                file.new_dataset::<u32>()
                    .scale_offset(8)
                    .create_anon(100)
                    .unwrap()
                    .filters()
                    .get_scale_offset(),
                Some(8)
            );
        });

        with_tmp_file(|file| {
            let filters = Filters::new().fletcher32(true).shuffle(true).clone();
            assert_eq!(
                file.new_dataset::<u32>().filters(&filters).create_anon(100).unwrap().filters(),
                filters
            );
        })
    }

    #[test]
    pub fn test_resizable() {
        with_tmp_file(|file| {
            assert_eq!(file.new_dataset::<u32>().create_anon(1).unwrap().is_resizable(), false);
            assert_eq!(
                file.new_dataset::<u32>().resizable(false).create_anon(1).unwrap().is_resizable(),
                false
            );
            assert_eq!(
                file.new_dataset::<u32>().resizable(true).create_anon(1).unwrap().is_resizable(),
                true
            );
        })
    }

    #[test]
    pub fn test_track_times() {
        with_tmp_file(|file| {
            assert_eq!(file.new_dataset::<u32>().create_anon(1).unwrap().tracks_times(), false);
            assert_eq!(
                file.new_dataset::<u32>().track_times(false).create_anon(1).unwrap().tracks_times(),
                false
            );
            assert_eq!(
                file.new_dataset::<u32>().track_times(true).create_anon(1).unwrap().tracks_times(),
                true
            );
        });

        with_tmp_path(|path| {
            let mut buf1: Vec<u8> = Vec::new();
            File::create(&path).unwrap().new_dataset::<u32>().create("foo", 1).unwrap();
            fs::File::open(&path).unwrap().read_to_end(&mut buf1).unwrap();

            let mut buf2: Vec<u8> = Vec::new();
            File::create(&path)
                .unwrap()
                .new_dataset::<u32>()
                .track_times(false)
                .create("foo", 1)
                .unwrap();
            fs::File::open(&path).unwrap().read_to_end(&mut buf2).unwrap();

            assert_eq!(buf1, buf2);

            let mut buf2: Vec<u8> = Vec::new();
            File::create(&path)
                .unwrap()
                .new_dataset::<u32>()
                .track_times(true)
                .create("foo", 1)
                .unwrap();
            fs::File::open(&path).unwrap().read_to_end(&mut buf2).unwrap();
            assert_ne!(buf1, buf2);
        });
    }

    #[test]
    pub fn test_storage_size_offset() {
        with_tmp_file(|file| {
            let ds = file.new_dataset::<u16>().create_anon(3).unwrap();
            assert_eq!(ds.storage_size(), 0);
            assert!(ds.offset().is_none());

            let buf: Vec<u16> = vec![1, 2, 3];
            h5call!(H5Dwrite(
                ds.id(),
                Datatype::from_type::<u16>().unwrap().id(),
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                buf.as_ptr() as *const _
            ))
            .unwrap();
            assert_eq!(ds.storage_size(), 6);
            assert!(ds.offset().is_some());
        })
    }

    #[test]
    pub fn test_datatype() {
        with_tmp_file(|file| {
            assert_eq!(
                file.new_dataset::<f32>().create_anon(1).unwrap().dtype().unwrap(),
                Datatype::from_type::<f32>().unwrap()
            );
        })
    }

    #[test]
    pub fn test_create_anon() {
        with_tmp_file(|file| {
            let ds = file.new_dataset::<u32>().create("foo/bar", (1, 2)).unwrap();
            assert!(ds.is_valid());
            assert_eq!(ds.shape(), vec![1, 2]);
            assert_eq!(ds.name(), "/foo/bar");
            assert_eq!(file.group("foo").unwrap().dataset("bar").unwrap().shape(), vec![1, 2]);

            let ds = file.new_dataset::<u32>().create_anon((2, 3)).unwrap();
            assert!(ds.is_valid());
            assert_eq!(ds.name(), "");
            assert_eq!(ds.shape(), vec![2, 3]);
        })
    }

    #[test]
    pub fn test_fill_value() {
        with_tmp_file(|file| {
            macro_rules! check_fill_value {
                ($ds:expr, $tp:ty, $v:expr) => {
                    assert_eq!(($ds).fill_value::<$tp>().unwrap(), Some(($v) as $tp));
                };
            }

            macro_rules! check_fill_value_approx {
                ($ds:expr, $tp:ty, $v:expr) => {{
                    let fill_value = ($ds).fill_value::<$tp>().unwrap().unwrap();
                    // FIXME: should inexact float->float casts be prohibited?
                    assert!((fill_value - (($v) as $tp)).abs() < (1.0e-6 as $tp));
                }};
            }

            macro_rules! check_all_fill_values {
                ($ds:expr, $v:expr) => {
                    check_fill_value!($ds, u8, $v);
                    check_fill_value!($ds, u16, $v);
                    check_fill_value!($ds, u32, $v);
                    check_fill_value!($ds, u64, $v);
                    check_fill_value!($ds, i8, $v);
                    check_fill_value!($ds, i16, $v);
                    check_fill_value!($ds, i32, $v);
                    check_fill_value!($ds, i64, $v);
                    check_fill_value!($ds, usize, $v);
                    check_fill_value!($ds, isize, $v);
                    check_fill_value_approx!($ds, f32, $v);
                    check_fill_value_approx!($ds, f64, $v);
                };
            }

            let ds = file.new_dataset::<u16>().create_anon(100).unwrap();
            check_all_fill_values!(ds, 0);

            let ds = file.new_dataset::<u16>().fill_value(42).create_anon(100).unwrap();
            check_all_fill_values!(ds, 42);

            let ds = file.new_dataset::<f32>().fill_value(1.234).create_anon(100).unwrap();
            check_all_fill_values!(ds, 1.234);
        })
    }
}
