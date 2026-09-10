use std::path::PathBuf;

#[cfg(target_os = "windows")]
mod imp {
    use std::ffi::OsString;
    use std::os::windows::ffi::{OsStringExt, OsStrExt};
    use std::path::PathBuf;

    type HMODULE = *mut std::ffi::c_void;
    type WCHAR = u16;
    type DWORD = u32;
    type BOOL = i32;

    extern "system" {
        fn GetModuleHandleExW(dwFlags: DWORD, lpModuleName: *const WCHAR, phModule: *mut HMODULE) -> BOOL;
        fn GetModuleFileNameW(hModule: HMODULE, lpFilename: *mut WCHAR, nSize: DWORD) -> DWORD;
        fn SetDllDirectoryW(lpPathName: *const WCHAR) -> BOOL;
    }

    pub fn get_current_module_dir() -> Option<PathBuf> {
        let mut module: HMODULE = std::ptr::null_mut();
        let flags: DWORD = 0x00000004 | 0x00000002;
        let address = (get_current_module_dir as usize) as *const WCHAR;

        unsafe {
            if GetModuleHandleExW(flags, address, &mut module) != 0 {
                let mut buffer = vec![0u16; 1024];
                let len = GetModuleFileNameW(module, buffer.as_mut_ptr(), buffer.len() as DWORD) as usize;
                if len > 0 {
                    let path = PathBuf::from(OsString::from_wide(&buffer[..len]));
                    return path.parent().map(|p| p.to_path_buf());
                }
            }
        }
        None
    }

    pub fn set_dll_directory(dir: &std::path::Path) -> bool {
        let wide: Vec<u16> = dir.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
        unsafe { SetDllDirectoryW(wide.as_ptr()) != 0 }
    }
}

#[cfg(not(target_os = "windows"))]
mod imp {
    use std::path::PathBuf;

    pub fn get_current_module_dir() -> Option<PathBuf> {
        std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf()))
    }

    pub fn set_dll_directory(_dir: &std::path::Path) -> bool {
        true
    }
}

pub fn get_plugin_dir() -> Option<PathBuf> {
    imp::get_current_module_dir()
}

pub fn init_runtime_environment() -> Option<PathBuf> {
    let plugin_dir = imp::get_current_module_dir()?;
    let libs_dir = plugin_dir.join("libs");

    imp::set_dll_directory(&libs_dir);

    let ort_lib_name = if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    };

    let ort_path = libs_dir.join(ort_lib_name);
    if ort_path.exists() {
        std::env::set_var("ORT_DYLIB_PATH", &ort_path);
    }

    Some(plugin_dir)
}