use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

static LOG_FILE: Mutex<Option<File>> = Mutex::new(None);

pub fn init(plugin_dir: &std::path::Path) {
    let log_path = plugin_dir.join("rvc_plugin.log");
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&log_path)
        .ok();
    
    if let Ok(mut guard) = LOG_FILE.lock() {
        *guard = file;
    }
}

pub fn log(msg: &str) {
    if let Ok(mut guard) = LOG_FILE.lock() {
        if let Some(ref mut file) = *guard {
            let _ = writeln!(file, "[{}] {}", chrono_now(), msg);
            let _ = file.flush();
        }
    }
}

fn chrono_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{}", secs)
}

pub fn get_plugin_dir() -> PathBuf {
    // 简单实现：使用当前工作目录
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}