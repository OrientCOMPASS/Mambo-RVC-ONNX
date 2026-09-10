use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct AudioRingBuffer {
    buffer: UnsafeCell<Vec<f32>>,
    capacity: usize,
    write_pos: AtomicUsize,
    read_pos: AtomicUsize,
}

unsafe impl Send for AudioRingBuffer {}
unsafe impl Sync for AudioRingBuffer {}

impl AudioRingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: UnsafeCell::new(vec![0.0; capacity]),
            capacity,
            write_pos: AtomicUsize::new(0),
            read_pos: AtomicUsize::new(0),
        }
    }

    pub fn push(&self, data: &[f32]) -> Result<usize, ()> {
        let write_pos = self.write_pos.load(Ordering::Acquire);
        let read_pos = self.read_pos.load(Ordering::Acquire);

        let available = if write_pos >= read_pos {
            self.capacity - (write_pos - read_pos) - 1
        } else {
            read_pos - write_pos - 1
        };

        if data.len() > available {
            return Err(());
        }

        let buffer = unsafe { &mut *self.buffer.get() };
        let end_pos = write_pos + data.len();

        if end_pos <= self.capacity {
            buffer[write_pos..end_pos].copy_from_slice(data);
        } else {
            let first_part = self.capacity - write_pos;
            buffer[write_pos..self.capacity].copy_from_slice(&data[..first_part]);
            buffer[0..(data.len() - first_part)].copy_from_slice(&data[first_part..]);
        }

        self.write_pos.store((write_pos + data.len()) % self.capacity, Ordering::Release);
        Ok(data.len())
    }

    pub fn pop(&self, out: &mut [f32]) -> usize {
        let write_pos = self.write_pos.load(Ordering::Acquire);
        let read_pos = self.read_pos.load(Ordering::Acquire);

        let available = if write_pos >= read_pos {
            write_pos - read_pos
        } else {
            self.capacity - read_pos + write_pos
        };

        let to_read = out.len().min(available);
        if to_read == 0 {
            return 0;
        }

        let buffer = unsafe { &*self.buffer.get() };
        let end_pos = read_pos + to_read;

        if end_pos <= self.capacity {
            out[..to_read].copy_from_slice(&buffer[read_pos..end_pos]);
        } else {
            let first_part = self.capacity - read_pos;
            out[..first_part].copy_from_slice(&buffer[read_pos..self.capacity]);
            out[first_part..to_read].copy_from_slice(&buffer[0..(to_read - first_part)]);
        }

        self.read_pos.store((read_pos + to_read) % self.capacity, Ordering::Release);
        to_read
    }

    pub fn available(&self) -> usize {
        let write_pos = self.write_pos.load(Ordering::Acquire);
        let read_pos = self.read_pos.load(Ordering::Acquire);
        if write_pos >= read_pos {
            write_pos - read_pos
        } else {
            self.capacity - read_pos + write_pos
        }
    }
}