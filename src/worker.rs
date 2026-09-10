use anyhow::{bail, Result};
use ort::session::{Session, SessionOutputs};
use ort::value::{DynValue, Tensor};
use rustfft::{num_complex::Complex, FftPlanner};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use crossbeam_channel::Receiver;
use std::f32::consts::PI;

use crate::logger;
use crate::path_resolver;
use crate::ring_buffer::AudioRingBuffer;
use crate::{CHUNK_MS, EXTRA_MS, CROSSFADE_MS, F0_UP_KEY};

pub struct WorkerConfig {
    pub sample_rate: usize,
    pub hop: usize,
    pub speaker: i64,
    pub model_path: String,
    pub hubert_path: String,
    pub rmvpe_path: String,
}

pub fn worker_loop(
    cfg: WorkerConfig,
    input_rb: Arc<AudioRingBuffer>,
    output_rb: Arc<AudioRingBuffer>,
    signal_rx: Receiver<()>,
    is_running: Arc<AtomicBool>,
) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        worker_loop_inner(cfg, input_rb, output_rb, signal_rx, is_running);
    }));
    if let Err(e) = result {
        let msg = if let Some(s) = e.downcast_ref::<&str>() { s.to_string() } 
                  else if let Some(s) = e.downcast_ref::<String>() { s.clone() } 
                  else { "Unknown panic".to_string() };
        logger::log(&format!("[Worker] PANIC: {}", msg));
    }
}

fn worker_loop_inner(
    cfg: WorkerConfig,
    input_rb: Arc<AudioRingBuffer>,
    output_rb: Arc<AudioRingBuffer>,
    signal_rx: Receiver<()>,
    is_running: Arc<AtomicBool>,
) {
    logger::log("[Worker] Thread started (Dynamic OLA Stream)");

    let plugin_dir = path_resolver::init_runtime_environment();
    let resolve_model_path = |relative: &str| -> String {
        if let Some(ref dir) = plugin_dir { dir.join(relative).to_string_lossy().to_string() } 
        else { relative.to_string() }
    };

    let model_path = resolve_model_path(&cfg.model_path);
    let hubert_path = resolve_model_path(&cfg.hubert_path);
    let rmvpe_path = resolve_model_path(&cfg.rmvpe_path);

    logger::log("[Worker] Initializing ORT...");
    ort::init().with_name("rvc-worker").commit();
    let cuda_ep = ort::ep::CUDA::default().with_device_id(0).build();

    let load_session = |path: &str| -> Result<Session> {
        Session::builder()
            .map_err(|e| anyhow::anyhow!("builder: {}", e))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level1) 
            .map_err(|e| anyhow::anyhow!("opt: {}", e))?
            .with_execution_providers([cuda_ep.clone()])
            .map_err(|e| anyhow::anyhow!("EP: {}", e))?
            .commit_from_file(path)
            .map_err(|e| anyhow::anyhow!("load: {}", e))
    };

    let mut rvc_session = match load_session(&model_path) {
        Ok(s) => s, Err(e) => { logger::log(&format!("[Worker] FATAL: RVC: {}", e)); return; }
    };
    let mut hubert_session = match load_session(&hubert_path) {
        Ok(s) => s, Err(e) => { logger::log(&format!("[Worker] FATAL: HuBERT: {}", e)); return; }
    };
    let mut rmvpe_session = match load_session(&rmvpe_path) {
        Ok(s) => s, Err(e) => { logger::log(&format!("[Worker] FATAL: RMVPE: {}", e)); return; }
    };

    let mut history_buf = Vec::with_capacity(480000); 
    let mut frame_buf = vec![0.0f32; 480]; 
    
    let mut fade_in = vec![0.0f32; 0];
    let mut fade_out = vec![0.0f32; 0];
    let mut prev_tail = vec![0.0f32; 0];
    let mut is_first_chunk = true;
    let mut last_crossfade_samples = 0;

    logger::log("[Worker] Entering dynamic window loop");

    while is_running.load(Ordering::Relaxed) {
        if signal_rx.recv().is_err() { break; }

        while input_rb.available() >= 480 {
            let read_count = input_rb.pop(&mut frame_buf);
            if read_count < 480 { break; }
            history_buf.extend_from_slice(&frame_buf[..read_count]);
        }

        let chunk_ms = CHUNK_MS.load(Ordering::Relaxed).clamp(50, 1000);
        let extra_ms = EXTRA_MS.load(Ordering::Relaxed).clamp(50, 1000);
        let crossfade_ms = CROSSFADE_MS.load(Ordering::Relaxed).clamp(5, 100);

        let block_samples = (chunk_ms as f32 * cfg.sample_rate as f32 / 1000.0) as usize;
        let extra_samples = (extra_ms as f32 * cfg.sample_rate as f32 / 1000.0) as usize;
        let mut crossfade_samples = (crossfade_ms as f32 * cfg.sample_rate as f32 / 1000.0) as usize;
        
        crossfade_samples = crossfade_samples.min(block_samples / 2).min(extra_samples);

        if crossfade_samples != last_crossfade_samples {
            fade_in = vec![0.0f32; crossfade_samples];
            fade_out = vec![0.0f32; crossfade_samples];
            for i in 0..crossfade_samples {
                let val = 0.5 * (1.0 - (PI * i as f32 / crossfade_samples as f32).cos());
                fade_in[i] = val;
                fade_out[i] = 1.0 - val;
            }
            prev_tail = vec![0.0f32; crossfade_samples];
            is_first_chunk = true;
            last_crossfade_samples = crossfade_samples;
        }

        let window_samples = block_samples + 2 * extra_samples;
        let window_samples_16k = window_samples / 3;
        let window_frames = window_samples / cfg.hop;

        while history_buf.len() >= window_samples {
            let start_idx = history_buf.len() - window_samples;
            let window_48k = &history_buf[start_idx..];

            let mut window_16k = vec![0.0f32; window_samples_16k];
            for i in 0..window_samples_16k {
                window_16k[i] = (window_48k[i * 3] + window_48k[i * 3 + 1] + window_48k[i * 3 + 2]) / 3.0;
            }

            let rms = (window_16k.iter().map(|x| x * x).sum::<f32>() / window_16k.len() as f32).sqrt();
            if rms < 1e-4 {
                let mut silence_chunk = vec![0.0f32; block_samples];
                if !is_first_chunk && crossfade_samples > 0 {
                    for i in 0..crossfade_samples {
                        silence_chunk[i] = prev_tail[i] * fade_out[i];
                    }
                }
                prev_tail.fill(0.0);
                let _ = output_rb.push(&silence_chunk);
                is_first_chunk = false;
            } else {
                let phone_50fps = match extract_phone(&mut hubert_session, &window_16k) {
                    Ok(p) => p, Err(_) => continue,
                };
                
                let mut f0 = match extract_f0_rmvpe(&mut rmvpe_session, &window_16k, window_frames) {
                    Ok(f) => f, Err(_) => continue,
                };

                let key = F0_UP_KEY.load(Ordering::Relaxed);
                if key != 0 {
                    let factor = 2.0f32.powf(key as f32 / 12.0);
                    for f in f0.iter_mut() {
                        if *f > 0.0 { *f *= factor; }
                    }
                }
                
                let inputs = match build_inputs(window_frames, cfg.speaker, &f0, &phone_50fps) {
                    Ok(i) => i, Err(_) => continue,
                };
                
                let outputs = match rvc_session.run(inputs) {
                    Ok(o) => o, Err(_) => continue,
                };
                
                let audio = match extract_output_audio(&outputs) {
                    Ok(a) => a, Err(_) => continue,
                };
                
                let curr_head_start = extra_samples.saturating_sub(crossfade_samples);
                let curr_body_start = extra_samples;
                let curr_body_end = extra_samples + block_samples - crossfade_samples;
                let next_tail_start = extra_samples + block_samples - crossfade_samples;
                let next_tail_end = extra_samples + block_samples;

                if audio.len() < next_tail_end {
                    let _ = output_rb.push(&vec![0.0f32; block_samples]);
                } else {
                    let mut out_chunk = vec![0.0f32; block_samples];

                    if is_first_chunk || crossfade_samples == 0 {
                        out_chunk[..crossfade_samples].copy_from_slice(&audio[curr_head_start..curr_head_start + crossfade_samples]);
                        out_chunk[crossfade_samples..].copy_from_slice(&audio[curr_body_start..curr_body_end]);
                        is_first_chunk = false;
                    } else {
                        for i in 0..crossfade_samples {
                            out_chunk[i] = prev_tail[i] * fade_out[i] + audio[curr_head_start + i] * fade_in[i];
                        }
                        out_chunk[crossfade_samples..].copy_from_slice(&audio[curr_body_start..curr_body_end]);
                    }

                    prev_tail.copy_from_slice(&audio[next_tail_start..next_tail_end]);
                    let _ = output_rb.push(&out_chunk);
                }
            }

            history_buf.drain(..block_samples);
        }
    }
    logger::log("[Worker] Thread exiting");
}

fn normalize_audio(samples: &mut [f32]) {
    let max_amp = samples.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    if max_amp > 1e-6 {
        let gain = (0.9 / max_amp).min(10.0);
        for s in samples.iter_mut() { *s *= gain; }
    }
}

fn extract_phone(session: &mut Session, samples_16k: &[f32]) -> Result<Vec<f32>> {
    if samples_16k.is_empty() { return Ok(vec![]); }
    let mut padded_samples = samples_16k.to_vec();
    normalize_audio(&mut padded_samples);
    let target_len = padded_samples.len() as i64;
    let mut inputs: Vec<(String, DynValue)> = Vec::new();
    for input in session.inputs().iter() {
        let name = input.name().to_lowercase();
        if name.contains("feats") || name.contains("source") || name.contains("input_values") || name == "input" {
            inputs.push((input.name().to_string(), Tensor::from_array((vec![1, 1, target_len], padded_samples.clone()))?.into_dyn()));
        } else if name.contains("mask") || name.contains("length") {
            if name.contains("mask") { inputs.push((input.name().to_string(), Tensor::from_array((vec![1, target_len], vec![1i64; target_len as usize]))?.into_dyn())); }
            else { inputs.push((input.name().to_string(), Tensor::from_array((vec![1], vec![target_len]))?.into_dyn())); }
        }
    }
    let outputs = session.run(inputs)?;
    for (_name, value) in outputs.iter() {
        if let Ok(v) = value.try_extract_tensor::<f32>() {
            if v.1.len() % 768 == 0 && v.1.len() > 768 { return Ok(v.1.to_vec()); }
        }
    }
    bail!("failed to extract phone features");
}

fn extract_f0_rmvpe(session: &mut Session, samples_16k: &[f32], frames: usize) -> Result<Vec<f32>> {
    if samples_16k.is_empty() { return Ok(vec![0.0; frames]); }
    let audio = samples_16k.to_vec();
    let n_fft = 1024; let hop_length = 160; let win_length = 1024; let n_mels = 128; let sample_rate = 16000;
    let mel_fmin = 30.0f32; let mel_fmax = 8000.0f32; let clamp = 1e-5f32; let pad_amount = n_fft / 2;
    let mut padded_samples = Vec::with_capacity(audio.len() + 2 * pad_amount);
    for i in (1..=pad_amount).rev() { padded_samples.push(audio[i.min(audio.len() - 1)]); }
    padded_samples.extend_from_slice(&audio);
    for i in 1..=pad_amount { padded_samples.push(audio[audio.len().saturating_sub(2 + i)]); }
    let num_frames = if padded_samples.len() >= n_fft { (padded_samples.len() - n_fft) / hop_length + 1 } else { 1 };
    let window: Vec<f32> = (0..win_length).map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (win_length - 1) as f32).cos())).collect();
    let n_freqs = n_fft / 2 + 1;
    let fft_freqs: Vec<f32> = (0..n_freqs).map(|i| (sample_rate as f32 / n_fft as f32) * i as f32).collect();
    let mel_min = 2595.0 * (1.0 + mel_fmin / 700.0).log10();
    let mel_max = 2595.0 * (1.0 + mel_fmax / 700.0).log10();
    let mel_points: Vec<f32> = (0..n_mels + 2).map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32).collect();
    let freq_points: Vec<f32> = mel_points.iter().map(|&m| 700.0 * (10.0f32.powf(m / 2595.0) - 1.0)).collect();
    let mut mel_basis = vec![0.0f32; n_mels * n_freqs];
    for mel_idx in 0..n_mels {
        let left = freq_points[mel_idx]; let center = freq_points[mel_idx + 1]; let right = freq_points[mel_idx + 2];
        for freq_idx in 0..n_freqs {
            let f = fft_freqs[freq_idx];
            if f >= left && f <= center && center > left { mel_basis[mel_idx * n_freqs + freq_idx] = (f - left) / (center - left); }
            else if f > center && f <= right && right > center { mel_basis[mel_idx * n_freqs + freq_idx] = (right - f) / (right - center); }
        }
    }
    for mel_idx in 0..n_mels {
        let enorm = 2.0 / (freq_points[mel_idx + 2] - freq_points[mel_idx]);
        for freq_idx in 0..n_freqs { mel_basis[mel_idx * n_freqs + freq_idx] *= enorm; }
    }
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut buffer = vec![Complex::new(0.0, 0.0); n_fft];
    let mut magnitude = vec![0.0f32; n_freqs * num_frames];
    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_length;
        for i in 0..n_fft { buffer[i] = Complex::new(padded_samples[start + i] * window[i], 0.0); }
        fft.process(&mut buffer);
        for i in 0..n_freqs { magnitude[i * num_frames + frame_idx] = buffer[i].norm(); }
    }
    let mut mel_output = vec![0.0f32; n_mels * num_frames];
    for frame_idx in 0..num_frames {
        for mel_idx in 0..n_mels {
            let mut sum = 0.0f32;
            for freq_idx in 0..n_freqs { sum += mel_basis[mel_idx * n_freqs + freq_idx] * magnitude[freq_idx * num_frames + frame_idx]; }
            mel_output[mel_idx * num_frames + frame_idx] = sum.max(clamp).ln();
        }
    }
    let pad_time = 32 * ((num_frames.saturating_sub(1)) / 32 + 1) - num_frames;
    let total_frames = num_frames + pad_time;
    let mut padded_mel = Vec::with_capacity(n_mels * total_frames);
    for mel_idx in 0..n_mels {
        let channel = &mel_output[mel_idx * num_frames..(mel_idx + 1) * num_frames];
        padded_mel.extend_from_slice(channel);
        for i in 0..pad_time { padded_mel.push(channel[num_frames.saturating_sub(2 + i)]); }
    }
    let mut inputs: Vec<(String, DynValue)> = Vec::new();
    for input in session.inputs().iter() {
        inputs.push((input.name().to_string(), Tensor::from_array((vec![1, 128, total_frames as i64], padded_mel.clone()))?.into_dyn()));
    }
    let outputs = session.run(inputs)?;
    let mut f0 = vec![0.0f32; frames];
    let mut cents_mapping = vec![0.0f32; 368];
    for i in 0..360 { cents_mapping[i + 4] = 20.0 * i as f32 + 1997.3794084376191; }
    for (_name, value) in outputs.iter() {
        if let Ok(v) = value.try_extract_tensor::<f32>() {
            let hidden_output = v.1;
            let max_f0_frames = frames.min(num_frames);
            for frame_idx in 0..max_f0_frames {
                let mut salience = vec![0.0f32; 360];
                for bin in 0..360 { salience[bin] = hidden_output[frame_idx * 360 + bin]; }
                let mut center = 0; let mut max_val = f32::NEG_INFINITY;
                for i in 0..360 { if salience[i] > max_val { max_val = salience[i]; center = i; } }
                if max_val <= 0.03 { f0[frame_idx] = 0.0; continue; }
                let mut padded_salience = vec![0.0f32; 368];
                padded_salience[4..364].copy_from_slice(&salience);
                let c = center + 4; let start = c.saturating_sub(4); let end = (c + 5).min(368);
                let mut product_sum = 0.0f32; let mut weight_sum = 0.0f32;
                for i in start..end { product_sum += padded_salience[i] * cents_mapping[i]; weight_sum += padded_salience[i]; }
                let divided = if weight_sum > 1e-8 { product_sum / weight_sum } else { 0.0 };
                let freq = 10.0 * (2.0f32.powf(divided / 1200.0));
                if (freq - 10.0).abs() < 1e-5 { f0[frame_idx] = 0.0; } else { f0[frame_idx] = freq; }
            }
            if max_f0_frames >= 3 {
                let mut filtered = f0[..max_f0_frames].to_vec();
                for i in 1..max_f0_frames - 1 {
                    let mut window = [f0[i - 1], f0[i], f0[i + 1]]; window.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    filtered[i] = window[1];
                }
                f0[..max_f0_frames].copy_from_slice(&filtered);
            }
            return Ok(f0);
        }
    }
    bail!("failed to extract f0 from RMVPE");
}

fn build_inputs(frames: usize, speaker: i64, f0: &[f32], phone_50fps: &[f32]) -> Result<Vec<(String, DynValue)>> {
    let mut inputs = Vec::new();
    let feat_dim = 768;
    let frames_50 = phone_50fps.len() / feat_dim;
    let mut phone_100fps = vec![0.0_f32; frames * feat_dim];
    for i in 0..frames {
        let t = if frames > 1 { i as f32 * (frames_50 - 1) as f32 / (frames - 1) as f32 } else { 0.0 };
        let t0 = t.floor() as usize; let t1 = (t0 + 1).min(frames_50.saturating_sub(1));
        let alpha = t - t0 as f32;
        for j in 0..feat_dim {
            phone_100fps[i * feat_dim + j] = (1.0 - alpha) * phone_50fps[t0 * feat_dim + j] + alpha * phone_50fps[t1 * feat_dim + j];
        }
    }
    inputs.push(("phone".to_string(), Tensor::from_array((vec![1, frames as i64, 768], phone_100fps))?.into_dyn()));
    inputs.push(("phone_lengths".to_string(), Tensor::from_array((vec![1], vec![frames as i64]))?.into_dyn()));
    let pitch_data: Vec<i64> = f0.iter().map(|&f| f0_coarse(f)).collect();
    inputs.push(("pitch".to_string(), Tensor::from_array((vec![1, frames as i64], pitch_data))?.into_dyn()));
    inputs.push(("nsff0".to_string(), Tensor::from_array((vec![1, frames as i64], f0.to_vec()))?.into_dyn()));
    inputs.push(("sid".to_string(), Tensor::from_array((vec![1], vec![speaker]))?.into_dyn()));
    Ok(inputs)
}

fn extract_output_audio(outputs: &SessionOutputs) -> Result<Vec<f32>> {
    let value = &outputs[0];
    if let Ok(v) = value.try_extract_tensor::<f32>() { return Ok(v.1.to_vec()); }
    bail!("no audio output");
}

fn f0_coarse(f: f32) -> i64 {
    if f <= 0.0 { return 0; }
    let mel = 1127.0_f32 * (1.0_f32 + f / 700.0_f32).ln();
    let min_mel = 1127.0_f32 * (1.0_f32 + 40.0_f32 / 700.0_f32).ln();
    let max_mel = 1127.0_f32 * (1.0_f32 + 1100.0_f32 / 700.0_f32).ln();
    let x = (mel - min_mel) / (max_mel - min_mel) * 255.0_f32;
    x.round().clamp(1.0_f32, 255.0_f32) as i64
}