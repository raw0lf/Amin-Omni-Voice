#!/usr/bin/env python3
"""
OmniVoice HuggingFace Space - Minimal app entrypoint.
Rewritten from scratch for clarity.
"""

import os
import re
import warnings
import shutil
import soundfile as sf
import json
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_settings(s):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(s, f)
    except:
        pass

def save_single_setting(key, val):
    s = load_settings()
    s[key] = val
    save_settings(s)

USER_SETTINGS = load_settings()

def browse_directory_host():
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    folder_path = filedialog.askdirectory()
    root.destroy()
    if folder_path:
        return folder_path
    import gradio as gr
    return gr.update()


import gradio as gr
import numpy as np
import torch

try:
    import spaces
except ImportError:
    class _GPU:
        def __init__(self, duration=60):
            pass

        def __call__(self, fn):
            return fn

    class _Spaces:
        GPU = _GPU

    spaces = _Spaces()

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
# build_demo import removed, UI has been fully inlined here for feature injection

LANGUAGES = [
    ("Auto", "Auto"),
    ("Arabic (ar)", "ar"),
    ("Chinese (zh)", "zh"),
    ("Dutch (nl)", "nl"),
    ("English (en)", "en"),
    ("French (fr)", "fr"),
    ("German (de)", "de"),
    ("Hindi (hi)", "hi"),
    ("Italian (it)", "it"),
    ("Japanese (ja)", "ja"),
    ("Korean (ko)", "ko"),
    ("Polish (pl)", "pl"),
    ("Portuguese (pt)", "pt"),
    ("Russian (ru)", "ru"),
    ("Spanish (es)", "es"),
    ("Turkish (tr)", "tr"),
]

TAG_CHOICES = [
    "[laughter]", "[sigh]", "[confirmation-en]", "[question-en]",
    "[question-ah]", "[question-oh]", "[question-ei]", "[question-yi]",
    "[surprise-ah]", "[surprise-oh]", "[surprise-wa]", "[surprise-yo]", "[dissatisfaction-hnn]",
]

_CATEGORIES = {
    "Gender / 性别": ["Male / 男", "Female / 女"],
    "Age / 年龄": [
        "Child / 儿童",
        "Teenager / 少年",
        "Young Adult / 青年",
        "Middle-aged / 中年",
        "Elderly / 老年",
    ],
    "Pitch / 音调": [
        "Very Low Pitch / 极低音调",
        "Low Pitch / 低音调",
        "Moderate Pitch / 中音调",
        "High Pitch / 高音调",
        "Very High Pitch / 极高音调",
    ],
    "Style / 风格": ["Whisper / 耳语"],
    "English Accent / 英文口音": [
        "American Accent / 美式口音",
        "Australian Accent / 澳大利亚口音",
        "British Accent / 英国口音",
        "Chinese Accent / 中国口音",
        "Canadian Accent / 加拿大口音",
        "Indian Accent / 印度口音",
        "Korean Accent / 韩国口音",
        "Portuguese Accent / 葡萄牙口音",
        "Russian Accent / 俄罗斯口音",
        "Japanese Accent / 日本口音",
    ],
    "Chinese Dialect / 中文方言": [
        "Henan Dialect / 河南话",
        "Shaanxi Dialect / 陕西话",
        "Sichuan Dialect / 四川话",
        "Guizhou Dialect / 贵州话",
        "Yunnan Dialect / 云南话",
        "Guilin Dialect / 桂林话",
        "Jinan Dialect / 济南话",
        "Shijiazhuang Dialect / 石家庄话",
        "Gansu Dialect / 甘肃话",
        "Ningxia Dialect / 宁夏话",
        "Qingdao Dialect / 青岛话",
        "Northeast Dialect / 东北话",
    ],
}

_ATTR_INFO = {
    "English Accent / 英文口音": "Only effective for English speech.",
    "Chinese Dialect / 中文方言": "Only effective for Chinese speech.",
}

_VALID_DEVICES = {"cuda", "mps", "cpu"}

def resolve_device(env_var=None):
    if env_var:
        if env_var not in _VALID_DEVICES:
            warnings.warn(
                f"OMNIVOICE_DEVICE={env_var!r} is not a valid device "
                f"(expected one of {_VALID_DEVICES}). Falling back to auto-detection.",
                stacklevel=2,
            )
        else:
            return env_var
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(device):
    return torch.float16 if device == "cuda" else torch.float32


def env_bool(name, default=True):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ["false", "0", "no", "off"]


def env_int(*names):
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and str(raw).strip() != "":
            return int(str(raw).strip())
    return None


# Model loading
ckpt = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
device = resolve_device(os.environ.get("OMNIVOICE_DEVICE"))
dtype = resolve_dtype(device)
load_asr = env_bool("OMNIVOICE_LOAD_ASR", True)

print(f"Loading {ckpt} on {device} ...")
model = OmniVoice.from_pretrained(ckpt, device_map=device, dtype=dtype, load_asr=load_asr)
sampling_rate = model.sampling_rate
print(f"Model ready. [Device: {device} | CUDA Available: {torch.cuda.is_available()}]")


def synthesize(text, language, ref_audio, instruct, num_step, guidance, denoise, speed, duration, preproc, postproc, mode, ref_text=None):
    if not text or not str(text).strip():
        return None, "Input text required."
    gen_conf = OmniVoiceGenerationConfig(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance or 2.0),
        denoise=bool(denoise),
        preprocess_prompt=bool(preproc),
        postprocess_output=bool(postproc),
    )
    lang = language if language and language != "Auto" else None
    args = dict(text=str(text).strip(), language=lang, generation_config=gen_conf)
    if speed and float(speed) != 1.0:
        args["speed"] = float(speed)
    if duration and float(duration) > 0:
        args["duration"] = float(duration)
    if mode == "clone":
        if not ref_audio:
            return None, "Reference audio required for cloning."
        args["voice_clone_prompt"] = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
    if mode == "design" and instruct and str(instruct).strip():
        args["instruct"] = instruct.strip()
    try:
        audio = model.generate(**args)
    except Exception as e:
        return None, f"Generation error: {type(e).__name__}: {e}"
    if not audio:
        return None, "Generation error: model returned no audio."
    arr = audio[0]
    # Robust squeeze: only squeeze if the first dimension is size 1 and it's not a 1D array
    if hasattr(arr, "ndim") and arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    elif hasattr(arr, "dim") and arr.dim() > 1 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
        
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu()
    
    # Final array conversion for numpy
    if hasattr(arr, "numpy"):
        wav_float = arr.numpy()
    else:
        wav_float = np.array(arr)
        
    wav = np.clip(wav_float, -1.0, 1.0)
    return (sampling_rate, (wav * 32767).astype(np.int16)), "Done."


def parse_dialogue(script):
    PATTERN = re.compile(r"^\s*\[speaker_(\d+)\]:\s*(.*)$", re.I)
    results = []
    cur_speaker = None
    cur_lines = []
    for line in str(script).strip().splitlines():
        m = PATTERN.match(line)
        if m:
            if cur_speaker is not None and cur_lines:
                results.append((cur_speaker, " ".join(cur_lines).strip()))
            cur_speaker = int(m[1])
            cur_lines = [m[2].strip()] if m[2].strip() else []
            continue
        line = line.strip()
        if line and cur_speaker is not None:
            cur_lines.append(line)
    if cur_speaker is not None and cur_lines:
        results.append((cur_speaker, " ".join(cur_lines).strip()))
    return results


def synthesize_dialogue(
    script, language, num_speakers, num_step, guidance, denoise, speed, duration,
    pause, preprocess, postprocess,
    s1_audio, s1_ref, s1_instr, s1_lang,
    s2_audio, s2_ref, s2_instr, s2_lang,
    s3_audio, s3_ref, s3_instr, s3_lang,
    s4_audio, s4_ref, s4_instr, s4_lang,
):
    if not script or not str(script).strip():
        return None, "Input dialogue required."
    turns = parse_dialogue(script)
    if not turns:
        return None, "No lines found. Use format [Speaker_N]: line"
    n = int(num_speakers or 2)
    speakers = {
        1: {"audio": s1_audio, "ref": s1_ref, "instr": s1_instr, "lang": s1_lang},
        2: {"audio": s2_audio, "ref": s2_ref, "instr": s2_instr, "lang": s2_lang},
        3: {"audio": s3_audio, "ref": s3_ref, "instr": s3_instr, "lang": s3_lang},
        4: {"audio": s4_audio, "ref": s4_ref, "instr": s4_instr, "lang": s4_lang},
    }
    global_lang = language if language and language != "Auto" else None
    gen_conf = OmniVoiceGenerationConfig(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance or 2.0),
        denoise=bool(denoise),
        preprocess_prompt=bool(preprocess),
        postprocess_output=bool(postprocess),
    )
    prompts, audios = {}, []
    for idx, (spk, line) in enumerate(turns, 1):
        if not (1 <= spk <= n):
            return None, f"Line {idx}: Speaker {spk} not in 1..{n}"
        cfg = speakers.get(spk, {})
        lang = cfg.get("lang")
        turn_lang = lang if lang and lang != "Auto" else global_lang
        kw = dict(text=line, language=turn_lang, generation_config=gen_conf)
        if speed and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration and float(duration) > 0:
            kw["duration"] = float(duration)
        ref_audio = cfg.get("audio")
        ref_text = (cfg.get("ref") or "").strip()
        instr = (cfg.get("instr") or "").strip()
        if ref_audio:
            if spk not in prompts:
                prompts[spk] = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text or None)
            kw["voice_clone_prompt"] = prompts[spk]
        if instr:
            kw["instruct"] = instr
        try:
            result = model.generate(**kw)
        except Exception as e:
            return None, f"Speaker {spk}, line {idx}: {type(e).__name__}: {e}"
        arr = result[0].squeeze(0)
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu()
        audios.append(arr.numpy().astype(np.float32))
    if not audios:
        return None, "No output."
    if pause and float(pause) > 0:
        silence = np.zeros(int(float(pause) * sampling_rate), dtype=np.float32)
        merged = audios[0]
        for seg in audios[1:]:
            merged = np.concatenate([merged, silence, seg], 0)
    else:
        merged = np.concatenate(audios, 0)
    import gradio as gr
    waveform = np.clip(merged, -1, 1)
    return gr.update(value=(sampling_rate, (waveform * 32767).astype(np.int16)), visible=True), f"Done. {len(turns)} lines."


def speaker_box_visibility(num_speakers):
    n = int(num_speakers or 2)
    return [gr.update(visible=(i <= n)) for i in range(1, 5)]


def append_tag_to_text(text, tag):
    t = text or ""
    if not tag:
        return t, gr.update(value=None)
    if not t:
        return tag, gr.update(value=None)
    sep = "" if t.endswith((" ", "\n")) else " "
    return f"{t}{sep}{tag}", gr.update(value=None)


# --- FEATURE 1: Saved Voices Logic ---
SAVED_VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_voices")
os.makedirs(SAVED_VOICES_DIR, exist_ok=True)

def get_saved_voices():
    if not os.path.exists(SAVED_VOICES_DIR):
        return []
    envs = [d for d in os.listdir(SAVED_VOICES_DIR) if os.path.isdir(os.path.join(SAVED_VOICES_DIR, d))]
    return sorted(envs)

def save_voice(voice_name, ref_audio_path, ref_text):
    if not voice_name or not str(voice_name).strip():
        return "Error: Voice Name is required.", gr.update()
    if not ref_audio_path:
         return "Error: Reference Audio is required.", gr.update()
         
    voice_name = str(voice_name).strip()
    target_dir = os.path.join(SAVED_VOICES_DIR, voice_name)
    os.makedirs(target_dir, exist_ok=True)
    
    ext = os.path.splitext(ref_audio_path)[1]
    if not ext:
        ext = ".wav"
    out_audio = os.path.join(target_dir, f"audio{ext}")
    
    try:
        shutil.copy2(ref_audio_path, out_audio)
        with open(os.path.join(target_dir, "transcript.txt"), "w", encoding="utf-8") as f:
            f.write(str(ref_text) if ref_text else "")
    except Exception as e:
        return f"Error saving voice: {str(e)}", gr.update()
        
    return f"Success: Saved '{voice_name}' preset.", gr.update(choices=get_saved_voices(), value=voice_name)

def load_voice(voice_name):
    if not voice_name:
        return None, ""
    
    target_dir = os.path.join(SAVED_VOICES_DIR, str(voice_name))
    if not os.path.exists(target_dir):
         return None, ""
         
    audio_path = None
    for f in os.listdir(target_dir):
        if f.startswith("audio."):
             audio_path = os.path.join(target_dir, f)
             break
             
    transcript = ""
    txt_path = os.path.join(target_dir, "transcript.txt")
    if os.path.exists(txt_path):
         with open(txt_path, "r", encoding="utf-8") as f:
             transcript = f.read()
             
    return audio_path, transcript


# --- FEATURE 2: Export Logic ---
def export_file(audio_data, save_dir, filename):
    if audio_data is None:
        return "Error: No generated audio to export."
    if not save_dir or not str(save_dir).strip():
        return "Error: Save Directory is required."
    if not filename or not str(filename).strip():
        return "Error: Filename is required."
        
    save_dir = str(save_dir).strip()
    filename = str(filename).strip()
    if filename.endswith(".wav"):
        filename = filename[:-4]
        
    try:
        os.makedirs(save_dir, exist_ok=True)
        out_file = os.path.join(save_dir, f"{filename}.wav")
        if isinstance(audio_data, str):
            shutil.copy2(audio_data, out_file)
        elif isinstance(audio_data, tuple):
            sr, y = audio_data
            if y.dtype == np.int16:
                y = y.astype(np.float32) / 32768.0
            sf.write(out_file, y, sr)
        else:
            return "Error: Unknown audio data format."
        
        save_single_setting("last_export_dir", save_dir)
        return f"Success: Audio saved to {out_file}"
    except Exception as e:
        return f"Export Error: {e}"


# --- Gradio/Spaces Wrap ---

def _parse_batch_sections(raw_text: str) -> list[str]:
    """Split raw script text into sections on lines that are exactly '---'."""
    raw_text = raw_text.replace('\r\n', '\n').replace('\r', '\n')
    parts = re.split(r"^\s*---\s*$", raw_text, flags=re.MULTILINE)
    sections = []
    for part in parts:
        stripped = part.strip()
        if stripped:
            sections.append(stripped)
    return sections

def run_batch_generation(
    script, lang, ref_audio, ref_text, ns, gs, dn, sp, du, pp, po,
    save_dir, project_name, progress=gr.Progress()
):
    """Generator: yields (status_html, total_progress, last_audio_path)"""
    sections = _parse_batch_sections(script)
    if not sections:
        return "<p style='color:red;'>No sections found. Use '---' on a new line to separate sections.</p>", 0, None

    # Resolve output directory
    base_dir = Path(save_dir.strip()) if save_dir.strip() else Path(os.path.dirname(__file__)) / "batch_output"
    if project_name.strip():
        out_dir = base_dir / project_name.strip()
    else:
        out_dir = base_dir
        
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return f"<p style='color:red;'>Folder Error: {e}</p>", 0, None

    total = len(sections)
    history_html = []
    last_audio = None
    
    for idx, text in enumerate(sections):
        label = f"Processing section {idx+1} of {total}..."
        progress(idx/total, desc=label)
        
        # UI Status Preview
        preview = (text[:60] + "...") if len(text) > 60 else text
        current_status = f"<div style='color:#ffc107;'>⏳ <b>Section {idx+1}:</b> {preview}</div>"
        yield "".join(history_html) + current_status, (idx/total)*100, last_audio

        try:
            # Call single generation
            # Note: ref_audio and ref_text are passed as they are in clone mode
            audio_tuple, msg = synthesize(
                text, lang, ref_audio, None, ns, gs, dn, sp, du, pp, po, mode="clone", ref_text=ref_text
            )
            
            if not audio_tuple:
                raise Exception(msg or "Failed to generate audio segment.")
            
            sr, wav_data = audio_tuple
            filename = f"{idx+1:02d}.wav"
            out_path = out_dir / filename
            
            # Save to disk
            sf.write(str(out_path), wav_data, sr)
            
            # Copy to CWD for Gradio preview (avoids InvalidPathError for external folders)
            preview_file = "batch_preview_latest.wav"
            shutil.copy(str(out_path), preview_file)
            last_audio = preview_file
            
            # Record success
            history_html.append(f"<div style='color:#28a745;'>✅ <b>Section {idx+1}:</b> {preview} → <code>{filename}</code></div>")
            
        except Exception as e:
            history_html.append(f"<div style='color:#dc3545;'>❌ <b>Section {idx+1}:</b> {preview} Fail: {str(e)[:80]}</div>")
            
        yield "".join(history_html), ((idx+1)/total)*100, last_audio

    final_msg = f"<div style='font-weight:bold;margin-top:10px;'>Done! {total} sections processed. Files in: <code>{out_dir}</code></div>"
    yield "".join(history_html) + final_msg, 100, last_audio

# Zero-state initialization (Manual load only)

@spaces.GPU(duration=60)
def generate_fn(*a, **kw):
    return synthesize(*a, **kw)


@spaces.GPU(duration=120)
def generate_dialogue_fn(*a, **kw):
    return synthesize_dialogue(*a, **kw)


# --- UI Wrappers ---
def _clone_fn_wrap(text, lang, ref_aud, ref_text, instruct, ns, gs, dn, sp, du, pp, po):
    s = load_settings()
    s.update({'vc_ns': ns, 'vc_gs': gs, 'vc_dn': dn, 'vc_sp': sp, 'vc_du': du, 'vc_pp': pp, 'vc_po': po})
    save_settings(s)
    if not text or not str(text).strip():
        return None, "Please enter the text to synthesize."
    if not ref_aud:
        return None, "Please upload a reference audio."
    audio_tuple, msg = generate_fn(
        text, lang, ref_aud, instruct, ns, gs, dn, sp, du, pp, po, mode="clone", ref_text=ref_text or None
    )
    return audio_tuple


def detect_trim_bounds(base_audio, base_orig_audio):
    import numpy as np
    import soundfile as sf
    if not base_orig_audio or not base_audio or base_orig_audio == base_audio:
        return 0.0, 0.0
        
    orig_y, sr = sf.read(base_orig_audio)
    orig_y = orig_y.astype(np.float32)
    if len(orig_y.shape) > 1:
        orig_y = orig_y.mean(axis=1)

    trim_y, _ = sf.read(base_audio)
    trim_y = trim_y.astype(np.float32)
    if len(trim_y.shape) > 1:
        trim_y = trim_y.mean(axis=1)

    pad_size = len(orig_y) - len(trim_y)
    if pad_size > 0:
        # Use robust Energy Envelope Cross-Correlation to bypass phase/compression noise
        chunk_size = int(sr / 50) # 20 ms chunks (50 Hz resolution)
        
        N_orig = (len(orig_y) // chunk_size) * chunk_size
        e_orig = np.abs(orig_y[:N_orig].reshape(-1, chunk_size)).mean(axis=1)
        
        N_trim = (len(trim_y) // chunk_size) * chunk_size
        e_trim = np.abs(trim_y[:N_trim].reshape(-1, chunk_size)).mean(axis=1)

        m_orig = e_orig - np.mean(e_orig)
        m_trim = e_trim - np.mean(e_trim)
        L = len(m_orig) + len(m_trim) - 1
        
        pad_orig = np.pad(m_orig, (0, L - len(m_orig)))
        pad_trim = np.pad(m_trim[::-1], (0, L - len(m_trim)))

        cc = np.fft.ifft(np.fft.fft(pad_orig) * np.fft.fft(pad_trim)).real
        max_idx = np.argmax(cc)
        
        s_chunk = max(0, max_idx - len(m_trim) + 1)
        s_idx = s_chunk * chunk_size
        e_idx = s_idx + len(trim_y)
        return float(s_idx / sr), float(e_idx / sr)
    return 0.0, 0.0

def _splice_fn_wrap(base_audio, start_time, end_time, text, lang, ref_aud, ref_text, ns, gs, dn, sp, du, pp, po, base_orig_audio=None):
    import gradio as gr
    if not base_audio:
        raise gr.Error("Please upload Base Audio!")
    if not text or not str(text).strip():
        raise gr.Error("Please enter the Fixed Text!")
        
    s = load_settings()
    s.update({'vc_ns': ns, 'vc_gs': gs, 'vc_dn': dn, 'vc_sp': sp, 'vc_du': du, 'vc_pp': pp, 'vc_po': po})
    save_settings(s)
    
    if not ref_aud and s.get("saved_voice"):
        # Auto-fallback if the user forgot to click "Load Voice" for their dropdown preset
        target_dir = os.path.join(SAVED_VOICES_DIR, str(s.get("saved_voice")))
        if os.path.exists(target_dir):
            for f in os.listdir(target_dir):
                if f.startswith("audio."):
                    fallback_path = os.path.join(target_dir, f)
                    print(f"[Audio Splice] Auto-loading missing voice block: {fallback_path}")
                    ref_aud = fallback_path
                    break
            
    if not ref_aud:
        raise gr.Error("Please click 'Load Voice' or upload a Reference Audio!")
    
    import numpy as np
    import soundfile as sf
    try:
        orig_y, sr = sf.read(base_orig_audio if base_orig_audio else base_audio)
        orig_y = orig_y.astype(np.float32)
        if len(orig_y.shape) > 1:
            orig_y = orig_y.mean(axis=1)

        # 1. Provide absolute explicit override if user typed numbers into the UI boxes
        forced_s = float(start_time or 0.0)
        forced_e = float(end_time or 0.0)
        
        if forced_s > 0.0 or forced_e > 0.0:
            s_idx = int(forced_s * sr)
            e_idx = int(forced_e * sr) if forced_e > 0.0 else len(orig_y)
            print(f"[Audio Splice] Manual override detected! Cutting physically from {forced_s}s to {forced_e}s")
            
        elif base_orig_audio and base_orig_audio != base_audio:
            # 2. Run highly accurate Energy Envelope FFT Cross Correlation
            det_start, det_end = detect_trim_bounds(base_audio, base_orig_audio)
            s_idx = int(det_start * sr)
            e_idx = int(det_end * sr) if det_end > 0.0 else len(orig_y)
            print(f"[Audio Splice] Selection successfully locked from {det_start:.3f}s to {det_end:.3f}s via AI")
        else:
            s_idx = 0
            e_idx = len(orig_y)
            
        print(f"[Audio Splice] Splicing out [{s_idx}:{e_idx}] for model interpolation...")

        import time
        t0 = time.time()
        print(f"[Audio Splice] Engine parameters: text='{text}', ref_aud='{ref_aud}', ref_text='{ref_text}'")
        audio_tuple, msg = generate_fn(
            text, lang, ref_aud, None, ns, gs, dn, sp, du, pp, po, mode="clone", ref_text=ref_text or None
        )
        print(f"[Audio Splice] Engine generation returned successfully after {time.time() - t0:.2f}s!")
        if not audio_tuple:
            raise gr.Error(msg or "Model generation completely failed to produce audio!")
            
        new_sr, new_y = audio_tuple
        
        new_y_float = new_y.astype(np.float32) / 32768.0
        if len(new_y_float.shape) > 1:
            new_y_float = new_y_float.mean(axis=1)

        # 1. Strip natural silent padding produced by CosyVoice inference
        threshold = 0.015 # 1.5% amplitude silence threshold
        non_silent = np.where(np.abs(new_y_float) > threshold)[0]
        if len(non_silent) > 0:
            pad = int(sr * 0.1) # Leave 100ms natural padding
            start_i = max(0, non_silent[0] - pad)
            end_i = min(len(new_y_float), non_silent[-1] + pad)
            new_y_float = new_y_float[start_i:end_i]

        prefix = orig_y[:s_idx]
        suffix = orig_y[e_idx:]
        
        # 2. Apply 50ms Cross-fading for seamless transitions
        fade_len = int(sr * 0.05)
        if len(prefix) > fade_len and len(suffix) > fade_len and len(new_y_float) > fade_len * 2:
            fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            
            cross1 = (prefix[-fade_len:] * fade_out) + (new_y_float[:fade_len] * fade_in)
            cross2 = (new_y_float[-fade_len:] * fade_out) + (suffix[:fade_len] * fade_in)
            
            final_y = np.concatenate([
                prefix[:-fade_len], 
                cross1, 
                new_y_float[fade_len:-fade_len], 
                cross2, 
                suffix[fade_len:]
            ])
        else:
            final_y = np.concatenate([prefix, new_y_float, suffix])
        
        # FIX: The concatenated file can be massive. If we return a 35MB raw Numpy tuple,
        # Starlette WebSockets will silently crash on the payload limit, leaving the UI hanging forever.
        # Instead, we manually encode it to a disk WAV file and route around the WebSocket!
        out_temp = os.path.abspath(os.path.join(os.path.dirname(__file__), "output_splice.wav"))
        sf.write(out_temp, final_y, sr)
        
        return out_temp
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Audio Splice Failed: {str(e)}")

def _design_fn_wrap(text, lang, ns, gs, dn, sp, du, pp, po, *groups):
    s = load_settings()
    s.update({'vd_ns': ns, 'vd_gs': gs, 'vd_dn': dn, 'vd_sp': sp, 'vd_du': du, 'vd_pp': pp, 'vd_po': po})
    save_settings(s)
    if not text or not str(text).strip():
        return None, "Please enter the text to synthesize."
    selected = [g for g in groups if g and g != "Auto"]
    instruct = None
    if selected:
        parts = []
        for v in selected:
            if " / " in v:
                en, zh = v.split(" / ", 1)
                if "Dialect" in en:
                    parts.append(zh.strip())
                else:
                    parts.append(en.strip())
            else:
                parts.append(v)
        instruct = ", ".join(parts)
        
    import gradio as gr
    audio_tuple, msg = generate_fn(
        text, lang, None, instruct, ns, gs, dn, sp, du, pp, po, mode="design"
    )
    return gr.update(value=audio_tuple, visible=True) if audio_tuple else gr.update()

# --- Build App UI ---

theme = gr.themes.Base(primary_hue=gr.themes.colors.orange, secondary_hue=gr.themes.colors.orange, neutral_hue=gr.themes.colors.slate)
css = """
.compact-audio { min-height: 150px !important; }
.speaker-box { border: 1px solid #ddd; padding: 10px; border-radius: 8px; margin-bottom: 10px; }
"""

import base64
def get_base64_logo():
    logo_path = os.path.join(os.path.dirname(__file__), "AminMedia_WHT_INKBLEED.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return ""

b64_img = get_base64_logo()

# --- Build App UI ---

with gr.Blocks(title="Amin OmniVoice") as demo:
    gr.HTML(f"""
    <style>
    #custom-brand-header {{ margin-bottom: -5px !important; margin-top: -10px !important; display: flex; align-items: center; gap: 20px; }}
    .padded-group {{ padding: 15px !important; box-sizing: border-box; }}
    </style>
    <div id="custom-brand-header">
        <img src="data:image/png;base64,{b64_img}" style="height: 85px; width: auto; display: block;" />
        <span style="font-size: 4em; font-weight: 800; line-height: 1; margin: 0; padding: 0;">OmniVoice</span>
    </div>
    """)

    with gr.Tabs():
        # --- Voice Clone Tab ---
        with gr.Tab("Voice Clone"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("<span style='margin-left: 10px;'>**Saved Voices (Voice Preset System)**</span>")
                        with gr.Row():
                            saved_voice_dropdown = gr.Dropdown(value=USER_SETTINGS.get("saved_voice", None), choices=get_saved_voices(), label="Load Saved Voice", interactive=True)
                            load_voice_btn = gr.Button("Load Voice", size="sm")

                        with gr.Row():
                            voice_name_input = gr.Textbox(label="Save Voice As", placeholder="Premium Narrator")
                            save_voice_btn = gr.Button("Save Current Voice", size="sm")
                            clear_voice_btn = gr.Button("Clear Ref", size="sm", variant="secondary")
                        save_status = gr.Textbox(label="", show_label=False, interactive=False, max_lines=1)
                        
                    vc_text = gr.Textbox(
                        label="Text to Synthesize",
                        lines=4,
                        placeholder="Enter the text you want to synthesize...",
                    )
                    vc_ref_audio = gr.Audio(
                        label="Reference Audio",
                        type="filepath",
                        elem_classes="compact-audio",
                    )
                    vc_ref_text = gr.Textbox(
                        label="Reference Text (optional)",
                        lines=2,
                        placeholder="Transcript of the reference audio. Leave empty to auto-transcribe.",
                    )
                    vc_lang = gr.Dropdown(label="Language (optional)", choices=LANGUAGES, value="Auto", allow_custom_value=True)
                    vc_instruct = gr.Textbox(label="Instruct (optional)", lines=2)
                    
                with gr.Column(scale=1):
                    vc_btn = gr.Button("Generate", variant="primary", size="lg")
                    vc_audio = gr.Audio(
                        label="Output Audio [VC]",
                        type="numpy",
                        interactive=False,
                        elem_classes="compact-audio"
                    )
                    with gr.Group():
                        gr.Markdown("<span style='margin-left: 10px;'>**Export to PC**</span>")
                        with gr.Row():
                            exp_dir = gr.Textbox(label="Save Directory", value=USER_SETTINGS.get("last_export_dir", ""), placeholder="C:\\MyProject\\Audio", scale=4)
                            exp_browse = gr.Button("Browse...", min_width=80, scale=1)
                        exp_name = gr.Textbox(label="Filename (without .wav)", placeholder="generated_audio")
                        exp_btn = gr.Button("Save to PC", size="sm")
                        exp_status = gr.Textbox(label="", show_label=False, interactive=False, max_lines=1)
                        
                    with gr.Accordion("Generation Settings [VC]", open=True):
                        vc_sp = gr.Slider(0.5, 1.5, value=USER_SETTINGS.get("vc_sp", 1.0), step=0.05, label="Speed")
                        vc_du = gr.Number(value=USER_SETTINGS.get("vc_du", 0.0), label="Duration (seconds)")
                        vc_ns = gr.Slider(4, 64, value=USER_SETTINGS.get("vc_ns", 32), step=1, label="Inference Steps")
                        vc_dn = gr.Checkbox(label="Denoise", value=USER_SETTINGS.get("vc_dn", True))
                        vc_gs = gr.Slider(0.0, 4.0, value=USER_SETTINGS.get("vc_gs", 2.0), step=0.1, label="Guidance Scale")
                        vc_pp = gr.Checkbox(label="Preprocess Prompt", value=True)
                        vc_po = gr.Checkbox(label="Postprocess Output", value=True)

            # Wiring Voice Clone Features
            load_voice_btn.click(
                load_voice,
                inputs=[saved_voice_dropdown],
                outputs=[vc_ref_audio, vc_ref_text]
            )
            saved_voice_dropdown.change(
                load_voice,
                inputs=[saved_voice_dropdown],
                outputs=[vc_ref_audio, vc_ref_text]
            ).then(
                lambda v: save_settings({**load_settings(), 'saved_voice': v}), 
                inputs=[saved_voice_dropdown], outputs=[]
            )
            save_voice_btn.click(
                save_voice,
                inputs=[voice_name_input, vc_ref_audio, vc_ref_text],
                outputs=[save_status, saved_voice_dropdown]
            )
            clear_voice_btn.click(lambda: (None, ""), inputs=[], outputs=[vc_ref_audio, vc_ref_text])

            vc_btn.click(
                _clone_fn_wrap,
                inputs=[vc_text, vc_lang, vc_ref_audio, vc_ref_text, vc_instruct, vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po],
                outputs=[vc_audio]
            )
            
            exp_browse.click(browse_directory_host, inputs=[], outputs=[exp_dir])
            exp_btn.click(
                export_file,
                inputs=[vc_audio, exp_dir, exp_name],
                outputs=[exp_status]
            )

        # --- Audio Splice Tab ---
        with gr.Tab("Audio Splice"):
            with gr.Row():
                with gr.Column(scale=1):
                    sp_base_orig_state = gr.State(None)
                    sp_base_audio = gr.Audio(label="Base Audio to Edit (Trim with scissors to select automatically!)", type="filepath", editable=True, elem_classes="compact-audio")
                    sp_base_audio.upload(lambda x: x, inputs=[sp_base_audio], outputs=[sp_base_orig_state])
                    sp_base_audio.clear(lambda: None, outputs=[sp_base_orig_state])
                    
                    with gr.Row():
                        sp_mark_start = gr.Button("👇 Grab Start Time", min_width=100, variant="secondary")
                        sp_mark_end = gr.Button("👇 Grab End Time", min_width=100, variant="secondary")
                    
                    with gr.Row():
                        sp_start = gr.Number(value=0.0, label="Exact Start Time Override (s)", min_width=120, info="Leave 0 to use Visual Trimmer")
                        sp_end = gr.Number(value=0.0, label="Exact End Time Override (s)", min_width=120, info="Leave 0 to use Visual Trimmer")
                    
                    sp_text = gr.Textbox(label="Fixed Text (Replacement Chunk)", lines=2, placeholder="Type the corrected sentence or word here...")

                    with gr.Accordion("Generation Settings [SP]", open=False):
                        sp_sp = gr.Slider(0.5, 1.5, value=USER_SETTINGS.get("vc_sp", 1.0), step=0.05, label="Speed")
                        sp_du = gr.Number(value=USER_SETTINGS.get("vc_du", 0.0), label="Duration (seconds)")
                        sp_ns = gr.Slider(4, 64, value=USER_SETTINGS.get("vc_ns", 32), step=1, label="Inference Steps")
                        sp_dn = gr.Checkbox(label="Denoise", value=USER_SETTINGS.get("vc_dn", True))
                        sp_gs = gr.Slider(0.0, 4.0, value=USER_SETTINGS.get("vc_gs", 2.0), step=0.1, label="Guidance Scale")
                        sp_pp = gr.Checkbox(label="Preprocess Prompt", value=True)
                        sp_po = gr.Checkbox(label="Postprocess Output", value=True)

                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("<span style='margin-left: 10px;'>**Saved Voices (Load Preset via Dropdown)**</span>")
                        with gr.Row():
                            sp_saved_voice_dropdown = gr.Dropdown(value=USER_SETTINGS.get("saved_voice", None), choices=get_saved_voices(), label="Load Saved Voice", interactive=True)
                            with gr.Column(min_width=100):
                                sp_load_voice_btn = gr.Button("Load Voice", size="sm")
                                sp_unload_voice_btn = gr.Button("Unload Voice", size="sm", variant="secondary")

                    sp_btn = gr.Button("Perform Audio Surgery", variant="primary", size="lg")
                    sp_audio = gr.Audio(label="Output Audio [SP]", type="filepath", interactive=False, elem_classes="compact-audio")
                    
                    with gr.Group():
                        gr.Markdown("<span style='margin-left: 10px;'>**Export to PC**</span>")
                        with gr.Row():
                            sp_exp_dir = gr.Textbox(label="Save Directory", value=USER_SETTINGS.get("last_export_dir", ""), placeholder="C:\\MyProject\\Audio", scale=4)
                            sp_exp_browse = gr.Button("Browse...", min_width=80, scale=1)
                        sp_exp_name = gr.Textbox(label="Filename (without .wav)", placeholder="generated_audio")
                        sp_exp_btn = gr.Button("Save to PC", size="sm")
                        sp_exp_status = gr.Textbox(label="", show_label=False, interactive=False, max_lines=1)
                    
                    sp_ref_audio = gr.File(label="Reference Audio (.wav)", file_count="single")
                    sp_ref_text = gr.Textbox(label="Reference Text (optional)", lines=1)
                    sp_lang = gr.Dropdown(label="Language [SP]", choices=LANGUAGES, value="Auto", allow_custom_value=True)
                    
            sp_load_voice_btn.click(
                load_voice,
                inputs=[sp_saved_voice_dropdown],
                outputs=[sp_ref_audio, sp_ref_text]
            )
            sp_saved_voice_dropdown.change(
                load_voice,
                inputs=[sp_saved_voice_dropdown],
                outputs=[sp_ref_audio, sp_ref_text]
            ).then(
                lambda v: save_settings({**load_settings(), 'saved_voice': v}), 
                inputs=[sp_saved_voice_dropdown], outputs=[]
            )
            sp_unload_voice_btn.click(lambda: (None, ""), inputs=[], outputs=[sp_ref_audio, sp_ref_text])
            
            sp_btn.click(
                _splice_fn_wrap,
                inputs=[sp_base_audio, sp_start, sp_end, sp_text, sp_lang, sp_ref_audio, sp_ref_text, sp_ns, sp_gs, sp_dn, sp_sp, sp_du, sp_pp, sp_po, sp_base_orig_state],
                outputs=[sp_audio]
            )
            sp_exp_browse.click(browse_directory_host, inputs=[], outputs=[sp_exp_dir])
            sp_exp_btn.click(
                export_file,
                inputs=[sp_audio, sp_exp_dir, sp_exp_name],
                outputs=[sp_exp_status]
            )
            
            _js_fetch_time = """function(curr) {
                let audios = [];
                function traverse(root) {
                    if (!root) return;
                    if (root.tagName === 'AUDIO') audios.push(root);
                    if (root.shadowRoot) traverse(root.shadowRoot);
                    for (let child of root.children) traverse(child);
                }
                traverse(document.body);
                
                if (audios.length > 0) {
                    for (let a of audios) {
                        if (!a.paused) return Number(a.currentTime.toFixed(3));
                    }
                    for (let a of audios) {
                        if (a.currentTime > 0) return Number(a.currentTime.toFixed(3));
                    }
                    return Number(audios[0].currentTime.toFixed(3));
                }
                return curr;
            }"""
            sp_mark_start.click(fn=None, inputs=[sp_start], outputs=[sp_start], js=_js_fetch_time)
            sp_mark_end.click(fn=None, inputs=[sp_end], outputs=[sp_end], js=_js_fetch_time)


        # --- Batch Generation Tab ---
        with gr.TabItem("Batch Generation"):
            with gr.Row():
                with gr.Column(scale=1):
                    batch_script = gr.Textbox(
                        label="Batch Script",
                        lines=12,
                        placeholder="Section 1 text here...\n---\nSection 2 text here...\n---\nSection 3 text here...",
                        info="Use '---' on a new line to separate sections."
                    )
                    
                    with gr.Group():
                        gr.Markdown("<span style='margin-left: 10px;'>**Saved Voices (Load Preset)**</span>")
                        with gr.Row():
                            batch_saved_voice_dropdown = gr.Dropdown(value=USER_SETTINGS.get("saved_voice", None), choices=get_saved_voices(), label="Load Saved Voice", interactive=True)
                            with gr.Column(min_width=100):
                                batch_load_voice_btn = gr.Button("Load Voice", size="sm")
                                batch_unload_voice_btn = gr.Button("Unload Voice", size="sm", variant="secondary")
                    
                    with gr.Accordion("Generation Settings [Batch]", open=False):
                        batch_sp = gr.Slider(0.5, 1.5, value=USER_SETTINGS.get("vc_sp", 1.0), step=0.05, label="Speed")
                        batch_du = gr.Number(value=USER_SETTINGS.get("vc_du", 0.0), label="Duration (seconds)")
                        batch_ns = gr.Slider(4, 64, value=USER_SETTINGS.get("vc_ns", 32), step=1, label="Inference Steps")
                        batch_dn = gr.Checkbox(label="Denoise", value=USER_SETTINGS.get("vc_dn", True))
                        batch_gs = gr.Slider(0.0, 4.0, value=USER_SETTINGS.get("vc_gs", 2.0), step=0.1, label="Guidance Scale")
                        batch_pp = gr.Checkbox(label="Preprocess Prompt", value=True)
                        batch_po = gr.Checkbox(label="Postprocess Output", value=True)

                with gr.Column(scale=1):
                    batch_btn = gr.Button("Process Batch Generation", variant="primary", size="lg")
                    
                    with gr.Group():
                        gr.Markdown("<span style='margin-left: 10px;'>**Output Configuration**</span>")
                        with gr.Row():
                            batch_save_dir = gr.Textbox(label="Base Save Directory", value=USER_SETTINGS.get("last_export_dir", ""), placeholder="C:\\Audio\\Exports", scale=4)
                            batch_save_browse = gr.Button("Browse...", min_width=80, scale=1)
                        batch_project_name = gr.Textbox(label="Subfolder / Project Name (Optional)", placeholder="My_Audio_Book", info="If empty, saves directly into the Base Save Directory.")
                    
                    batch_progress = gr.Slider(0, 100, value=0, label="Total Progress (%)", interactive=False)
                    batch_status_log = gr.HTML(label="Generation Log", value="<div style='color:gray;'>Waiting for input...</div>")
                    batch_preview = gr.Audio(label="Last Generated Segment Preview", type="filepath", interactive=False)
                    
                    gr.Markdown("<span style='margin-left: 10px;'>**Reference Info (Active)**</span>")
                    batch_ref_audio = gr.Audio(label="Loaded Reference Audio", type="filepath", interactive=False, elem_classes="compact-audio")
                    batch_ref_text = gr.Textbox(label="Loaded Reference Text", lines=2, interactive=False)
                    batch_lang = gr.Dropdown(label="Language [Batch]", choices=LANGUAGES, value="Auto", allow_custom_value=True)

            batch_load_voice_btn.click(
                load_voice,
                inputs=[batch_saved_voice_dropdown],
                outputs=[batch_ref_audio, batch_ref_text]
            )
            batch_saved_voice_dropdown.change(
                load_voice,
                inputs=[batch_saved_voice_dropdown],
                outputs=[batch_ref_audio, batch_ref_text]
            ).then(
                lambda v: save_settings({**load_settings(), 'saved_voice': v}), 
                inputs=[batch_saved_voice_dropdown], outputs=[]
            )
            batch_unload_voice_btn.click(lambda: (None, ""), inputs=[], outputs=[batch_ref_audio, batch_ref_text])
            batch_save_browse.click(browse_directory_host, inputs=[], outputs=[batch_save_dir])

            batch_btn.click(
                run_batch_generation,
                inputs=[
                    batch_script, batch_lang, batch_ref_audio, batch_ref_text,
                    batch_ns, batch_gs, batch_dn, batch_sp, batch_du, batch_pp, batch_po,
                    batch_save_dir, batch_project_name
                ],
                outputs=[batch_status_log, batch_progress, batch_preview]
            )

        with gr.Tab("Dialogue"):
            gr.Markdown("Generate multi-speaker dialogue with `[Speaker_N]:` tags and per-speaker voice cloning.")
            script = gr.Textbox(label="Dialogue Script", lines=10, value="[Speaker_1]: Hello, I'm speaker one.\n[Speaker_2]: Hi! I'm speaker two.", placeholder="[Speaker_1]: First line\n[Speaker_2]: Reply...")
            with gr.Row():
                d_tag = gr.Dropdown(label="Insert Tag", choices=TAG_CHOICES, value=None, allow_custom_value=False, scale=5)
                d_btn = gr.Button("Insert", scale=1, min_width=80)
            gr.Markdown("Tip: use these to insert tags in the script field. Manual entry also works (e.g. CMU tokens for English).")
            with gr.Row():
                d_lang = gr.Dropdown(label="Language", choices=LANGUAGES, value="Auto", allow_custom_value=True, info="Pick or type language code.")
                d_nspeak = gr.Slider(minimum=2, maximum=4, step=1, value=2, label="Number of Speakers")
                d_pause = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=0.3, label="Pause Between Speakers (seconds)")
            with gr.Accordion("Generation Settings", open=False):
                with gr.Row():
                    d_nstep = gr.Slider(minimum=4, maximum=64, step=1, value=32, label="num_step")
                    d_guid = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, value=2.0, label="guidance_scale")
                    d_speed = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0, label="speed")
                    d_dur = gr.Slider(minimum=0.0, maximum=30.0, step=0.5, value=0.0, label="duration (0 = auto)")
                with gr.Row():
                    d_denoise = gr.Checkbox(value=True, label="denoise")
                    d_pre = gr.Checkbox(value=True, label="preprocess_prompt")
                    d_post = gr.Checkbox(value=True, label="postprocess_output")
            spk_boxes, spk_audio, spk_ref, spk_instr, spk_lang = [], [], [], [], []
            with gr.Row():
                # Column 1: Speaker 1 and 3
                with gr.Column(scale=1):
                    for i in [1, 3]:
                        with gr.Column(visible=(i == 1), elem_classes="speaker-box") as box:
                            gr.Markdown(f"**Speaker {i}**")
                            with gr.Group():
                                d_v_drop = gr.Dropdown(value=USER_SETTINGS.get("saved_voice", None), choices=get_saved_voices(), label="Saved Voice", interactive=True)
                                with gr.Row():
                                    d_v_load = gr.Button("Load", size="sm")
                                    d_v_unload = gr.Button("Unload", size="sm", variant="secondary")
                            
                            a = gr.Audio(label=f"Spk {i} Ref", type="filepath", elem_classes="compact-audio")
                            r = gr.Textbox(label=f"Spk {i} Text", lines=1)
                            ins = gr.Textbox(label=f"Spk {i} Style", lines=1)
                            lang = gr.Dropdown(label=f"Spk {i} Lang", choices=LANGUAGES, value="Auto")
                            
                            # Registration
                            # Note: We append to a temporary list and then sort later or just be careful.
                            # Better: Initialize lists with 4 empty and fill by index i-1.
                            
                            # Actually, order in spk_boxes must match range(1,5) for speaker_box_visibility
                            # So I'll just append and re-order the spk_boxes list once done. (See below)
                            
                            d_v_load.click(load_voice, inputs=[d_v_drop], outputs=[a, r])
                            d_v_unload.click(lambda: (None, ""), inputs=[], outputs=[a, r])
                            d_v_drop.change(load_voice, inputs=[d_v_drop], outputs=[a, r])
                            
                            # Store in temp dict to sort later
                            locals()[f"spk_box_{i}"] = box
                            locals()[f"spk_a_{i}"] = a
                            locals()[f"spk_r_{i}"] = r
                            locals()[f"spk_ins_{i}"] = ins
                            locals()[f"spk_l_{i}"] = lang

                # Column 2: Speaker 2 and 4
                with gr.Column(scale=1):
                    for i in [2, 4]:
                        with gr.Column(visible=(i == 2), elem_classes="speaker-box") as box:
                            gr.Markdown(f"**Speaker {i}**")
                            with gr.Group():
                                d_v_drop = gr.Dropdown(value=USER_SETTINGS.get("saved_voice", None), choices=get_saved_voices(), label="Saved Voice", interactive=True)
                                with gr.Row():
                                    d_v_load = gr.Button("Load", size="sm")
                                    d_v_unload = gr.Button("Unload", size="sm", variant="secondary")

                            a = gr.Audio(label=f"Spk {i} Ref", type="filepath", elem_classes="compact-audio")
                            r = gr.Textbox(label=f"Spk {i} Text", lines=1)
                            ins = gr.Textbox(label=f"Spk {i} Style", lines=1)
                            lang = gr.Dropdown(label=f"Spk {i} Lang", choices=LANGUAGES, value="Auto")

                            d_v_load.click(load_voice, inputs=[d_v_drop], outputs=[a, r])
                            d_v_unload.click(lambda: (None, ""), inputs=[], outputs=[a, r])
                            d_v_drop.change(load_voice, inputs=[d_v_drop], outputs=[a, r])

                            locals()[f"spk_box_{i}"] = box
                            locals()[f"spk_a_{i}"] = a
                            locals()[f"spk_r_{i}"] = r
                            locals()[f"spk_ins_{i}"] = ins
                            locals()[f"spk_l_{i}"] = lang

            # Assemble lists in correct order (1-4)
            for i in range(1, 5):
                spk_boxes.append(locals()[f"spk_box_{i}"])
                spk_audio.append(locals()[f"spk_a_{i}"])
                spk_ref.append(locals()[f"spk_r_{i}"])
                spk_instr.append(locals()[f"spk_ins_{i}"])
                spk_lang.append(locals()[f"spk_l_{i}"])

            d_run = gr.Button("Generate Dialogue", variant="primary")
            d_audio = gr.Audio(label="Dialogue Output", elem_classes="compact-audio")
            d_status = gr.Textbox(label="Status", interactive=False)
            d_nspeak.change(fn=speaker_box_visibility, inputs=[d_nspeak], outputs=spk_boxes)
            d_btn.click(append_tag_to_text, inputs=[script, d_tag], outputs=[script, d_tag])
            d_run.click(
                generate_dialogue_fn,
                inputs=[
                    script, d_lang, d_nspeak, d_nstep, d_guid, d_denoise, d_speed, d_dur, d_pause, d_pre, d_post,
                    spk_audio[0], spk_ref[0], spk_instr[0], spk_lang[0],
                    spk_audio[1], spk_ref[1], spk_instr[1], spk_lang[1],
                    spk_audio[2], spk_ref[2], spk_instr[2], spk_lang[2],
                    spk_audio[3], spk_ref[3], spk_instr[3], spk_lang[3],
                ],
                outputs=[d_audio, d_status],
                api_name="generate_dialogue"
            )

        # --- Voice Design Tab (Moved to End) ---
        with gr.Tab("Voice Design"):
            with gr.Row():
                with gr.Column(scale=1):
                    vd_text = gr.Textbox(
                        label="Text to Synthesize",
                        lines=4,
                        placeholder="Enter the text you want to synthesize...",
                    )
                    vd_lang = gr.Dropdown(label="Language [VD]", choices=LANGUAGES, value="Auto", allow_custom_value=True)

                    vd_groups = []
                    for _cat, _choices in _CATEGORIES.items():
                        vd_groups.append(
                            gr.Dropdown(
                                label=_cat,
                                choices=["Auto"] + _choices,
                                value="Auto",
                                info=_ATTR_INFO.get(_cat),
                            )
                        )

                with gr.Column(scale=1):
                    vd_btn = gr.Button("Generate", variant="primary", size="lg")
                    vd_audio = gr.Audio(
                        label="Output Audio [VD]",
                        type="numpy",
                        interactive=False,
                        visible=False,
                        elem_classes="compact-audio"
                    )

                    with gr.Group():
                        gr.Markdown("<span style='margin-left: 10px;'>**Export Output [VD]**</span>")
                        with gr.Row():
                            vd_exp_dir = gr.Textbox(label="Save Directory", value=USER_SETTINGS.get("last_export_dir", ""), placeholder="C:\\MyProject\\Audio", scale=4)
                            vd_exp_browse = gr.Button("Browse...", min_width=80, scale=1)
                        vd_exp_name = gr.Textbox(label="Filename (without .wav)", placeholder="generated_audio")
                        vd_exp_btn = gr.Button("Save to PC", size="sm")
                        vd_exp_status = gr.Textbox(label="", show_label=False, interactive=False, max_lines=1)

                    with gr.Accordion("Generation Settings [VD]", open=True):
                        vd_sp = gr.Slider(0.5, 1.5, value=USER_SETTINGS.get("vd_sp", 1.0), step=0.05, label="Speed")
                        vd_du = gr.Number(value=USER_SETTINGS.get("vd_du", 0.0), label="Duration (seconds)")
                        vd_ns = gr.Slider(4, 64, value=USER_SETTINGS.get("vd_ns", 32), step=1, label="Inference Steps")
                        vd_dn = gr.Checkbox(label="Denoise", value=USER_SETTINGS.get("vd_dn", True))
                        vd_gs = gr.Slider(0.0, 4.0, value=USER_SETTINGS.get("vd_gs", 2.0), step=0.1, label="Guidance Scale")
                        vd_pp = gr.Checkbox(label="Preprocess Prompt", value=True)
                        vd_po = gr.Checkbox(label="Postprocess Output", value=True)

            vd_btn.click(
                _design_fn_wrap,
                inputs=[vd_text, vd_lang, vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po] + vd_groups,
                outputs=[vd_audio]
            )
            
            vd_exp_browse.click(browse_directory_host, inputs=[], outputs=[vd_exp_dir])
            vd_exp_btn.click(
                export_file,
                inputs=[vd_audio, vd_exp_dir, vd_exp_name],
                outputs=[vd_exp_status]
            )

    # demo.load disabled to prevent UI freeze

if __name__ == "__main__":
    launch_args = {"inbrowser": True, "share": False}
    host = os.environ.get("OMNIVOICE_HOST", "127.0.0.1")
    launch_args["server_name"] = host
    port = env_int("OMNIVOICE_PORT", "PORT", "GRADIO_SERVER_PORT")
    if port:
        launch_args["server_port"] = port
    
    launch_args.update({"theme": theme, "css": css})
    
    try:
        demo.queue().launch(**launch_args)
    except OSError:
        # If the specific port from environment is busy, let Gradio pick the next available
        print("[WARNING] Requested port was busy. Finding another available port...")
        if "server_port" in launch_args:
            del launch_args["server_port"]
        demo.queue().launch(**launch_args)
