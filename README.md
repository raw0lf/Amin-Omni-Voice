# <img src="app/AminMedia_WHT_INKBLEED.png" height="32">min OmniVoice

**Amin OmniVoice** is a premium, all-in-one AI Audio Studio designed for zero-shot multilingual text-to-speech, professional voice cloning, and advanced audio editing. 

Built on the cutting-edge foundations of [OmniVoice](https://github.com/k2-fsa/OmniVoice), this version has been heavily refined into a standalone Studio experience with custom workflows.

---

## ✨ Key Features

- **🎯 Voice Clone**: Create high-fidelity voice clones in seconds using just a short reference audio clip. Supports 600+ languages.
- **✂️ Audio Surgery (Splice)**: Perform precision edits on existing audio. Replace specific words or phrases while maintaining the original voice's tone and prosody.
- **📦 Batch Generation**: Narrate entire books or long-form content by processing scripts in bulk. Supports direct project-level exports.
- **🗨️ Multi-Speaker Dialogue**: Orchestrate complex conversations with up to 4 distinct speakers, each with their own unique voice clone.
- **🎨 Voice Design**: Generate entirely new, unique voices from scratch by describing their attributes (Pitch, Tone, Emotion, etc.).

---

## 🚀 Getting Started (Standalone)

Amin OmniVoice is designed to run locally on your VPC without requiring complex external launchers.

### 1. Installation
Simply run the installer to set up the virtual environment and download the required libraries:
```bash
install.bat
```
*Note: This will install Python dependencies and PyTorch with CUDA acceleration automatically.*

### 2. Launching
Once installed, use the launcher to start the studio:
```bash
run.bat
```
The browser interface will automatically open at `http://127.0.0.1:7860`.

---

## 🔒 Privacy & Persistence

- **Local First**: All processing happens on your local hardware. No audio is ever sent to external APIs.
- **Smart Git Hygiene**: Your custom voices (saved in `app/saved_voices/`) are automatically ignored by Git. This ensures your private presets stay on your machine and aren't accidentally shared during updates or commits.
- **Integrated Cache**: Model weights and temporary files are stored in the local `cache/` directory for fast, offline-ready performance.

---

## 🛠️ Credits & Attribution

Amin OmniVoice is a forked and enhanced version of the original OmniVoice project. [k2-fsa](https://github.com/k2-fsa/OmniVoice)

- **OmniVoice Core**: [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)
- **Papers**: [arXiv:2604.00688](https://arxiv.org/abs/2604.00688)
