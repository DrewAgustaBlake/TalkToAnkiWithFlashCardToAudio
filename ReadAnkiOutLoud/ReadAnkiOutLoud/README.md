# 🧠🔊 Anki Voice Reviewer
_A hands-free, voice-controlled Anki study assistant powered by OpenAI._

Anki Voice Reviewer lets you **review flashcards without looking at your screen**.  
It reads questions **and** answers out loud, describes images, waits for your **voice commands**, and automatically marks cards as reviewed in Anki.

Perfect for:
- Medical students + residents (USMLE / Step prep, clerkships, call shifts)
- People with busy hands (driving, meal prep, gym)
- ADHD / auditory learners
- Anyone who wants Anki to feel like a personal tutor
 
---

## 🚀 Features

### 🔊 Text-to-Speech Reading
- Reads **questions and answers** using OpenAI TTS.
- Produces natural, human-sounding audio.
- Automatically rewrites clunky flashcard text into **smooth spoken English**.

### 🖼️ Image Descriptions
- Detects images in your Anki notes.
- Uses OpenAI vision models to produce **concise 1–2 sentence descriptions**.

### 🎤 Voice Control
Say:
- **“next” / “continue”** → move to the next card  
- **“answer”** → reveal answer immediately  
- **“repeat”** → hear the answer again  
- **“stop”** → exit the session  

Uses Whisper to transcribe your voice.

### 🔁 Automatic Card Review
After finishing a card, it calls AnkiConnect to **mark the card reviewed** with your chosen ease level.

### 🛠 Compatible with:
- Anki desktop
- AnkiConnect (must be installed)
- OpenAI API (any paid or free tier API key)
- Windows, Linux, macOS

---

## 📦 Requirements

Install Python packages:

```bash
pip install openai requests sounddevice numpy
____________________________________________________________________________________
If you like what I try to build then dontate to me and I can build even more stuff!
MetaMask Any Network: 0x27C3e609158e1A12eDD4F01c3291fb68eC092441
Preferred fee free networks
Nanocurrency: nano_18rrc9sm744x7tccuy1rzsjxms8fq8qxxjs639wrgfa6pyuc3erfngwbuxi7
Hathor HTR: HSHe8UkVSteRHFNxHWG6YCqt6Cc2NUnVGH
