<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# AI Notes

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/hallidayz/ai-notes)

**On-Device, Private, AI Powered Notes**

## Features

- **On-Device Processing**: All AI processing happens locally in your browser - no data sent to external servers
- **Real-Time Transcription**: Live transcription during recording with SpeechRecognition API
- **Enhanced Speaker Identification**: Heuristic-based speaker diarization using silence gaps and timing patterns
- **AI-Powered Analysis**: Automatic generation of summaries, action items, and topic-grouped outlines
- **Multi-Language Support**: Transcription in 12+ languages (English, Spanish, French, German, Italian, Portuguese, Chinese, Japanese, Korean, Russian, Arabic, Hindi)
- **Audio Enhancement**: Built-in noise suppression and audio normalization for better transcription quality
- **Performance Tracking**: Detailed timing breakdown for each processing stage with 60-second SLA
- **Industry Context**: Optimized prompts for therapy, medical, legal, and business meetings
- **PIN-Based Encryption**: All notes encrypted with AES-GCM using your PIN
- **Export Capabilities**: Export sessions as TXT, JSON, or Markdown
- **Search Functionality**: Search sessions by title, participants, or date
- **Task Management**: Convert action items to tasks with priority and due dates

## Repository
- GitHub: https://github.com/hallidayz/ai-notes

## Run Locally

**Prerequisites:** Node.js 18+ and npm

1. Install dependencies:
   ```bash
   npm install
   ```

2. Run the development server:
   ```bash
   npm run dev
   ```

3. Open your browser to `http://localhost:3001`

## First-Time Setup

1. **Set PIN**: Create a 4-digit PIN when you first open the app
2. **Model Download**: AI models will download automatically on first use (requires internet connection)
   - Transcription model: ~75MB (Xenova/whisper-tiny.en or whisper-tiny for multilingual)
   - Analysis model: ~300MB (Xenova/LaMini-Flan-T5-783M)
   - Models are cached in browser IndexedDB for offline use

## Usage

1. **Create Session**: Click "New Session" and fill in details
2. **Record Audio**: 
   - Choose microphone or system audio
   - Click "Record" to start
   - Live transcript appears during recording (Chrome/Edge only)
3. **Save Session**: Recording auto-saves when stopped (if title provided)
4. **Run Analysis**: Click "Run On-Device Analysis" to generate:
   - Summary
   - Action Items
   - Topic-Grouped Outline
   - Speaker-identified Transcript
5. **Export**: Export sessions in TXT, JSON, or Markdown format

## Browser Support

- **Chrome/Edge**: Full support including real-time transcription
- **Firefox**: Core features work, real-time transcription may be limited
- **Safari**: Core features work, may need experimental features enabled

## Technical Details

- **Framework**: React + TypeScript
- **Build Tool**: Vite
- **AI Models**: 
  - Whisper (transcription) via @xenova/transformers
  - LaMini-Flan-T5-783M (analysis) via @xenova/transformers
- **Storage**: IndexedDB for sessions and models
- **Encryption**: Web Crypto API (AES-GCM)
- **Audio Processing**: Web Audio API with noise suppression

## Performance

- **Target**: 60-second processing time for typical recordings
- **Tracking**: Per-stage timing (preprocessing, transcription, analysis)
- **Optimization**: Audio preprocessing, model caching, optimized prompts

## Privacy

- **100% On-Device**: No data leaves your browser
- **No Telemetry**: Zero analytics or tracking
- **Encrypted Storage**: All notes encrypted with your PIN
- **Offline-First**: Works completely offline after initial setup

## Troubleshooting

See `TROUBLESHOOTING.md` for common issues and solutions.

## License

[Add your license here]
