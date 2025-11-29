# First-Time Run Testing Checklist

## Server Status
✅ Development server is running on **http://localhost:3001**

## Step-by-Step First-Time Testing

### 1. Open the Application
- [ ] Open browser and navigate to: **http://localhost:3001**
- [ ] Check browser console (F12) for any errors
- [ ] Verify page loads without errors

### 2. First-Time Setup Detection
- [ ] **Expected**: Welcome message appears: "Welcome! This is your first time. Models will download automatically on first use."
- [ ] Message should display for 8 seconds
- [ ] Check browser console for `firstTimeSetup` event
- [ ] Verify no errors in console

### 3. UI Elements Check
- [ ] Logo displays in header (owl logo, 48x48px)
- [ ] Favicon displays in browser tab (owl logo)
- [ ] Brand colors applied:
  - Deep blue background gradient
  - Purple header text
  - Teal secondary buttons
  - Deep blue primary buttons
- [ ] Industry selector dropdown visible in header
- [ ] "Prompt context:" label visible

### 4. Set PIN
- [ ] Enter a PIN (e.g., "1234")
- [ ] PIN is accepted
- [ ] Application unlocks

### 5. Create New Session
- [ ] Click "New Session" button
- [ ] Fill in session details:
  - Title: "Test Meeting"
  - Participants: "Test User"
  - Date: Today's date
- [ ] Select audio source (Microphone or System Audio)

### 6. Real-Time Transcription Test
- [ ] Click "Record" button
- [ ] **Expected**: Recording starts immediately
- [ ] **Expected**: Timer starts counting
- [ ] Speak clearly for 10-15 seconds
- [ ] **Expected**: Live transcript appears (if SpeechRecognition supported in Chrome/Edge)
- [ ] **Expected**: Transcript updates in real-time
- [ ] Stop recording after 15-20 seconds
- [ ] **Expected**: Audio blob created
- [ ] **Expected**: "Audio recorded" message appears

### 7. Save Session
- [ ] Click "Save Session" button
- [ ] **Expected**: Session saved successfully
- [ ] **Expected**: Session appears in sessions list
- [ ] **Expected**: No errors in console

### 8. AI Analysis Test (First-Time Model Download)
- [ ] Open the saved session
- [ ] Click "Run On-Device Analysis" button
- [ ] **Expected**: Progress indicators appear:
  - "Initializing transcription model..."
  - "Downloading transcription model..." (with progress %)
  - "Transcribing audio..."
  - "Initializing analysis model..."
  - "Downloading analysis model..." (with progress %)
  - "Analyzing transcript..."
  - "Tokenizing input..."
  - "Generating analysis..."
  - "Decoding results..."
- [ ] **Expected**: Timing shows: "X s / Y s remaining"
- [ ] **Expected**: Models download (first time only, 10-30 seconds)
- [ ] **Expected**: Analysis completes within 60 seconds
- [ ] **Expected**: Results displayed:
  - Summary section
  - Action Items section
  - Outline section
  - Transcript with speaker labels

### 9. Industry Context Test
- [ ] Select "Business Meeting" from industry dropdown
- [ ] Create new session
- [ ] Record business-related audio
- [ ] Run analysis
- [ ] **Expected**: Analysis results reflect business context
- [ ] Verify industry selection persists after page reload

### 10. Search Functionality
- [ ] Type in search box in sessions list
- [ ] **Expected**: Sessions filter in real-time
- [ ] Test searching by:
  - Title
  - Participants
  - Date
- [ ] **Expected**: Case-insensitive search
- [ ] Clear search
- [ ] **Expected**: All sessions shown again

### 11. Export Functionality
- [ ] Open session with completed analysis
- [ ] Click "TXT" export button
- [ ] **Expected**: File downloads as `session-[id]-[title].txt`
- [ ] Click "JSON" export button
- [ ] **Expected**: File downloads as `session-[id]-[title].json`
- [ ] Click "MD" export button
- [ ] **Expected**: File downloads as `session-[id]-[title].md`
- [ ] Verify exported files contain complete data

### 12. Speaker Identification
- [ ] Check transcript in session detail
- [ ] **Expected**: Speakers assigned sequentially (Speaker 1, Speaker 2, etc.)
- [ ] **Expected**: Not all hardcoded to "Speaker 1"
- [ ] Click on speaker name to edit
- [ ] **Expected**: Speaker name editable
- [ ] Save changes
- [ ] **Expected**: Changes persist

### 13. Performance Timing
- [ ] Run analysis on a session
- [ ] **Expected**: Progress shows elapsed time
- [ ] **Expected**: Time remaining decreases
- [ ] **Expected**: Final message shows total time: "Analysis complete (X s)"
- [ ] **Expected**: Analysis completes within 60 seconds

### 14. Error Handling
- [ ] Try recording blank/silent audio
- [ ] Run analysis
- [ ] **Expected**: Error message: "No speech detected in the audio"
- [ ] **Expected**: User-friendly error message
- [ ] **Expected**: Retry option available

### 15. Data Persistence
- [ ] Create multiple sessions
- [ ] Close browser
- [ ] Reopen application
- [ ] **Expected**: All sessions persist
- [ ] **Expected**: Industry selection persists
- [ ] **Expected**: PIN still required

## Browser Console Checks

### Expected Console Output (No Errors)
- ✅ No red error messages
- ✅ Models load successfully
- ✅ Transformers.js initializes correctly
- ✅ onnxruntime-web loads from CDN
- ✅ Proxy requests work (check Network tab for `/Xenova/` requests)

### Network Tab Checks
- [ ] `/Xenova/` requests go through proxy
- [ ] Content-Type headers are `application/json` (not `text/plain`)
- [ ] Model files download successfully
- [ ] No 404 errors for models

## Known Limitations to Note

1. **SpeechRecognition API**: Only works in Chrome/Edge. Firefox/Safari will gracefully degrade (recording still works via MediaRecorder).

2. **First-Time Model Download**: Requires internet connection. Takes 10-30 seconds.

3. **Speaker Identification**: Currently sequential assignment (Speaker 1, Speaker 2, etc.), not true diarization.

4. **Performance**: Analysis time varies by audio length and device performance. Target is under 60 seconds.

## Issues to Report

If you encounter any of the following, note them:

- ❌ Console errors (especially during model loading)
- ❌ Models fail to download
- ❌ Analysis times out incorrectly
- ❌ Real-time transcription doesn't work (in Chrome/Edge)
- ❌ Export files are corrupted or incomplete
- ❌ Brand colors not applied correctly
- ❌ Logo doesn't display
- ❌ First-time setup message doesn't appear
- ❌ Data doesn't persist after browser close

## Success Criteria

✅ All critical tests pass
✅ No console errors
✅ Models download and work correctly
✅ Analysis completes within 60 seconds
✅ All features functional
✅ Branding complete (logo, colors)
✅ Data persists correctly

---

**Ready to test!** Open http://localhost:3001 in your browser and follow the checklist above.

