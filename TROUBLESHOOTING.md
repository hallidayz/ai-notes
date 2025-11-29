# Troubleshooting AI Analysis Failures

## Quick Diagnostic Steps

### 1. Check Browser Console (F12)
Open browser DevTools (F12) and check the Console tab for errors. Common errors:

- **"onnxruntime-web failed to load"** → CDN connection issue
- **"Transformers.js not loaded"** → Module import failure
- **"Failed to load transcription model"** → Model download/proxy issue
- **"Content-Type" errors** → Proxy configuration issue
- **Network errors (404, CORS)** → Hugging Face CDN access issue

### 2. Check Network Tab
1. Open DevTools (F12)
2. Go to Network tab
3. Click "Retry Analysis"
4. Look for failed requests:
   - `/Xenova/` requests should go through proxy
   - Model files (`.json`, `.onnx`) should download successfully
   - Check if requests return `200 OK` or errors

### 3. Common Issues and Fixes

#### Issue: "onnxruntime-web failed to load from CDN"
**Cause:** CDN script not loading from jsdelivr.net

**Fix:**
- Check internet connection
- Check if CDN is blocked (corporate firewall, ad blocker)
- Try refreshing the page
- Check browser console for CORS or network errors

#### Issue: "Transformers.js not loaded"
**Cause:** Module import failure

**Fix:**
- Check if `node_modules/@xenova/transformers` exists
- Run `npm install` to reinstall dependencies
- Check browser console for import errors
- Try hard refresh (Ctrl+Shift+R or Cmd+Shift+R)

#### Issue: "Failed to load transcription model"
**Cause:** Model download failure or proxy issue

**Fix:**
- Check Network tab for `/Xenova/whisper-tiny.en/` requests
- Verify proxy is working (requests should go to `localhost:3001/Xenova/...`)
- Check if Hugging Face CDN is accessible
- Try clearing browser cache and IndexedDB:
  ```javascript
  // In browser console:
  indexedDB.deleteDatabase('TherapyDB');
  location.reload();
  ```

#### Issue: "Content-Type" errors or JSON parsing failures
**Cause:** Proxy not fixing Content-Type headers

**Fix:**
- Verify Vite proxy is running (check terminal for dev server)
- Check `vite.config.ts` proxy configuration
- Restart dev server: Stop (Ctrl+C) and run `npm run dev` again

#### Issue: "Analysis timeout: Processing exceeded 60 seconds"
**Cause:** Analysis taking too long

**Fix:**
- This is expected for very long audio files
- Try with shorter audio (30 seconds to 2 minutes)
- Check device performance (close other apps)
- First-time model download adds time (10-30 seconds)

### 4. Step-by-Step Debugging

1. **Verify Server is Running**
   - Check terminal: Should see "VITE v6.x.x ready in XXX ms"
   - URL should be: http://localhost:3001

2. **Check Initial Load**
   - Open http://localhost:3001
   - Check console for errors on page load
   - Verify logo and UI load correctly

3. **Check onnxruntime-web Loading**
   - In console, type: `window.ort`
   - Should return an object (not undefined)
   - If undefined, CDN script didn't load

4. **Check Transformers.js**
   - Wait a few seconds after page load
   - In console, check for any import errors
   - Look for "Failed to initialize transformers.js" errors

5. **Test Model Download**
   - Create a session with audio
   - Click "Run On-Device Analysis"
   - Watch Network tab for `/Xenova/` requests
   - First request should be `tokenizer.json`
   - Should see progress indicators in UI

6. **Check Proxy**
   - In Network tab, click on a `/Xenova/` request
   - Check "Response Headers"
   - `Content-Type` should be `application/json` (not `text/plain`)
   - If `text/plain`, proxy isn't fixing headers

### 5. Reset Everything

If nothing works, try a complete reset:

```bash
# Stop dev server (Ctrl+C)

# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear browser data
# In browser console:
indexedDB.deleteDatabase('TherapyDB');
localStorage.clear();
location.reload();

# Restart dev server
npm run dev
```

### 6. Browser-Specific Issues

#### Chrome/Edge
- Should work fully
- SpeechRecognition API available
- Best compatibility

#### Firefox
- Core features work
- SpeechRecognition may not work (graceful degradation)
- Recording still works via MediaRecorder

#### Safari
- Core features work
- SpeechRecognition may have limited support
- May need to enable experimental features

### 7. Network/Firewall Issues

If behind a corporate firewall:
- Hugging Face CDN may be blocked
- jsdelivr.net CDN may be blocked
- Check with IT about whitelisting:
  - `huggingface.co`
  - `hf.co`
  - `cdn.jsdelivr.net`

### 8. Get Help

When reporting issues, include:
1. Browser and version
2. Error message from console
3. Network tab screenshot (showing failed requests)
4. Steps to reproduce
5. Whether it's first-time or returning user

---

## Expected Behavior

### First-Time User
1. Page loads → Welcome message appears
2. Set PIN → App unlocks
3. Create session → Record audio
4. Run analysis → Models download (10-30 seconds)
5. Analysis completes → Results displayed

### Returning User
1. Page loads → No welcome message
2. Enter PIN → App unlocks
3. Open session → Run analysis
4. Models use cache → Faster (5-15 seconds)
5. Analysis completes → Results displayed

---

## Still Not Working?

1. **Check the exact error message** in browser console
2. **Take a screenshot** of the Network tab showing failed requests
3. **Note your browser and version**
4. **Describe what happens** when you click "Run On-Device Analysis"

The improved error logging will now show detailed error messages in an alert dialog and in the browser console.

