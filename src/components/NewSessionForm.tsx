
import React, { useState, useRef } from 'react';
import { Session } from '../types';
import { AppIcon } from './AppIcon';

interface NewSessionFormProps {
    onAddSession: (session: Omit<Session, 'id' | 'timestamp' | 'notes'>, notes: string, audioBlob: Blob | null) => Promise<boolean>;
    showStatus: (message: string, type: 'success' | 'error' | 'info', duration?: number) => void;
    isDarkMode: boolean;
}

export const NewSessionForm: React.FC<NewSessionFormProps> = ({ onAddSession, showStatus, isDarkMode }) => {
    const [sessionTitle, setSessionTitle] = useState('');
    const [participants, setParticipants] = useState('');
    const [date, setDate] = useState(new Date().toISOString().split('T')[0]);
    const [notes, setNotes] = useState('');
    const [isRecording, setIsRecording] = useState(false);
    const [isSaving, setIsSaving] = useState(false);
    const [recordingType, setRecordingType] = useState<'mic' | 'system'>('mic');
    const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
    const [duration, setDuration] = useState(0);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const chunksRef = useRef<Blob[]>([]);
    const timerRef = useRef<number | null>(null);

    const stopStream = (stream: MediaStream) => {
        stream.getTracks().forEach(track => track.stop());
    };

    const getRecordingErrorMessage = (err: unknown): string => {
        if (err instanceof DOMException) {
            if (err.name === 'NotAllowedError') {
                return recordingType === 'mic'
                    ? 'Microphone access was denied. Allow microphone permission in your browser settings, then try again.'
                    : 'Screen sharing was cancelled or denied. Try Microphone mode, or enable "Share system audio" in the picker.';
            }
            if (err.name === 'NotFoundError') {
                return 'No microphone was found. Connect an audio input device and try again.';
            }
            if (err.name === 'NotSupportedError') {
                return 'Recording is not supported in this browser.';
            }
        }
        return 'Could not start recording. Check permissions and try again.';
    };

    const handleStartRecording = async () => {
        let stream: MediaStream | null = null;
        try {
            if (!navigator.mediaDevices?.getUserMedia) {
                showStatus('Recording is not supported in this browser.', 'error');
                return;
            }

            if (recordingType === 'mic') {
                stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            } else {
                stream = await navigator.mediaDevices.getDisplayMedia({
                    video: true,
                    audio: true,
                });
                stream.getVideoTracks().forEach(track => track.stop());

                if (!stream.getAudioTracks().length) {
                    stopStream(stream);
                    showStatus("System audio was not shared. Enable 'Share system audio' in the browser picker, or switch to Microphone.", 'error', 6000);
                    return;
                }
            }

            setIsRecording(true);
            setDuration(0);
            timerRef.current = window.setInterval(() => setDuration(prev => prev + 1), 1000);

            const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus'
                : 'audio/webm';

            mediaRecorderRef.current = new MediaRecorder(stream, { mimeType });
            mediaRecorderRef.current.ondataavailable = (e) => {
                if (e.data.size > 0) chunksRef.current.push(e.data);
            };
            mediaRecorderRef.current.onstop = () => {
                const blob = new Blob(chunksRef.current, { type: mimeType });
                setAudioBlob(blob);
                chunksRef.current = [];
                stopStream(stream!);
            };
            mediaRecorderRef.current.start();
        } catch (err) {
            if (stream) stopStream(stream);
            setIsRecording(false);
            if (timerRef.current) {
                clearInterval(timerRef.current);
                timerRef.current = null;
            }
            console.error("Error starting recording:", err);
            showStatus(getRecordingErrorMessage(err), 'error', 6000);
        }
    };

    const handleStopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
            if(timerRef.current) clearInterval(timerRef.current);
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!sessionTitle) {
            showStatus('Session title is required.', 'error');
            return;
        }
        setIsSaving(true);
        const success = await onAddSession({
            sessionTitle,
            participants,
            date,
            duration,
            transcript: [],
        }, notes, audioBlob);
        
        if (success) {
            setSessionTitle('');
            setParticipants('');
            setNotes('');
            setAudioBlob(null);
            setDuration(0);
        }
        setIsSaving(false);
    };
    
    const formatTime = (seconds: number) => {
        const h = Math.floor(seconds / 3600).toString().padStart(2, '0');
        const m = Math.floor((seconds % 3600) / 60).toString().padStart(2, '0');
        const s = (seconds % 60).toString().padStart(2, '0');
        return h !== '00' ? `${h}:${m}:${s}` : `${m}:${s}`;
    };

    return (
        <div className="card new-session">
            <h3>New Session</h3>
            <form onSubmit={handleSubmit} autoComplete="off">
                <div className="form-grid">
                    <input
                        type="text"
                        name="session-title"
                        autoComplete="off"
                        placeholder="Session Title"
                        value={sessionTitle}
                        onChange={e => setSessionTitle(e.target.value)}
                        required
                    />
                    <input
                        type="text"
                        name="session-participants"
                        autoComplete="off"
                        placeholder="Participants (optional)"
                        value={participants}
                        onChange={e => setParticipants(e.target.value)}
                    />
                    <input
                        type="date"
                        name="session-date"
                        autoComplete="off"
                        value={date}
                        onChange={e => setDate(e.target.value)}
                        className="grid-col-span-2"
                    />
                    <textarea
                        placeholder="Enter notes or a summary here..."
                        value={notes}
                        onChange={e => setNotes(e.target.value)}
                        rows={5}
                        className="grid-col-span-2"
                    ></textarea>
                </div>
                 <div className="recording-options">
                    <label>Audio Source</label>
                    <div className="sliding-toggle-container" onClick={() => setRecordingType(prev => prev === 'mic' ? 'system' : 'mic')}>
                        <div className={`sliding-toggle-bg ${recordingType}`}></div>
                        <div className={`sliding-toggle-option ${recordingType === 'mic' ? 'active' : ''}`}>Microphone</div>
                        <div className={`sliding-toggle-option ${recordingType === 'system' ? 'active' : ''}`}>System Audio</div>
                    </div>
                </div>
                <div className="recording-controls">
                    {!isRecording ? (
                        <button type="button" className="btn-record" onClick={handleStartRecording}>
                            <AppIcon name="record" size={16} isDarkMode={isDarkMode} />
                            Record
                        </button>
                    ) : (
                        <button type="button" className="btn-record recording" onClick={handleStopRecording}>
                            <AppIcon name="stop" size={16} isDarkMode={isDarkMode} />
                            Stop ({formatTime(duration)})
                        </button>
                    )}
                </div>
                {audioBlob && <div className="status info">Audio recorded. Save the session to attach it.</div>}
                <button type="submit" className="btn-primary" disabled={isSaving || isRecording} style={{marginTop: '16px'}}>
                    {isSaving ? 'Saving...' : 'Save Session'}
                </button>
            </form>
        </div>
    );
};
