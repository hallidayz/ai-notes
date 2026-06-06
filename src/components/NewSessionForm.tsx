
import React, { useState, useRef } from 'react';
import { Session } from '../types';

interface NewSessionFormProps {
    onAddSession: (session: Omit<Session, 'id' | 'timestamp' | 'notes'>, notes: string, audioBlob: Blob | null) => Promise<boolean>;
    showStatus: (message: string, type: 'success' | 'error' | 'info', duration?: number) => void;
}

export const NewSessionForm: React.FC<NewSessionFormProps> = ({ onAddSession, showStatus }) => {
    const [sessionTitle, setSessionTitle] = useState('');
    const [participants, setParticipants] = useState('');
    const [date, setDate] = useState(new Date().toISOString().split('T')[0]);
    const [notes, setNotes] = useState('');
    const [isRecording, setIsRecording] = useState(false);
    const [isSaving, setIsSaving] = useState(false);
    const [recordingType, setRecordingType] = useState<'mic' | 'system'>('system');
    const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
    const [duration, setDuration] = useState(0);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const chunksRef = useRef<Blob[]>([]);
    const timerRef = useRef<number | null>(null);

    const handleStartRecording = async () => {
        try {
            let stream: MediaStream;
            if (recordingType === 'mic') {
                stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            } else {
                stream = await navigator.mediaDevices.getDisplayMedia({
                    video: true,
                    audio: true
                });
                if (!stream.getAudioTracks().length) {
                    showStatus("Your system audio is not being shared. Please check the 'Share system audio' box in the prompt.", 'error', 5000);
                    return;
                }
            }
            
            setIsRecording(true);
            setDuration(0);
            timerRef.current = window.setInterval(() => setDuration(prev => prev + 1), 1000);

            mediaRecorderRef.current = new MediaRecorder(stream);
            mediaRecorderRef.current.ondataavailable = (e) => {
                if (e.data.size > 0) chunksRef.current.push(e.data);
            };
            mediaRecorderRef.current.onstop = () => {
                const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
                setAudioBlob(blob);
                chunksRef.current = [];
                 stream.getTracks().forEach(track => track.stop());
            };
            mediaRecorderRef.current.start();
        } catch (err) {
            console.error("Error starting recording:", err);
            showStatus("Could not start recording. Please ensure you have given microphone permissions and selected a source.", 'error');
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
            <form onSubmit={handleSubmit}>
                <div className="form-grid">
                    <input
                        type="text"
                        placeholder="Session Title"
                        value={sessionTitle}
                        onChange={e => setSessionTitle(e.target.value)}
                        required
                    />
                    <input
                        type="text"
                        placeholder="Participants (optional)"
                        value={participants}
                        onChange={e => setParticipants(e.target.value)}
                    />
                    <input
                        type="date"
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
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M8 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zm0 1A5 5 0 1 1 8 3a5 5 0 0 1 0 10z"/><path d="M10 8a2 2 0 1 1-4 0 2 2 0 0 1 4 0z"/></svg>
                            Record
                        </button>
                    ) : (
                        <button type="button" className="btn-record recording" onClick={handleStopRecording}>
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M5.5 5.5A.5.5 0 0 1 6 6v4a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm4 0a.5.5 0 0 1 .5.5v4a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5z"/><path d="M0 2a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V2zm15 0a1 1 0 0 0-1-1H2a1 1 0 0 0-1 1v12a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V2z"/></svg>
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
