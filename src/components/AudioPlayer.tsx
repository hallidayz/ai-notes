
import React, { useState, useEffect, useRef } from 'react';

interface AudioPlayerProps {
    audioUrl: string;
}

export const AudioPlayer: React.FC<AudioPlayerProps> = ({ audioUrl }) => {
    const audioRef = useRef<HTMLAudioElement>(null);
    const progressRef = useRef<HTMLDivElement>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);

    useEffect(() => {
        const audio = audioRef.current;
        if (!audio) return;

        const handleTimeUpdate = () => setCurrentTime(audio.currentTime);
        const handleDurationChange = () => setDuration(audio.duration);
        const handleEnded = () => setIsPlaying(false);

        audio.addEventListener('timeupdate', handleTimeUpdate);
        audio.addEventListener('durationchange', handleDurationChange);
        audio.addEventListener('ended', handleEnded);

        return () => {
            audio.removeEventListener('timeupdate', handleTimeUpdate);
            audio.removeEventListener('durationchange', handleDurationChange);
            audio.removeEventListener('ended', handleEnded);
        };
    }, []);
    
    const togglePlayPause = () => {
        if (audioRef.current) {
            if (isPlaying) {
                audioRef.current.pause();
            } else {
                audioRef.current.play();
            }
            setIsPlaying(!isPlaying);
        }
    };

    const handleProgressClick = (e: React.MouseEvent<HTMLDivElement>) => {
        if (audioRef.current && progressRef.current) {
            const rect = progressRef.current.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const width = rect.width;
            const newTime = (clickX / width) * duration;
            audioRef.current.currentTime = newTime;
        }
    };
    
    const formatTime = (time: number) => {
        if (isNaN(time) || time === 0) return '00:00';
        const minutes = Math.floor(time / 60);
        const seconds = Math.floor(time % 60);
        return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    };

    const progressPercentage = duration > 0 ? (currentTime / duration) * 100 : 0;

    return (
        <div className="player-controls">
            <audio ref={audioRef} src={audioUrl} preload="metadata"></audio>
            <div className="audio-player">
                <button onClick={togglePlayPause} className="playback-btn">
                    {isPlaying ? '❚❚' : '►'}
                </button>
                <span className="time-display">{formatTime(currentTime)}</span>
                <div className="progress-container" ref={progressRef} onClick={handleProgressClick}>
                    <div className="progress-bar">
                        <div className="progress-fill" style={{ width: `${progressPercentage}%` }}></div>
                    </div>
                </div>
                <span className="time-display">{formatTime(duration)}</span>
            </div>
        </div>
    );
};
