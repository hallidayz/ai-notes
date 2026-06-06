import React, { useState, useEffect } from 'react';
import { TranscriptChunk } from '../types';

interface TranscriptViewProps {
    transcript: TranscriptChunk[];
    onUpdateTranscript: (newTranscript: TranscriptChunk[]) => void;
}

export const TranscriptView: React.FC<TranscriptViewProps> = ({ transcript, onUpdateTranscript }) => {
    const [speakerMap, setSpeakerMap] = useState<{[key: string]: string}>({});
    const [editingSpeaker, setEditingSpeaker] = useState<{chunkIndex: number, oldName: string} | null>(null);

    useEffect(() => {
        const uniqueSpeakers: string[] = [];
        transcript.forEach((c) => {
            if (!uniqueSpeakers.includes(c.speaker)) {
                uniqueSpeakers.push(c.speaker);
            }
        });
        const initialMap: {[key: string]: string} = {};
        uniqueSpeakers.forEach(speaker => {
            initialMap[speaker] = speaker;
        });
        setSpeakerMap(initialMap);
    }, [transcript]);

    const handleSpeakerNameChange = (newName: string) => {
        if (!editingSpeaker) return;

        const { oldName } = editingSpeaker;
        const newMap = { ...speakerMap, [oldName]: newName };
        setSpeakerMap(newMap);

        const newTranscript = transcript.map(chunk => {
            if (chunk.speaker === oldName) {
                return { ...chunk, speaker: newName };
            }
            return chunk;
        });

        onUpdateTranscript(newTranscript);
        setEditingSpeaker(null);
    };

    const getSpeakerClass = (speaker: string) => {
        const speakers = Object.keys(speakerMap);
        const index = speakers.indexOf(speaker);
        return `speaker-style-${(index % 5) + 1}`;
    };

    if (!transcript || transcript.length === 0) {
        return null;
    }

    return (
        <>
            <h3>Transcript</h3>
            <div className="transcript">
                {transcript.map((chunk, index) => (
                    <div key={index} className={`transcript-chunk ${getSpeakerClass(chunk.speaker)}`}>
                        {editingSpeaker?.chunkIndex === index ? (
                            <input
                                type="text"
                                defaultValue={editingSpeaker.oldName}
                                onBlur={(e) => handleSpeakerNameChange(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && handleSpeakerNameChange(e.currentTarget.value)}
                                autoFocus
                                className="speaker-input"
                            />
                        ) : (
                            <span
                                className="speaker-label editable"
                                onClick={() => setEditingSpeaker({ chunkIndex: index, oldName: chunk.speaker })}
                            >
                                {speakerMap[chunk.speaker] || chunk.speaker}:
                            </span>
                        )}
                        <p>{chunk.text}</p>
                    </div>
                ))}
            </div>
        </>
    );
};
