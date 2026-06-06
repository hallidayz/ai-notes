import React, { useState, useMemo } from 'react';
import { TranscriptChunk } from '../types';

interface TranscriptViewProps {
    transcript: TranscriptChunk[];
    onUpdateTranscript: (newTranscript: TranscriptChunk[]) => void;
}

export const TranscriptView: React.FC<TranscriptViewProps> = ({ transcript, onUpdateTranscript }) => {
    const [speakerOverrides, setSpeakerOverrides] = useState<Record<string, string>>({});
    const [editingSpeaker, setEditingSpeaker] = useState<{chunkIndex: number, oldName: string} | null>(null);

    const uniqueSpeakers = useMemo(() => {
        const speakers: string[] = [];
        transcript.forEach((c) => {
            if (!speakers.includes(c.speaker)) {
                speakers.push(c.speaker);
            }
        });
        return speakers;
    }, [transcript]);

    const speakerMap = useMemo(() => {
        const map: Record<string, string> = {};
        uniqueSpeakers.forEach(speaker => {
            map[speaker] = speakerOverrides[speaker] ?? speaker;
        });
        return map;
    }, [uniqueSpeakers, speakerOverrides]);

    const handleSpeakerNameChange = (newName: string) => {
        if (!editingSpeaker) return;

        const { oldName } = editingSpeaker;
        setSpeakerOverrides(prev => ({ ...prev, [oldName]: newName }));

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
