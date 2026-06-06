import React, { useState, useEffect } from 'react';

interface NotesEditorProps {
    notes: string;
    isDecrypting: boolean;
    onSave: (editedNotes: string) => void;
}

export const NotesEditor: React.FC<NotesEditorProps> = ({ notes, isDecrypting, onSave }) => {
    const [isEditingNotes, setIsEditingNotes] = useState(false);
    const [editedNotes, setEditedNotes] = useState('');

    useEffect(() => {
        setEditedNotes(notes);
    }, [notes]);

    const handleSave = () => {
        onSave(editedNotes);
        setIsEditingNotes(false);
    };

    return (
        <>
            <h3>Notes</h3>
            {isDecrypting ? (
                <div className="loading">Decrypting...</div>
            ) : (
                isEditingNotes ? (
                    <div>
                        <textarea
                            value={editedNotes}
                            onChange={e => setEditedNotes(e.target.value)}
                            rows={8}
                            style={{ width: '100%' }}
                        />
                        <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
                            <button className="btn-primary" onClick={handleSave}>Save</button>
                            <button className="btn-stop" onClick={() => setIsEditingNotes(false)}>Cancel</button>
                        </div>
                    </div>
                ) : (
                    <div>
                        <div className="transcript" style={{ whiteSpace: 'pre-wrap' }} onClick={() => setIsEditingNotes(true)}>
                            {notes || <span style={{color: '#94a3b8'}}>Click to add notes...</span>}
                        </div>
                    </div>
                )
            )}
        </>
    );
};
