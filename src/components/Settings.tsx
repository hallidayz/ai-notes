
import React from 'react';
import { X } from 'lucide-react';
import { StorageProvider } from '../types';
import { LocalModels } from './LocalModels';

interface SettingsProps {
    isOpen: boolean;
    onClose: () => void;
    industry: string;
    onIndustryChange: (industry: string) => void;
    storageProvider: StorageProvider;
    showStatus: (msg: string, type: 'success' | 'error' | 'info') => void;
}

export const Settings: React.FC<SettingsProps> = ({
    isOpen,
    onClose,
    industry,
    onIndustryChange,
    storageProvider,
    showStatus,
}) => {
    if (!isOpen) return null;

    return (
        <div className="modal active" onClick={onClose}>
            <div className="modal-content settings-modal" onClick={e => e.stopPropagation()}>
                <button className="close-btn" onClick={onClose} aria-label="Close settings">
                    <X size={18} />
                </button>

                <h2>Settings</h2>

                <section className="settings-section">
                    <h3>Industry Context</h3>
                    <p className="settings-section-desc">
                        Tailors AI analysis and summaries to your professional domain.
                    </p>
                    <label htmlFor="industrySelector" className="settings-label">Industry</label>
                    <select
                        id="industrySelector"
                        className="settings-select"
                        value={industry}
                        onChange={(e) => onIndustryChange(e.target.value)}
                    >
                        <option value="General">General</option>
                        <option value="Medical">Medical</option>
                        <option value="Legal">Legal</option>
                        <option value="Therapy">Therapy</option>
                        <option value="Business">Business/Meetings</option>
                        <option value="Education">Education</option>
                    </select>
                </section>

                <section className="settings-section">
                    <h3>Local Models</h3>
                    <p className="settings-section-desc">
                        Choose and download on-device models for transcription and analysis.
                    </p>
                    <div className="settings-local-models">
                        <LocalModels
                            storageProvider={storageProvider}
                            showStatus={showStatus}
                            embedded
                        />
                    </div>
                </section>

                <section className="settings-section">
                    <h3>Registration</h3>
                    <p className="settings-section-desc">
                        Account registration and device sync will be available in a future update.
                    </p>
                    <div className="settings-coming-soon">
                        Coming soon
                    </div>
                </section>
            </div>
        </div>
    );
};
