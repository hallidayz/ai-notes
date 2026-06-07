
import React, { useState, useEffect } from 'react';
import { LocalModel, ModelConfig, StorageProvider } from '../types';
import { onDeviceAIService } from '../services/onDeviceAIService';
import { AppIcon } from './AppIcon';

interface LocalModelsProps {
    storageProvider: StorageProvider;
    onBack?: () => void;
    showStatus: (msg: string, type: 'success' | 'error' | 'info') => void;
    embedded?: boolean;
    isDarkMode?: boolean;
}

const AVAILABLE_MODELS: LocalModel[] = [
    // Transcription Models
    {
        id: 'whisper-tiny-en',
        name: 'Whisper Tiny (English)',
        parameters: '39M',
        provider: 'OpenAI',
        type: 'transcription',
        huggingFacePath: 'Xenova/whisper-tiny.en'
    },
    {
        id: 'whisper-base-en',
        name: 'Whisper Base (English)',
        parameters: '74M',
        provider: 'OpenAI',
        type: 'transcription',
        huggingFacePath: 'Xenova/whisper-base.en'
    },
    // Analysis Models (from screenshot & standard)
    {
        id: 'lfm2-350m',
        name: 'LFM2-350M',
        parameters: '350M',
        provider: 'LiquidAI',
        type: 'analysis',
        huggingFacePath: 'Xenova/flan-t5-small' // Placeholder if real path unknown
    },
    {
        id: 'lfm2-700m',
        name: 'LFM2-700M',
        parameters: '700M',
        provider: 'LiquidAI',
        type: 'analysis',
        huggingFacePath: 'Xenova/flan-t5-base' // Placeholder
    },
    {
        id: 'qwen3-0.6b',
        name: 'Qwen3-0.6B',
        parameters: '0.6B',
        provider: 'Qwen',
        type: 'analysis',
        huggingFacePath: 'Xenova/qwen-0.5b-instruct' // Placeholder
    },
    {
        id: 'flan-t5-small',
        name: 'Flan-T5 Small',
        parameters: '60M',
        provider: 'Google',
        type: 'analysis',
        huggingFacePath: 'Xenova/flan-t5-small'
    },
    {
        id: 'phi-1_5',
        name: 'Phi-1.5',
        parameters: '1.3B',
        provider: 'Microsoft',
        type: 'analysis',
        huggingFacePath: 'Xenova/phi-1_5'
    }
];

export const LocalModels: React.FC<LocalModelsProps> = ({ storageProvider, onBack, showStatus, embedded = false, isDarkMode = false }) => {
    const [config, setConfig] = useState<ModelConfig>({
        transcriptionModelId: 'whisper-tiny-en',
        analysisModelId: 'flan-t5-small'
    });
    const [downloadingId, setDownloadingId] = useState<string | null>(null);
    const [downloadProgress, setDownloadProgress] = useState(0);
    const [downloadedModels, setDownloadedModels] = useState<Set<string>>(new Set(['whisper-tiny-en', 'flan-t5-small']));

    useEffect(() => {
        const loadConfig = async () => {
            try {
                const savedConfig = await storageProvider.getConfig('model_config');
                if (savedConfig) {
                    setConfig(savedConfig);
                }
                
                const savedDownloaded = await storageProvider.getConfig('downloaded_models');
                if (savedDownloaded) {
                    setDownloadedModels(new Set(savedDownloaded));
                }
            } catch (e) {
                console.error("Failed to load model config", e);
            }
        };
        loadConfig();
    }, [storageProvider]);

    const saveConfig = async (newConfig: ModelConfig) => {
        setConfig(newConfig);
        try {
            await storageProvider.saveConfig('model_config', newConfig);
            onDeviceAIService.updateConfig(newConfig);
        } catch (e) {
            console.error("Failed to save model config", e);
        }
    };

    const handleDownload = async (model: LocalModel) => {
        setDownloadingId(model.id);
        setDownloadProgress(0);
        
        try {
            await onDeviceAIService.preloadModel(model.huggingFacePath, model.type, (p) => {
                if (p.status === 'progress') {
                    setDownloadProgress(p.progress || 0);
                }
            });
            
            const newDownloaded = new Set(downloadedModels);
            newDownloaded.add(model.id);
            setDownloadedModels(newDownloaded);
            
            await storageProvider.saveConfig('downloaded_models', Array.from(newDownloaded));
            
            showStatus(`${model.name} downloaded successfully.`, 'success');
        } catch (e) {
            console.error("Download failed", e);
            showStatus(`Failed to download ${model.name}.`, 'error');
        } finally {
            setDownloadingId(null);
        }
    };

    const handleSelect = (model: LocalModel) => {
        if (!downloadedModels.has(model.id)) {
            handleDownload(model);
            return;
        }

        if (model.type === 'transcription') {
            saveConfig({ ...config, transcriptionModelId: model.id });
        } else {
            saveConfig({ ...config, analysisModelId: model.id });
        }
    };

    const renderModelCard = (model: LocalModel) => {
        const isSelected = config.transcriptionModelId === model.id || config.analysisModelId === model.id;
        const isDownloaded = downloadedModels.has(model.id);
        const isDownloading = downloadingId === model.id;

        return (
            <div 
                key={model.id}
                className={`local-model-card ${isSelected ? 'selected' : ''}`}
                onClick={() => handleSelect(model)}
            >
                <div className="local-model-radio">
                    {isSelected ? (
                        <div className="local-model-radio-selected">
                            <AppIcon name="check" size={14} isDarkMode={isDarkMode} />
                        </div>
                    ) : (
                        <div className="local-model-radio-empty" />
                    )}
                </div>
                
                <div className="local-model-info">
                    <div className="local-model-name">
                        <h4>{model.name}</h4>
                        {isDownloaded && !isSelected && (
                            <AppIcon name="check" size={12} isDarkMode={isDarkMode} className="local-model-check" />
                        )}
                    </div>
                    <p className="local-model-meta">
                        {model.parameters} parameters • {model.provider}
                    </p>
                    {isDownloaded && (
                        <div className="local-model-downloaded">
                            <AppIcon name="check" size={12} isDarkMode={isDarkMode} className="local-model-check" />
                            <span>Downloaded</span>
                        </div>
                    )}
                </div>

                <div className="local-model-action">
                    {isDownloading ? (
                        <div className="local-model-progress">
                            <AppIcon name="loader" size={20} isDarkMode={isDarkMode} className="app-icon-spin" />
                            <span>{Math.round(downloadProgress)}%</span>
                        </div>
                    ) : !isDownloaded ? (
                        <AppIcon name="download" size={20} isDarkMode={isDarkMode} className="local-model-download-icon" />
                    ) : null}
                </div>
            </div>
        );
    };

    const content = (
        <>
            <div className="local-models-info">
                <AppIcon name="info" size={18} isDarkMode={isDarkMode} className="local-models-info-icon" />
                <p>
                    These models run entirely on your device. Downloading them may take a few moments depending on your connection. Once downloaded, they work fully offline.
                </p>
            </div>

            <div className="local-models-group-label">Transcription Models</div>
            {AVAILABLE_MODELS.filter(m => m.type === 'transcription').map(renderModelCard)}

            <div className="local-models-group-label">Analysis Models</div>
            {AVAILABLE_MODELS.filter(m => m.type === 'analysis').map(renderModelCard)}
        </>
    );

    if (embedded) {
        return <div className="local-models-embedded">{content}</div>;
    }

    return (
        <div className="local-models-page">
            <div className="local-models-header">
                <button onClick={onBack} className="local-models-back" aria-label="Go back">
                    <AppIcon name="chevron-left" size={24} isDarkMode={isDarkMode} />
                </button>
                <h2>Local models</h2>
            </div>
            <div className="local-models-body">{content}</div>
        </div>
    );
};
