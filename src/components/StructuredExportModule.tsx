import React, { useState } from 'react';

interface Node {
    id: string;
    label: string;
    type: string; // 'start' | 'process' | 'decision' | 'action' | 'end'
}

interface Edge {
    from: string;
    to: string;
    label?: string;
}

interface MermaidData {
    mermaidCode: string;
    nodes: Node[];
    edges: Edge[];
}

interface PRDData {
    prdMarkdown: string;
    projectTitle: string;
    keyMetrics: string[];
}

interface StructuredExportProps {
    decryptedNotes: string;
    decryptedSummary: string;
    decryptedOutline: string;
    showStatus: (msg: string, type: 'success' | 'error' | 'info') => void;
}

export const StructuredExportModule: React.FC<StructuredExportProps> = ({
    decryptedNotes,
    decryptedSummary,
    decryptedOutline,
    showStatus
}) => {
    const [exportType, setExportType] = useState<'mermaid' | 'prd'>('mermaid');
    const [loading, setLoading] = useState(false);
    const [customInstruction, setCustomInstruction] = useState('');
    const [mermaidData, setMermaidData] = useState<MermaidData | null>(null);
    const [prdData, setPrdData] = useState<PRDData | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [selectedFlowNode, setSelectedFlowNode] = useState<string | null>(null);

    const handleGenerate = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch('/api/ai/structured-export', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    notes: decryptedNotes,
                    summary: decryptedSummary,
                    outline: decryptedOutline,
                    format: exportType,
                    customInstruction: customInstruction.trim()
                })
            });

            if (!response.ok) {
                const errJson = await response.json().catch(() => ({}));
                throw new Error(errJson.error || 'Server error generating exporting data');
            }

            const data = await response.json();
            if (exportType === 'mermaid') {
                setMermaidData(data);
                if (data.nodes && data.nodes.length > 0) {
                    setSelectedFlowNode(data.nodes[0].id);
                }
                showStatus('Flowchart successfully generated!', 'success');
            } else {
                setPrdData(data);
                showStatus('PRD Document successfully generated!', 'success');
            }
        } catch (err) {
            const errorObj = err as Error;
            console.error(errorObj);
            setError(errorObj.message || 'Failed to connect to the AI service');
            showStatus(errorObj.message || 'Failed to export', 'error');
        } finally {
            setLoading(false);
        }
    };

    const copyToClipboard = (text: string) => {
        navigator.clipboard.writeText(text);
        showStatus('Copied to clipboard!', 'success');
    };

    const downloadMarkdownFile = (filename: string, content: string) => {
        const element = document.createElement("a");
        const file = new Blob([content], { type: 'text/markdown;charset=utf-8' });
        element.href = URL.createObjectURL(file);
        element.download = filename;
        document.body.appendChild(element);
        element.click();
        document.body.removeChild(element);
        showStatus('Downloaded file successfully!', 'success');
    };

    // Style helper for nodes based on their semantic roles
    const getNodeStyles = (type: string, isSelected: boolean) => {
        let baseClass = "p-3 rounded-lg border text-center transition-all cursor-pointer shadow-sm ";
        if (isSelected) {
            baseClass += "scale-105 ring-2 ring-blue-500 shadow-md ";
        }

        switch (type) {
            case 'start':
                return baseClass + "bg-emerald-500/10 border-emerald-500 text-emerald-700 dark:text-emerald-400 font-medium rounded-full";
            case 'end':
                return baseClass + "bg-rose-500/10 border-rose-500 text-rose-700 dark:text-rose-400 font-medium rounded-full";
            case 'decision':
                return baseClass + "bg-amber-500/15 border-amber-500 text-amber-800 dark:text-amber-300 font-semibold rotate-1 animate-pulse-subtle";
            case 'action':
                return baseClass + "bg-blue-500/10 border-blue-400 text-blue-700 dark:text-blue-300 font-medium";
            case 'process':
            default:
                return baseClass + "bg-neutral-500/5 border-neutral-300 dark:border-neutral-700 text-neutral-800 dark:text-neutral-200";
        }
    };

    // Find incoming & outgoing edges for a focused node
    const getConnectionsForNode = (nodeId: string) => {
        if (!mermaidData) return { incoming: [], outgoing: [] };
        const incoming = mermaidData.edges.filter(e => e.to === nodeId);
        const outgoing = mermaidData.edges.filter(e => e.from === nodeId);
        return { incoming, outgoing };
    };

    const activeNodeDetails = selectedFlowNode && mermaidData 
        ? mermaidData.nodes.find(n => n.id === selectedFlowNode) 
        : null;

    const activeConnections = selectedFlowNode 
        ? getConnectionsForNode(selectedFlowNode) 
        : { incoming: [], outgoing: [] };

    return (
        <div id="structured-export-module" className="card" style={{ marginTop: '24px', padding: '24px', border: '1px solid var(--card-border)' }}>
            <div className="flex justify-between items-center" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
                <div>
                    <h3 style={{ margin: 0 }}>🪄 Professional Structured Export</h3>
                    <p style={{ color: '#94a3b8', fontSize: '0.9em', marginTop: '4px' }}>
                        Convert this session directly into industry-standard Mermaid flowcharts or a Product Requirements Document (PRD).
                    </p>
                </div>
                
                {/* Format Toggle Tabs */}
                <div style={{ display: 'flex', background: 'var(--input-bg)', borderRadius: '8px', padding: '4px', border: '1px solid var(--input-border)' }}>
                    <button 
                        onClick={() => setExportType('mermaid')}
                        className={`tab-btn ${exportType === 'mermaid' ? 'active' : ''}`}
                        style={{
                            padding: '8px 16px',
                            border: 'none',
                            borderRadius: '6px',
                            background: exportType === 'mermaid' ? 'var(--accent-color)' : 'transparent',
                            color: exportType === 'mermaid' ? '#fff' : 'var(--text-secondary)',
                            cursor: 'pointer',
                            fontSize: '0.9em',
                            fontWeight: 500,
                            transition: 'all 0.2s'
                        }}
                    >
                        📊 Process Flowchart
                    </button>
                    <button 
                        onClick={() => setExportType('prd')}
                        className={`tab-btn ${exportType === 'prd' ? 'active' : ''}`}
                        style={{
                            padding: '8px 16px',
                            border: 'none',
                            borderRadius: '6px',
                            background: exportType === 'prd' ? 'var(--accent-color)' : 'transparent',
                            color: exportType === 'prd' ? '#fff' : 'var(--text-secondary)',
                            cursor: 'pointer',
                            fontSize: '0.9em',
                            fontWeight: 500,
                            transition: 'all 0.2s'
                        }}
                    >
                        📄 Standard PRD
                    </button>
                </div>
            </div>

            {/* Instruction Customization Input */}
            <div style={{ marginBottom: '20px' }}>
                <label style={{ fontSize: '0.9em', fontWeight: '500', color: 'var(--text-secondary)', display: 'block', marginBottom: '8px' }}>
                    💡 Customize Output Focus (Optional)
                </label>
                <input 
                    type="text" 
                    value={customInstruction}
                    onChange={(e) => setCustomInstruction(e.target.value)}
                    placeholder={
                        exportType === 'mermaid' 
                          ? "E.g., Focus on backend state machine, make it process layout, highlight approval loops..."
                          : "E.g., Tailored for mobile React Native build, clarify database structure, B2B SaaS architecture..."
                    }
                    style={{
                        width: '100%',
                        padding: '12px',
                        borderRadius: '8px',
                        border: '1px solid var(--input-border)',
                        background: 'var(--input-bg)',
                        color: 'var(--text-primary)',
                        outline: 'none',
                        fontSize: '0.92em'
                    }}
                />
            </div>

            <button 
                onClick={handleGenerate}
                disabled={loading}
                className="btn-primary"
                style={{
                    width: '100%',
                    padding: '14px',
                    borderRadius: '8px',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    fontWeight: '600',
                    fontSize: '1em',
                    boxShadow: '0 4px 12px rgba(55, 64, 255, 0.15)',
                    cursor: 'pointer'
                }}
            >
                {loading ? (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <div className="spinner-small" style={{ width: '18px', height: '18px', borderSize: '2.5px' }}></div>
                        <span>Structuring Content using Gemini AI...</span>
                    </div>
                ) : (
                    <span>⚡ Generate Professional {exportType === 'mermaid' ? 'Flowchart' : 'PRD Document'}</span>
                )}
            </button>

            {error && (
                <div style={{ marginTop: '20px', padding: '16px', background: 'var(--danger-bg-light)', border: '1px solid #f87171', borderRadius: '8px', color: 'var(--danger-text-light)', fontSize: '0.9em' }}>
                    ⚠️ {error}
                </div>
            )}

            {/* MERMAID OUTPUT ZONE */}
            {exportType === 'mermaid' && mermaidData && (
                <div style={{ marginTop: '28px' }}>
                    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '20px', marginTop: '16px' }} className="grid-responsive">
                        
                        {/* Interactive Visual Flow Explorer */}
                        <div style={{ background: 'var(--input-bg)', border: '1px solid var(--input-border)', borderRadius: '12px', padding: '20px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                                <h4 style={{ margin: 0, fontSize: '1.05em' }}>🎯 Visual Flow Navigator</h4>
                                <span style={{ fontSize: '0.75em', background: 'var(--accent-color)', color: '#fff', padding: '2px 8px', borderRadius: '20px' }}>Interactive</span>
                            </div>
                            <p style={{ fontSize: '0.82em', color: 'var(--text-tertiary)', marginBottom: '16px' }}>
                                Select any system node inside the flow path below to walk through logical transitions and details!
                            </p>

                            {/* Node Array Explorer Card Grid */}
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))', gap: '10px', maxHeight: '280px', overflowY: 'auto', padding: '4px' }}>
                                {mermaidData.nodes.map(node => (
                                    <div 
                                        key={node.id} 
                                        onClick={() => setSelectedFlowNode(node.id)}
                                        className={getNodeStyles(node.type, selectedFlowNode === node.id)}
                                        style={{
                                            borderWidth: '1px',
                                            borderRadius: '8px',
                                            padding: '8px 12px',
                                            cursor: 'pointer',
                                            fontSize: '0.85em',
                                            textAlign: 'center',
                                            transition: 'all 0.15s'
                                        }}
                                    >
                                        <div style={{ fontSize: '0.7em', textTransform: 'uppercase', opacity: 0.6, marginBottom: '2px' }}>
                                            {node.type}
                                        </div>
                                        <div style={{ fontWeight: '600', lineHeight: '1.2' }}>{node.label}</div>
                                    </div>
                                ))}
                            </div>

                            {/* Connected Nodes Context Card */}
                            {activeNodeDetails && (
                                <div style={{ marginTop: '20px', background: 'var(--card-bg)', border: '1px solid var(--card-border)', borderRadius: '8px', padding: '14px' }}>
                                    <div style={{ display: 'flex', gap: '6px', alignItems: 'center', marginBottom: '8px' }}>
                                        <span style={{ fontSize: '0.7em', textTransform: 'uppercase', background: 'rgba(55, 64, 255, 0.1)', color: 'var(--accent-color)', padding: '2px 6px', borderRadius: '4px', fontWeight: 'bold' }}>
                                            {activeNodeDetails.type}
                                        </span>
                                        <span style={{ fontWeight: '700', fontSize: '0.9em', color: 'var(--text-primary)' }}>
                                            [ Node {activeNodeDetails.id} ] - {activeNodeDetails.label}
                                        </span>
                                    </div>

                                    {/* Incoming Triggers */}
                                    <div style={{ fontSize: '0.85em', marginTop: '10px' }}>
                                        <div style={{ color: 'var(--text-tertiary)', fontWeight: '500', marginBottom: '4px' }}>🔀 Input Connectors:</div>
                                        {activeConnections.incoming.length === 0 ? (
                                            <div style={{ fontStyle: 'italic', color: 'var(--text-tertiary)', fontSize: '0.9em', paddingLeft: '8px' }}>None (Start Node)</div>
                                        ) : (
                                            activeConnections.incoming.map((edge, i) => {
                                                const fromNode = mermaidData.nodes.find(n => n.id === edge.from);
                                                return (
                                                    <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '6px', margin: '3px 0', paddingLeft: '8px' }}>
                                                        <span style={{ fontWeight: 600 }}>{fromNode?.label || edge.from}</span>
                                                        <span style={{ color: 'var(--text-tertiary)' }}>➔</span>
                                                        {edge.label && <span style={{ background: 'var(--input-bg)', border: '1px solid var(--input-border)', color: 'var(--text-secondary)', padding: '1px 4px', borderRadius: '4px', fontSize: '0.8em' }}>{edge.label}</span>}
                                                    </div>
                                                );
                                            })
                                        )}
                                    </div>

                                    {/* Outgoing Paths */}
                                    <div style={{ fontSize: '0.85em', marginTop: '10px' }}>
                                        <div style={{ color: 'var(--text-tertiary)', fontWeight: '500', marginBottom: '4px' }}>⚙️ Outbound Directions:</div>
                                        {activeConnections.outgoing.length === 0 ? (
                                            <div style={{ fontStyle: 'italic', color: 'var(--text-tertiary)', fontSize: '0.9em', paddingLeft: '8px' }}>None (Terminal Endpoint)</div>
                                        ) : (
                                            activeConnections.outgoing.map((edge, i) => {
                                                const toNode = mermaidData.nodes.find(n => n.id === edge.to);
                                                return (
                                                    <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '6px', margin: '3px 0', paddingLeft: '8px' }}>
                                                        {edge.label && <span style={{ background: 'var(--input-bg)', border: '1px solid var(--input-border)', color: 'var(--text-secondary)', padding: '1px 4px', borderRadius: '4px', fontSize: '0.8em' }}>{edge.label}</span>}
                                                        <span style={{ color: 'var(--text-tertiary)' }}>➔</span>
                                                        <span style={{ fontWeight: 600 }}>{toNode?.label || edge.to}</span>
                                                    </div>
                                                );
                                            })
                                        )}
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Raw Clipboard Markdown Block */}
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                                <label style={{ fontSize: '0.95em', fontWeight: '600', color: 'var(--text-secondary)' }}>
                                    🔀 Mermaid.js Syntax Code
                                </label>
                                <button 
                                    onClick={() => copyToClipboard(mermaidData.mermaidCode)}
                                    className="btn-secondary"
                                    style={{ padding: '4px 10px', fontSize: '0.8em', borderRadius: '6px' }}
                                >
                                    📋 Copy Code
                                </button>
                            </div>
                            <textarea 
                                readOnly 
                                value={mermaidData.mermaidCode}
                                className="transcript"
                                style={{
                                    flex: 1,
                                    width: '100%',
                                    minHeight: '260px',
                                    fontFamily: 'var(--font-mono)',
                                    fontSize: '0.85em',
                                    padding: '12px',
                                    borderRadius: '10px',
                                    border: '1px solid var(--input-border)',
                                    background: 'var(--input-bg)',
                                    color: 'var(--text-primary)',
                                    whiteSpace: 'pre',
                                    outline: 'none',
                                    resize: 'none'
                                }}
                            />
                            <div style={{ marginTop: '8px', fontSize: '0.78em', color: 'var(--text-tertiary)', lineHeight: '1.4' }}>
                                💡 Perfect for direct copy-pasting into **Notion**, **GitHub Markdown**, **Obsidian**, or **Mermaid Live Editor** pages to instantly compile fully scalable SVG flow diagram trees.
                            </div>
                        </div>

                    </div>
                </div>
            )}

            {/* PRD DOCUMENT OUTPUT ZONE */}
            {exportType === 'prd' && prdData && (
                <div style={{ marginTop: '28px', borderTop: '1px solid var(--card-border)', paddingTop: '24px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px', marginBottom: '16px' }}>
                        <div>
                            <span style={{ fontSize: '0.75em', background: 'var(--accent-color)', color: '#fff', padding: '3px 8px', borderRadius: '4px', fontWeight: 'bold', textTransform: 'uppercase' }}>
                                Product Requirements Document
                            </span>
                            <h4 style={{ margin: '6px 0 0 0', fontSize: '1.25em', color: 'var(--text-primary)' }}>
                                {prdData.projectTitle}
                            </h4>
                        </div>
                        <div style={{ display: 'flex', gap: '8px' }}>
                            <button 
                                onClick={() => copyToClipboard(prdData.prdMarkdown)}
                                className="btn-secondary"
                                style={{ padding: '8px 14px', fontSize: '0.85em', borderRadius: '8px' }}
                            >
                                📋 Copy Markdown
                            </button>
                            <button 
                                onClick={() => downloadMarkdownFile(`${prdData.projectTitle.toLowerCase().replace(/\s+/g, '_')}_prd.md`, prdData.prdMarkdown)}
                                className="btn-primary"
                                style={{ padding: '8px 14px', fontSize: '0.85em', borderRadius: '8px', background: '#10b981', hoverBg: '#059669' }}
                            >
                                💾 Download PRD (.md)
                            </button>
                        </div>
                    </div>

                    {/* Quick KPIs / Metrics Row */}
                    <div style={{ marginBottom: '20px', background: 'var(--input-bg)', border: '1px solid var(--input-border)', borderRadius: '10px', padding: '14px' }}>
                        <div style={{ fontSize: '0.85em', fontWeight: 'bold', color: 'var(--text-secondary)', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <span>📈 High-Impact KPIs & Success Metrics Identified:</span>
                        </div>
                        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                            {prdData.keyMetrics.map((metric, i) => (
                                <span 
                                    key={i} 
                                    style={{
                                        fontSize: '0.82em',
                                        background: 'rgba(16, 185, 129, 0.1)',
                                        border: '1px solid rgba(16, 185, 129, 0.3)',
                                        color: '#10b981',
                                        padding: '4px 10px',
                                        borderRadius: '6px',
                                        fontWeight: '500'
                                    }}
                                >
                                    🎯 {metric}
                                </span>
                            ))}
                        </div>
                    </div>

                    {/* Beautiful PRD Render Container */}
                    <div 
                        className="transcript"
                        style={{
                            width: '100%',
                            maxHeight: '450px',
                            overflowY: 'auto',
                            padding: '20px',
                            background: 'var(--input-bg)',
                            border: '1px solid var(--input-border)',
                            borderRadius: '12px',
                            fontFamily: 'inherit',
                            fontSize: '0.92em',
                            color: 'var(--text-primary)',
                            whiteSpace: 'pre-wrap',
                            lineHeight: '1.6'
                        }}
                    >
                        {prdData.prdMarkdown}
                    </div>
                </div>
            )}
        </div>
    );
};
