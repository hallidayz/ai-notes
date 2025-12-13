# Feature Roadmap: Advanced Meeting Notes Features

## Overview
This document outlines the implementation plan for advanced features to transform MiNDS Talk into a comprehensive, enterprise-grade meeting notes solution.

---

## 1. Advanced Speaker Identification & Voice Learning

### Current State
- ✅ Basic heuristic-based speaker diarization (silence gaps, timing patterns)
- ✅ Sequential speaker assignment (Speaker 1, Speaker 2, etc.)
- ✅ Manual speaker name editing in UI

### Target Features
- **Voice Learning System**: After 2-3 meetings, automatically tag speakers by name
- **Voice Embeddings**: Store voice characteristics per speaker
- **Speaker Matching**: Match new audio segments to known speakers
- **Clear Attribution**: "John said..." instead of "Speaker 1 said..."

### Implementation Plan

#### Phase 1: Voice Embedding Extraction
```typescript
// New service: VoiceLearningService
class VoiceLearningService {
    // Extract voice embeddings from audio segments
    async extractVoiceEmbedding(audioBuffer: AudioBuffer, segment: {start: number, end: number}): Promise<Float32Array>
    
    // Store speaker voice profile
    async saveSpeakerProfile(speakerId: string, name: string, embeddings: Float32Array[]): Promise<void>
    
    // Match audio to known speakers
    async matchSpeaker(audioBuffer: AudioBuffer, segment: {start: number, end: number}): Promise<{speakerId: string, confidence: number} | null>
}
```

**Technical Approach:**
- Use `@xenova/transformers` with a speaker verification model (e.g., `speechbrain/spkrec-ecapa-voxceleb`)
- Extract embeddings from audio segments where speaker is identified
- Store embeddings in IndexedDB with speaker metadata
- Use cosine similarity to match new segments to known speakers

#### Phase 2: Speaker Learning Workflow
1. **First Meeting**: Assign sequential speakers (Speaker 1, 2, 3...)
2. **User Tags Speakers**: User manually names speakers in UI
3. **Voice Profiles Created**: System extracts and stores voice embeddings for each named speaker
4. **Subsequent Meetings**: 
   - Extract embeddings from new audio
   - Match to known speakers with confidence threshold (>0.7)
   - Auto-tag: "John said..." if match found
   - Fallback to "Speaker X" if no match

#### Phase 3: UI Enhancements
- Speaker profile management page
- "Learn Voice" button for manual speaker tagging
- Confidence indicators for auto-tagged speakers
- Ability to correct misidentified speakers (teaches system)

### Database Schema Updates
```typescript
interface SpeakerProfile {
    id: string;
    name: string;
    voiceEmbeddings: Float32Array[]; // Array of embeddings from different segments
    createdAt: Date;
    lastSeen: Date;
    meetingCount: number;
    confidenceThreshold: number; // Per-speaker threshold
}
```

---

## 2. Automatic Topic Labeling

### Current State
- ✅ Topic clustering in outline generation
- ✅ Basic outline structure

### Target Features
- **Auto-label Topics**: Extract and tag topics from discussions
- **Topic Timeline**: Show when topics were discussed
- **Topic Grouping**: Group related discussions across meetings
- **Smart Tags**: Industry-specific topic tags

### Implementation Plan

#### Phase 1: Topic Extraction
```typescript
// Enhance OnDeviceAIService
async generateTopics(transcript: string, industry?: string): Promise<Topic[]> {
    // Use FLAN-T5 to extract topics with timestamps
    // Return: [{topic: "Budget Discussion", startTime: 120, endTime: 300, keywords: [...]}]
}
```

**Prompt Engineering:**
```
Extract all topics discussed in this meeting transcript. For each topic:
1. Topic name (concise, 2-4 words)
2. Start timestamp (when topic began)
3. End timestamp (when topic ended)
4. Key keywords mentioned
5. Participants involved

Return as JSON array.
```

#### Phase 2: Topic Tagging System
- Store topics in database with metadata
- Link topics to transcript segments
- Create topic index for search
- Generate topic timeline visualization

#### Phase 3: Cross-Meeting Topic Analysis
- Identify recurring topics across meetings
- Show topic evolution over time
- Suggest related meetings based on topics

### Database Schema
```typescript
interface Topic {
    id: string;
    sessionId: string;
    name: string;
    startTime: number;
    endTime: number;
    keywords: string[];
    participants: string[];
    industry: string;
    createdAt: Date;
}
```

---

## 3. Human-Like Note Quality

### Current State
- ✅ Summary generation with FLAN-T5
- ✅ Industry-specific prompts
- ✅ Outline generation

### Target Features
- **Natural Language**: Summaries read like human-written notes
- **Contextual Understanding**: Better grasp of meeting flow
- **Digestible Format**: 2-3 paragraphs that make sense
- **Key Insights First**: Most important points at the top

### Implementation Plan

#### Enhanced Prompt Engineering
```typescript
const HUMAN_LIKE_SUMMARY_PROMPT = `
You are a professional note-taker. Write meeting notes as if a skilled human assistant took them.

Guidelines:
1. Start with the most important decisions or outcomes
2. Use natural, conversational language (not robotic)
3. Group related points together logically
4. Include context: "The team discussed X because Y"
5. Highlight action items naturally: "Sarah will follow up on..."
6. Keep it concise: 2-3 paragraphs maximum
7. Make it scannable: Use line breaks between major points

Write the notes now:
`;
```

#### Multi-Pass Summarization
1. **First Pass**: Extract key points and structure
2. **Second Pass**: Rewrite in natural language
3. **Third Pass**: Polish and ensure coherence

#### Quality Metrics
- Readability score (Flesch-Kincaid)
- Coherence checking
- Action item clarity
- Context preservation

---

## 4. Compliance Features

### HIPAA Compliance

#### Requirements
- ✅ Encryption at rest (already implemented)
- ⚠️ Audit logs for access
- ⚠️ Access controls
- ⚠️ Business Associate Agreement (BAA) template
- ⚠️ Data retention policies
- ⚠️ Breach notification procedures

#### Implementation
```typescript
interface AuditLog {
    id: string;
    userId: string;
    action: 'view' | 'edit' | 'delete' | 'export' | 'share';
    resourceType: 'session' | 'transcript' | 'notes';
    resourceId: string;
    timestamp: Date;
    ipAddress?: string;
    userAgent?: string;
}

class ComplianceService {
    async logAccess(action: AuditLog['action'], resource: {type: string, id: string}): Promise<void>
    async getAuditLogs(userId: string, startDate: Date, endDate: Date): Promise<AuditLog[]>
    async enforceRetentionPolicy(): Promise<void> // Auto-delete after retention period
}
```

### GDPR Compliance

#### Requirements
- ⚠️ Right to access (data export)
- ⚠️ Right to deletion
- ⚠️ Consent management
- ⚠️ Privacy policy integration
- ⚠️ Data processing records

#### Implementation
```typescript
class GDPRService {
    // Export all user data
    async exportUserData(userId: string): Promise<Blob> // JSON/CSV export
    
    // Delete all user data
    async deleteUserData(userId: string): Promise<void>
    
    // Consent tracking
    async recordConsent(userId: string, purpose: string, granted: boolean): Promise<void>
    
    // Privacy policy version tracking
    async getPrivacyPolicyVersion(): Promise<string>
    async recordPolicyAcceptance(userId: string, version: string): Promise<void>
}
```

### CCPA Compliance

#### Requirements
- ⚠️ California privacy rights notice
- ⚠️ Opt-out mechanisms
- ⚠️ Data sale disclosures (if applicable)
- ⚠️ Non-discrimination policy

#### Implementation
- Add CCPA notice in settings
- Opt-out toggle for data sharing
- Clear data sale disclosures (if any)
- Privacy rights request form

---

## 5. Calendar Integration

### Google Calendar Integration

#### Implementation
```typescript
class CalendarService {
    // Google Calendar API
    async connectGoogleCalendar(): Promise<void>
    async fetchUpcomingMeetings(daysAhead: number): Promise<Meeting[]>
    async getMeetingParticipants(meetingId: string): Promise<string[]>
    async getMeetingContext(meetingId: string): Promise<MeetingContext>
    
    // Outlook Calendar API
    async connectOutlookCalendar(): Promise<void>
    async fetchUpcomingMeetingsOutlook(daysAhead: number): Promise<Meeting[]>
}

interface Meeting {
    id: string;
    title: string;
    startTime: Date;
    endTime: Date;
    participants: string[];
    location?: string;
    description?: string;
    platform: 'zoom' | 'teams' | 'meet' | 'other';
}
```

#### Pre-Meeting Preparation
1. **Auto-detect upcoming meetings** from calendar
2. **Pull participant list** from calendar invite
3. **Extract meeting context** from description
4. **Pre-populate session** with meeting details
5. **Suggest relevant templates** based on meeting type

#### OAuth Flow
- Google OAuth 2.0 for Calendar API
- Microsoft Graph API for Outlook
- Store tokens securely (encrypted)
- Refresh token handling

---

## 6. CRM Sync

### Supported CRMs
- Salesforce
- HubSpot
- Pipedrive
- Custom API support

### Implementation
```typescript
class CRMSyncService {
    // Connect to CRM
    async connectCRM(provider: 'salesforce' | 'hubspot' | 'pipedrive', credentials: CRMCredentials): Promise<void>
    
    // Convert conversation to CRM data
    async syncSessionToCRM(session: Session, mapping: CRMMapping): Promise<CRMRecord>
    
    // Extract structured data
    async extractCRMData(session: Session): Promise<CRMData> {
        return {
            contacts: this.extractContacts(session),
            deals: this.extractDeals(session),
            tasks: this.extractTasks(session),
            notes: this.extractNotes(session),
            customFields: this.extractCustomFields(session)
        };
    }
}

interface CRMData {
    contacts: Contact[];
    deals: Deal[];
    tasks: Task[];
    notes: string;
    customFields: Record<string, any>;
}
```

#### Data Mapping
- **Contacts**: Extract participant names, emails, roles
- **Deals**: Identify opportunities, amounts, stages
- **Tasks**: Convert action items to CRM tasks
- **Notes**: Sync summary and key points
- **Custom Fields**: Map to CRM-specific fields

#### Sync Workflow
1. User configures CRM connection
2. After meeting, AI extracts structured data
3. User reviews and approves sync
4. Data pushed to CRM via API
5. Confirmation and error handling

---

## 7. Custom Templates

### Current State
- ✅ Industry-specific prompts (general, therapy, medical, legal, business)
- ⚠️ No customizable template system

### Target Features
- **Field-Specific Templates**: Medical, legal, therapy, sales, etc.
- **Configurable Sections**: User-defined sections
- **Template Library**: Pre-built templates
- **Template Sharing**: Share templates with team

### Implementation
```typescript
interface NoteTemplate {
    id: string;
    name: string;
    industry: string;
    sections: TemplateSection[];
    prompts: Record<string, string>; // Custom prompts per section
    isDefault: boolean;
    createdBy: string;
    createdAt: Date;
}

interface TemplateSection {
    id: string;
    name: string;
    order: number;
    prompt: string;
    required: boolean;
    aiGenerated: boolean; // Auto-generate or manual
}

class TemplateService {
    async getTemplates(industry?: string): Promise<NoteTemplate[]>
    async createTemplate(template: Omit<NoteTemplate, 'id' | 'createdAt'>): Promise<NoteTemplate>
    async applyTemplate(session: Session, templateId: string): Promise<Session>
}
```

#### Template Examples

**Medical Template:**
- Patient Information
- Chief Complaint
- History of Present Illness
- Assessment & Plan
- Follow-up Actions

**Legal Template:**
- Case Details
- Key Arguments
- Decisions Made
- Next Steps
- Deadlines

**Sales Template:**
- Opportunity Overview
- Pain Points Identified
- Proposed Solution
- Objections & Responses
- Next Steps & Commitments

---

## 8. Cross-Platform Compatibility

### Current State
- ✅ Web-based (works on Mac/Windows via browser)
- ⚠️ No native app
- ⚠️ Meeting platform compatibility untested

### Target Features
- **Mac Native App**: Electron or Tauri wrapper
- **Windows Native App**: Electron or Tauri wrapper
- **Meeting Platform Support**: Zoom, Teams, Google Meet, etc.
- **System Audio Capture**: Record system audio (not just mic)

### Implementation Plan

#### Phase 1: Electron/Tauri Wrapper
- Package web app as native app
- System tray integration
- Global hotkeys for recording
- Background recording capability

#### Phase 2: System Audio Capture
- Use system audio APIs (macOS: CoreAudio, Windows: WASAPI)
- Capture meeting audio even if not using mic
- Handle different audio sources

#### Phase 3: Meeting Platform Integration
- Detect active meeting platform
- Auto-start recording when meeting detected
- Platform-specific optimizations

---

## Implementation Priority

### Phase 1 (High Priority - Core Features)
1. ✅ Advanced Speaker Identification & Voice Learning
2. ✅ Automatic Topic Labeling
3. ✅ Human-Like Note Quality
4. ✅ Custom Templates

### Phase 2 (Medium Priority - Enterprise Features)
5. ✅ Calendar Integration
6. ✅ CRM Sync
7. ✅ Compliance Features (HIPAA, GDPR, CCPA)

### Phase 3 (Lower Priority - Platform Expansion)
8. ✅ Cross-Platform Native Apps
9. ✅ Advanced Meeting Platform Integration

---

## Technical Considerations

### Performance
- Voice embedding extraction: ~100-200ms per segment
- Speaker matching: ~50ms per segment
- Topic extraction: Add ~2-3s to analysis time
- Calendar sync: Background job, non-blocking

### Privacy & Security
- Voice embeddings stored locally (IndexedDB)
- No voice data sent to external servers
- Calendar tokens encrypted at rest
- CRM credentials encrypted
- Compliance logs stored securely

### User Experience
- Progressive enhancement: Features work without setup
- Graceful degradation: Fallback to basic features
- Clear onboarding for new features
- Helpful error messages

---

## Next Steps

1. **Review this roadmap** with stakeholders
2. **Prioritize features** based on user needs
3. **Create detailed technical specs** for Phase 1 features
4. **Begin implementation** with speaker learning system
5. **Iterate based on feedback**

---

## Questions to Resolve

1. **Voice Learning Model**: Which on-device model for voice embeddings? (speechbrain/spkrec-ecapa-voxceleb vs custom)
2. **Calendar API Limits**: Rate limits for Google/Outlook APIs?
3. **CRM Priority**: Which CRM to support first?
4. **Compliance Scope**: Full HIPAA/GDPR/CCPA or subset?
5. **Native App**: Electron vs Tauri for performance?
