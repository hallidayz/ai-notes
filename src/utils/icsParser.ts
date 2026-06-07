export interface ParsedCalendarEvent {
    id: string;
    title: string;
    start: string;
    end: string;
    provider: 'local';
}

function unfoldIcsLines(text: string): string[] {
    const raw = text.replace(/\r\n/g, '\n').split('\n');
    const lines: string[] = [];

    for (const line of raw) {
        if (line.startsWith(' ') || line.startsWith('\t')) {
            lines[lines.length - 1] += line.slice(1);
        } else {
            lines.push(line);
        }
    }

    return lines;
}

function parseIcsDate(value: string): string {
    if (!value) return new Date().toISOString();

    const cleaned = value.replace(/;.*$/, '').trim();
    if (cleaned.length === 8) {
        const y = cleaned.slice(0, 4);
        const m = cleaned.slice(4, 6);
        const d = cleaned.slice(6, 8);
        return new Date(`${y}-${m}-${d}T00:00:00`).toISOString();
    }

    const isoLike = cleaned.replace(
        /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z?$/,
        '$1-$2-$3T$4:$5:$6Z'
    );
    const parsed = new Date(isoLike);
    return Number.isNaN(parsed.getTime()) ? new Date().toISOString() : parsed.toISOString();
}

export function parseIcsEvents(icsText: string): ParsedCalendarEvent[] {
    const lines = unfoldIcsLines(icsText);
    const events: ParsedCalendarEvent[] = [];
    let inEvent = false;
    let current: Partial<ParsedCalendarEvent> = {};

    for (const line of lines) {
        if (line === 'BEGIN:VEVENT') {
            inEvent = true;
            current = { provider: 'local' };
            continue;
        }

        if (line === 'END:VEVENT') {
            if (current.title && current.start) {
                events.push({
                    id: current.id || `local-${events.length}-${current.start}`,
                    title: current.title,
                    start: current.start,
                    end: current.end || current.start,
                    provider: 'local',
                });
            }
            inEvent = false;
            current = {};
            continue;
        }

        if (!inEvent) continue;

        const [key, ...rest] = line.split(':');
        const value = rest.join(':');

        if (key === 'UID') current.id = `local-${value}`;
        if (key === 'SUMMARY') current.title = value.trim();
        if (key.startsWith('DTSTART')) current.start = parseIcsDate(value);
        if (key.startsWith('DTEND')) current.end = parseIcsDate(value);
    }

    return events.sort((a, b) => new Date(a.start).getTime() - new Date(b.start).getTime());
}

export function parseIcsEventsFromFile(file: File): Promise<ParsedCalendarEvent[]> {
    return file.text().then((text) => parseIcsEvents(text));
}
