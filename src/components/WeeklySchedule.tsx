/**
 * Weekly Schedule Component
 * Displays calendar events in a weekly view when calendar is connected
 * Adapted from React Native design with brand colors
 */

import React, { useState, useEffect } from 'react';
import { CalendarService, Meeting } from '../services/CalendarService';

interface WeeklyScheduleProps {
    calendarService: CalendarService | null;
    isConnected: boolean;
}

interface ScheduleItem {
    day: string;
    startTime: string;
    endTime: string;
    title: string;
    description: string;
    color: string;
}

const WeeklySchedule: React.FC<WeeklyScheduleProps> = ({ calendarService, isConnected }) => {
    const [selectedDate, setSelectedDate] = useState(new Date());
    const [isSearchVisible, setIsSearchVisible] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [scheduleItems, setScheduleItems] = useState<ScheduleItem[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string>('');
    const [isDarkMode, setIsDarkMode] = useState(false);

    useEffect(() => {
        const checkTheme = () => {
            const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
            setIsDarkMode(isDark);
        };
        
        checkTheme();
        const observer = new MutationObserver(checkTheme);
        observer.observe(document.documentElement, {
            attributes: true,
            attributeFilter: ['data-theme']
        });
        
        return () => observer.disconnect();
    }, []);

    // Brand colors - using your brand palette
    const brandColors = [
        'rgba(2, 41, 91, 0.1)',      // Authority Navy (light)
        'rgba(253, 167, 0, 0.15)',  // Achievement Gold (light)
        'rgba(44, 95, 65, 0.1)',    // Strategic Forest (light)
        'rgba(2, 41, 91, 0.08)',    // Navy (lighter)
        'rgba(253, 167, 0, 0.12)',  // Gold (lighter)
        'rgba(44, 95, 65, 0.08)',   // Forest (lighter)
    ];

    // Get week range
    const getWeekRange = (date: Date) => {
        const start = new Date(date);
        const day = start.getDay();
        const diff = start.getDate() - day + (day === 0 ? -6 : 1); // Adjust to Monday
        start.setDate(diff);
        
        const end = new Date(start);
        end.setDate(start.getDate() + 4); // Friday
        
        return { start, end };
    };

    const formatWeekRange = (date: Date) => {
        const { start, end } = getWeekRange(date);
        const options: Intl.DateTimeFormatOptions = { month: 'short', day: 'numeric', year: 'numeric' };
        return `${start.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} - ${end.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}`;
    };

    const formatTime = (date: Date) => {
        return date.toLocaleTimeString('en-US', { 
            hour: 'numeric', 
            minute: '2-digit',
            hour12: true 
        });
    };

    const getDayAbbreviation = (date: Date) => {
        const days = ['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT'];
        return days[date.getDay()];
    };

    // Fetch calendar events
    useEffect(() => {
        if (!isConnected || !calendarService) {
            setScheduleItems([]);
            return;
        }

        const fetchEvents = async () => {
            setIsLoading(true);
            setError('');
            try {
                const meetings = await calendarService.fetchUpcomingMeetings(7);
                
                // Convert meetings to schedule items
                const items: ScheduleItem[] = meetings.map((meeting, index) => {
                    const day = getDayAbbreviation(meeting.startTime);
                    const color = brandColors[index % brandColors.length];
                    
                    return {
                        day,
                        startTime: formatTime(meeting.startTime),
                        endTime: formatTime(meeting.endTime),
                        title: meeting.title,
                        description: meeting.description || meeting.location || '',
                        color
                    };
                });

                setScheduleItems(items);
            } catch (err: any) {
                setError(err.message || 'Failed to fetch calendar events');
                console.error('Error fetching calendar events:', err);
            } finally {
                setIsLoading(false);
            }
        };

        fetchEvents();
    }, [calendarService, isConnected, selectedDate]);

    const filteredItems = scheduleItems.filter(
        (item) =>
            item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
            item.description.toLowerCase().includes(searchQuery.toLowerCase())
    );

    const navigateWeek = (direction: 'prev' | 'next') => {
        const newDate = new Date(selectedDate);
        newDate.setDate(selectedDate.getDate() + (direction === 'next' ? 7 : -7));
        setSelectedDate(newDate);
    };

    const weekDays = ['MON', 'TUE', 'WED', 'THU', 'FRI'];

    if (!isConnected) {
        return null;
    }

    const styles = getStyles(isDarkMode);

    return (
        <div style={styles.container}>
            <div style={styles.header}>
                <button
                    onClick={() => {}}
                    style={styles.iconButton}
                    title="Back"
                >
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M19 12H5M12 19l-7-7 7-7"/>
                    </svg>
                </button>
                <h2 style={styles.headerTitle}>Weekly Schedule</h2>
                <button
                    onClick={() => setIsSearchVisible(!isSearchVisible)}
                    style={styles.iconButton}
                    title={isSearchVisible ? 'Close search' : 'Search'}
                >
                    {isSearchVisible ? (
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <line x1="18" y1="6" x2="6" y2="18"/>
                            <line x1="6" y1="6" x2="18" y2="18"/>
                        </svg>
                    ) : (
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="11" cy="11" r="8"/>
                            <path d="m21 21-4.35-4.35"/>
                        </svg>
                    )}
                </button>
            </div>

            {isSearchVisible && (
                <div style={styles.searchBar}>
                    <input
                        type="text"
                        style={styles.searchInput}
                        placeholder="Search events..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                    />
                </div>
            )}

            <div style={styles.weekSelector}>
                <button
                    onClick={() => navigateWeek('prev')}
                    style={styles.weekNavButton}
                    title="Previous week"
                >
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#999" strokeWidth="2">
                        <polyline points="15 18 9 12 15 6"/>
                    </svg>
                </button>
                <span style={styles.weekText}>{formatWeekRange(selectedDate)}</span>
                <button
                    onClick={() => navigateWeek('next')}
                    style={styles.weekNavButton}
                    title="Next week"
                >
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#999" strokeWidth="2">
                        <polyline points="9 18 15 12 9 6"/>
                    </svg>
                </button>
            </div>

            {isLoading && (
                <div style={styles.loadingState}>
                    <div className="spinner" style={{ width: '24px', height: '24px', borderWidth: '3px' }}></div>
                    <span>Loading events...</span>
                </div>
            )}

            {error && (
                <div style={styles.errorState}>
                    <span>⚠️ {error}</span>
                </div>
            )}

            {!isLoading && !error && (
                <div style={styles.scrollContainer}>
                    {weekDays.map((day) => {
                        const dayItems = filteredItems.filter((item) => item.day === day);
                        return (
                            <div key={day} style={styles.daySection}>
                                <h3 style={styles.dayTitle}>{day}</h3>
                                {dayItems.length === 0 ? (
                                    <div style={styles.emptyDay}>No events</div>
                                ) : (
                                    dayItems.map((item, index) => (
                                        <div key={index} style={[styles.eventCard, { backgroundColor: item.color }]}>
                                            <div style={styles.eventTime}>
                                                <div style={styles.eventTimeText}>{item.startTime}</div>
                                                <div style={styles.eventTimeText}>{item.endTime}</div>
                                            </div>
                                            <div style={styles.eventDetails}>
                                                <div style={styles.eventTitle}>{item.title}</div>
                                                {item.description && (
                                                    <div style={styles.eventDescription}>{item.description}</div>
                                                )}
                                            </div>
                                            <button style={styles.eventMenuButton} title="More options">
                                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#999" strokeWidth="2">
                                                    <circle cx="12" cy="5" r="1"/>
                                                    <circle cx="12" cy="12" r="1"/>
                                                    <circle cx="12" cy="19" r="1"/>
                                                </svg>
                                            </button>
                                        </div>
                                    ))
                                )}
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
};

const getStyles = (isDarkMode: boolean) => ({
    container: {
        flex: 1,
        backgroundColor: isDarkMode ? 'rgba(27, 52, 72, 0.5)' : '#fff',
        display: 'flex',
        flexDirection: 'column' as const,
        height: '100%',
        borderRadius: '8px',
        overflow: 'hidden',
    },
    header: {
        display: 'flex',
        flexDirection: 'row' as const,
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '16px',
        borderBottom: isDarkMode ? '1px solid rgba(255, 255, 255, 0.1)' : '1px solid #e0e0e0',
    },
    headerTitle: {
        fontSize: '18px',
        fontWeight: '600',
        color: isDarkMode ? '#fda700' : '#1a1a1a',
        margin: 0,
    },
    iconButton: {
        background: 'transparent',
        border: 'none',
        cursor: 'pointer',
        padding: '4px',
        borderRadius: '6px',
        color: '#1a1a1a',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        transition: 'all 0.15s',
    },
    searchBar: {
        padding: '16px',
        borderBottom: '1px solid #e0e0e0',
    },
    searchInput: {
        width: '100%',
        backgroundColor: isDarkMode ? 'rgba(27, 52, 72, 0.8)' : '#f0f0f0',
        borderRadius: '8px',
        padding: '8px 12px',
        border: isDarkMode ? '1px solid rgba(255, 255, 255, 0.1)' : '1px solid transparent',
        fontSize: '14px',
        outline: 'none',
        transition: 'all 0.15s',
        color: isDarkMode ? '#e2e8f0' : '#1a1a1a',
    },
    weekSelector: {
        display: 'flex',
        flexDirection: 'row' as const,
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '16px',
        borderBottom: isDarkMode ? '1px solid rgba(255, 255, 255, 0.1)' : '1px solid #e0e0e0',
    },
    weekText: {
        fontSize: '16px',
        fontWeight: '600',
        color: isDarkMode ? '#cbd5e1' : '#1a1a1a',
    },
    weekNavButton: {
        background: 'transparent',
        border: 'none',
        cursor: 'pointer',
        padding: '4px',
        borderRadius: '6px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        transition: 'all 0.15s',
    },
    scrollContainer: {
        flex: 1,
        overflowY: 'auto',
        paddingBottom: '16px',
    },
    daySection: {
        marginBottom: '16px',
    },
    dayTitle: {
        fontSize: '16px',
        fontWeight: '600',
        marginLeft: '16px',
        marginTop: '16px',
        marginBottom: '8px',
        color: isDarkMode ? '#fda700' : '#1a1a1a',
    },
    eventCard: {
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
        padding: '16px',
        marginHorizontal: '16px',
        marginBottom: '8px',
        borderRadius: '8px',
        border: '1px solid rgba(0, 0, 0, 0.05)',
    },
    eventTime: {
        marginRight: '16px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'flex-start',
        minWidth: '80px',
    },
    eventTimeText: {
        fontSize: '12px',
        color: '#666',
        lineHeight: '1.4',
    },
    eventDetails: {
        flex: 1,
        minWidth: 0,
    },
    eventTitle: {
        fontSize: '16px',
        fontWeight: '600',
        marginBottom: '4px',
        color: isDarkMode ? '#e2e8f0' : '#1a1a1a',
    },
    eventDescription: {
        fontSize: '14px',
        color: isDarkMode ? '#9ca3af' : '#666',
        lineHeight: '1.4',
    },
    eventMenuButton: {
        background: 'transparent',
        border: 'none',
        cursor: 'pointer',
        padding: '4px',
        borderRadius: '4px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        marginLeft: '8px',
        transition: 'all 0.15s',
    },
    emptyDay: {
        padding: '16px',
        marginHorizontal: '16px',
        color: '#999',
        fontSize: '14px',
        fontStyle: 'italic',
    },
    loadingState: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '12px',
        padding: '40px',
        color: '#666',
    },
    errorState: {
        padding: '16px',
        margin: '16px',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        border: '1px solid rgba(239, 68, 68, 0.2)',
        borderRadius: '8px',
        color: '#dc2626',
        fontSize: '14px',
    },
});

export default WeeklySchedule;
