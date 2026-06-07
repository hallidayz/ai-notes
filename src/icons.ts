export type IconName =
    | 'settings'
    | 'sun'
    | 'moon'
    | 'close'
    | 'shield'
    | 'delete'
    | 'record'
    | 'stop'
    | 'check'
    | 'download'
    | 'info'
    | 'chevron-left'
    | 'loader'
    | 'calendar'
    | 'warning'
    | 'plus'
    | 'summary'
    | 'action-items'
    | 'outline'
    | 'ai-chip'
    | 'google'
    | 'microsoft'
    | 'notion'
    | 'apple'
    | 'logo';

export const iconSrc = (name: IconName, isDarkMode: boolean, size = 24): string => {
    const theme = isDarkMode ? 'dark' : 'light';
    if (name === 'logo') {
        return size > 192 ? `/brand/logo-${theme}@512.png` : `/brand/logo-${theme}.png`;
    }
    const suffix = size > 24 && size <= 48 ? '@48' : size > 48 ? '@192' : '';
    return `/icons/${theme}/${name}${suffix}.png`;
};
