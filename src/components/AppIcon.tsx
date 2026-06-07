import React from 'react';
import { iconSrc } from '../icons';
import type { IconName } from '../icons';

export type { IconName };

export const AppIcon: React.FC<AppIconProps> = ({
    name,
    size = 20,
    className = '',
    isDarkMode = false,
    alt = '',
}) => (
    <img
        src={iconSrc(name, isDarkMode, size <= 24 ? 24 : size <= 48 ? 48 : 192)}
        width={size}
        height={size}
        alt={alt}
        className={`app-icon ${className}`.trim()}
        aria-hidden={!alt}
        draggable={false}
    />
);
