/**
 * Wave Component - Animated wave animation using react-wavify
 * Can be used as a decorative element or background
 * 
 * Usage Examples:
 * 
 * // Basic wave at bottom of section
 * <WaveMaker />
 * 
 * // Custom colored wave
 * <WaveMaker fill="#fda700" speed={0.5} amplitude={30} />
 * 
 * // Wave at top of section
 * <WaveMaker position="top" fill="#2c5f41" />
 * 
 * // Pre-configured brand waves
 * <BrandWave />
 * <GoldWave speed={0.3} />
 * <ForestWave amplitude={25} />
 * 
 * // In a section divider
 * <section>
 *   <h2>Title</h2>
 *   <WaveMaker position="bottom" />
 * </section>
 */

import React from 'react';
import Wave from 'react-wavify';

interface WaveProps {
    /** Wave fill color (default: brand blue) */
    fill?: string;
    /** Animation speed (default: 0.25) */
    speed?: number;
    /** Wave amplitude/height (default: 20) */
    amplitude?: number;
    /** Number of points forming the wave (default: 20) */
    points?: number;
    /** Whether to pause the animation */
    paused?: boolean;
    /** Additional CSS classes */
    className?: string;
    /** Style object */
    style?: React.CSSProperties;
    /** Wave position: 'top' or 'bottom' (default: 'bottom') */
    position?: 'top' | 'bottom';
}

export const WaveMaker: React.FC<WaveProps> = ({
    fill = '#02295b',
    speed = 0.25,
    amplitude = 20,
    points = 20,
    paused = false,
    className = '',
    style = {},
    position = 'bottom'
}) => {
    const waveStyle: React.CSSProperties = {
        width: '100%',
        height: '100px',
        display: 'block',
        ...style
    };

    return (
        <div 
            className={`wave-container ${className}`}
            style={{
                ...waveStyle,
                transform: position === 'top' ? 'rotate(180deg)' : 'none',
                marginTop: position === 'top' ? '-1px' : 0,
                marginBottom: position === 'bottom' ? '-1px' : 0,
                overflow: 'hidden'
            }}
        >
            <Wave
                fill={fill}
                paused={paused}
                options={{
                    height: amplitude,
                    amplitude: amplitude,
                    speed: speed,
                    points: points
                }}
            />
        </div>
    );
};

// Pre-configured wave variants
export const BrandWave: React.FC<Omit<WaveProps, 'fill'>> = (props) => (
    <WaveMaker {...props} fill="#02295b" />
);

export const GoldWave: React.FC<Omit<WaveProps, 'fill'>> = (props) => (
    <WaveMaker {...props} fill="#fda700" />
);

export const ForestWave: React.FC<Omit<WaveProps, 'fill'>> = (props) => (
    <WaveMaker {...props} fill="#2c5f41" />
);
