// Script to generate icon files from logo.png
// Run with: npm run generate-icons
// Requires: sharp package (npm install sharp --save-dev)

import sharp from 'sharp';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const logoPath = path.join(__dirname, '../public/logo.png');
const publicDir = path.join(__dirname, '../public');

async function generateIcons() {
    try {
        // Check if logo exists
        if (!fs.existsSync(logoPath)) {
            console.error('Error: logo.png not found in public/ directory');
            process.exit(1);
        }

        console.log('Generating icon files from logo.png...');

        // Generate PNG icons
        const sizes = [192, 512];
        for (const size of sizes) {
            const outputPath = path.join(publicDir, `icon-${size}.png`);
            await sharp(logoPath)
                .resize(size, size, {
                    fit: 'contain',
                    background: { r: 13, g: 71, b: 161, alpha: 1 } // #0d47a1 brand background
                })
                .toFile(outputPath);
            console.log(`✓ Generated icon-${size}.png`);
        }

        // Generate ICO file (multi-size: 16x16, 32x32, 48x48)
        console.log('\nGenerating favicon.ico...');
        try {
            const toIco = (await import('to-ico')).default;
            
            // Generate different sizes for ICO
            const sizes = [16, 32, 48];
            const buffers = [];
            
            for (const size of sizes) {
                const buffer = await sharp(logoPath)
                    .resize(size, size, {
                        fit: 'contain',
                        background: { r: 13, g: 71, b: 161, alpha: 1 } // #0d47a1 brand background
                    })
                    .png()
                    .toBuffer();
                buffers.push(buffer);
            }
            
            const icoBuffer = await toIco(buffers);
            const icoPath = path.join(publicDir, 'favicon.ico');
            fs.writeFileSync(icoPath, icoBuffer);
            console.log('✓ Generated favicon.ico (16x16, 32x32, 48x48)');
        } catch (error) {
            console.warn('⚠ Could not generate ICO file:', error.message);
            console.log('You can generate favicon.ico manually using:');
            console.log('  - Online: https://favicon.io or https://convertio.co');
            console.log('  - ImageMagick: magick logo.png -define icon:auto-resize=16,32,48 favicon.ico');
        }
        
        console.log('\n✓ Icon generation complete!');
        console.log('Generated files:');
        console.log('  - icon-192.png');
        console.log('  - icon-512.png');
        console.log('  - favicon.ico (if successful)');

    } catch (error) {
        console.error('Error generating icons:', error.message);
        if (error.message.includes('sharp')) {
            console.error('\nTo fix: Run "npm install sharp --save-dev"');
        }
        process.exit(1);
    }
}

generateIcons();

