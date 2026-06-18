import fs from 'fs/promises';
import path from 'path';
import sharp from 'sharp';

const outDir = path.join(process.cwd(), 'electron', 'assets');
await fs.mkdir(outDir, { recursive: true });

const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <rect width="512" height="512" rx="96" fill="#3740ff"/>
  <g fill="none" stroke="#ffffff" stroke-width="24" stroke-linecap="round" stroke-linejoin="round">
    <path d="M256 96l140 60v120c0 72-48 120-140 144-92-24-140-72-140-144V156l140-60z"/>
    <path d="M220 256h72M256 220v72"/>
  </g>
</svg>`;

await sharp(Buffer.from(svg)).png().toFile(path.join(outDir, 'icon.png'));
console.log('Wrote electron/assets/icon.png');
