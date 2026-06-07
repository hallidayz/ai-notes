/**
 * Generates Acaiguardian line-style PNG icons (Streamline Regular / Icons8 line aesthetic).
 * Run: node scripts/generate-icons.mjs
 */
import fs from 'fs/promises';
import path from 'path';
import sharp from 'sharp';

const ROOT = process.cwd();
const OUT_DIR = path.join(ROOT, 'public', 'icons');
const SIZES = [24, 48, 192];

const THEMES = {
  light: { stroke: '#1a2033', accent: '#3740ff' },
  dark: { stroke: '#e2e8f0', accent: '#6b7aff' },
};

const ICONS = {
  settings: `<circle cx="12" cy="12" r="3"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/>`,
  sun: `<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/>`,
  moon: `<path d="M20 14.5A8.5 8.5 0 0 1 9.5 4 7 7 0 1 0 20 14.5z"/>`,
  close: `<path d="M6 6l12 12M18 6L6 18"/>`,
  shield: `<path d="M12 3l7 3v6c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6l7-3z"/><path d="M9 12l2 2 4-4"/>`,
  delete: `<path d="M4 7h16M9 7V5h6v2M7 7l1 12h8l1-12"/>`,
  record: `<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3" fill="STROKE"/>`,
  stop: `<rect x="6" y="6" width="12" height="12" rx="1"/><rect x="8.5" y="8.5" width="7" height="7" rx="0.5" fill="STROKE"/>`,
  check: `<path d="M5 12l5 5 9-10"/>`,
  download: `<path d="M12 4v10M8 10l4 4 4-4M5 18h14"/>`,
  info: `<circle cx="12" cy="12" r="8"/><path d="M12 10v6M12 8h.01"/>`,
  'chevron-left': `<path d="M15 6l-6 6 6 6"/>`,
  loader: `<path d="M12 3a9 9 0 1 0 9 9" stroke-dasharray="14 40"/>`,
  calendar: `<rect x="4" y="5" width="16" height="15" rx="2"/><path d="M8 3v4M16 3v4M4 10h16"/>`,
  warning: `<path d="M12 5l8 14H4L12 5z"/><path d="M12 10v4M12 17h.01"/>`,
  plus: `<path d="M12 6v12M6 12h12"/>`,
  summary: `<path d="M7 5h10v14H7z"/><path d="M9 9h6M9 12h6M9 15h4"/>`,
  'action-items': `<path d="M6 7h12M6 12h12M6 17h8"/><path d="M4 7h.01M4 12h.01M4 17h.01"/>`,
  outline: `<path d="M8 7h8M8 12h8M8 17h5"/><circle cx="5" cy="7" r="1" fill="STROKE"/><circle cx="5" cy="12" r="1" fill="STROKE"/><circle cx="5" cy="17" r="1" fill="STROKE"/>`,
  'ai-chip': `<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 6V4M15 6V4M9 18v2M15 18v2M6 9H4M6 15H4M18 9h2M18 15h2"/><path d="M10 10h4v4h-4z"/>`,
  google: `<circle cx="12" cy="12" r="8"/><path d="M12 8v8M8 12h8"/>`,
  microsoft: `<rect x="5" y="5" width="6" height="6"/><rect x="13" y="5" width="6" height="6"/><rect x="5" y="13" width="6" height="6"/><rect x="13" y="13" width="6" height="6"/>`,
  notion: `<path d="M6 6h12v12H6z"/><path d="M9 9h6v6H9z"/>`,
  apple: `<path d="M16 10c0-2 1.5-3.5 2.5-4-1.5 0-3.5 1-4.5 2.5C13 7 12 5 10.5 5 8 5 6 7.5 6 11c0 4 2.5 7 5 7 1 0 2-.5 2.5-1 1 .5 2 1 3 1 2.5 0 4.5-3 4.5-8z"/>`,
  logo: `<path d="M12 3l7 3v6c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6l7-3z"/><path d="M10 11h4M12 9v6"/>`,
};

function svgForIcon(name, paths, colors) {
  const content = paths
    .replaceAll('STROKE', colors.stroke)
    .replaceAll('ACCENT', colors.accent);

  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
    <g fill="none" stroke="${colors.stroke}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
      ${content}
    </g>
  </svg>`;
}

async function writePng(svg, outPath, size) {
  await sharp(Buffer.from(svg))
    .resize(size, size)
    .png()
    .toFile(outPath);
}

async function main() {
  for (const theme of Object.keys(THEMES)) {
    const themeDir = path.join(OUT_DIR, theme);
    await fs.mkdir(themeDir, { recursive: true });

    for (const [name, paths] of Object.entries(ICONS)) {
      const svg = svgForIcon(name, paths, THEMES[theme]);
      for (const size of SIZES) {
        const filename = size === 24 ? `${name}.png` : `${name}@${size}.png`;
        await writePng(svg, path.join(themeDir, filename), size);
      }
    }
  }

  // App favicon / PWA icons (brand shield on accent background)
  const brandDir = path.join(ROOT, 'public', 'brand');
  await fs.mkdir(brandDir, { recursive: true });

  for (const [theme, colors] of Object.entries({
    light: { bg: '#3740ff', fg: '#ffffff' },
    dark: { bg: '#1a2033', fg: '#6b7aff' },
  })) {
    const logoSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 192 192" width="192" height="192">
      <rect width="192" height="192" rx="40" fill="${colors.bg}"/>
      <g fill="none" stroke="${colors.fg}" stroke-width="8" stroke-linecap="round" stroke-linejoin="round">
        <path d="M96 40l56 24v48c0 36-24 60-56 72-32-12-56-36-56-72V64l56-24z"/>
        <path d="M80 96h32M96 80v32"/>
      </g>
    </svg>`;
    await writePng(logoSvg, path.join(brandDir, `logo-${theme}.png`), 192);
    await writePng(logoSvg, path.join(brandDir, `logo-${theme}@512.png`), 512);
  }

  await writePng(
    (await fs.readFile(path.join(brandDir, 'logo-light.png'))),
    path.join(ROOT, 'public', 'favicon.png'),
    32
  );

  console.log(`Generated ${Object.keys(ICONS).length} icons × ${Object.keys(THEMES).length} themes in public/icons/`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
