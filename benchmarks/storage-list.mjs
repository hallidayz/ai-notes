import fs from 'fs/promises';
import path from 'path';

const STORAGE_DIR = path.join(process.cwd(), 'local_storage');
const numFiles = 1000;

async function setup() {
  await fs.mkdir(STORAGE_DIR, { recursive: true });
  const promises = [];
  for (let i = 0; i < numFiles; i++) {
    const data = { index: i, text: "Some dummy content to simulate file data. ".repeat(10) };
    promises.push(fs.writeFile(path.join(STORAGE_DIR, `bench_${i}.json`), JSON.stringify(data)));
  }
  await Promise.all(promises);
}

async function cleanup() {
  const files = await fs.readdir(STORAGE_DIR);
  const promises = files.filter(f => f.startsWith('bench_')).map(f => fs.unlink(path.join(STORAGE_DIR, f)));
  await Promise.all(promises);
}

async function measureOriginal() {
  const start = performance.now();
  const files = await fs.readdir(STORAGE_DIR);
  const items = [];
  for (const file of files) {
    if (file.endsWith('.json')) {
      const id = file.replace('.json', '');
      const content = await fs.readFile(path.join(STORAGE_DIR, file), 'utf-8');
      items.push({ id, data: JSON.parse(content) });
    }
  }
  const end = performance.now();
  return end - start;
}

async function measureOptimized() {
  const start = performance.now();
  const files = await fs.readdir(STORAGE_DIR);
  const itemsPromises = files
    .filter(file => file.endsWith('.json'))
    .map(async file => {
      const id = file.replace('.json', '');
      const content = await fs.readFile(path.join(STORAGE_DIR, file), 'utf-8');
      return { id, data: JSON.parse(content) };
    });
  const items = await Promise.all(itemsPromises);
  const end = performance.now();
  return end - start;
}

async function run() {
  await setup();
  console.log(`Created ${numFiles} test files.`);

  // Warmup
  await measureOriginal();
  await measureOptimized();

  // Measure Original
  let origTotal = 0;
  for (let i = 0; i < 5; i++) origTotal += await measureOriginal();
  const origAvg = origTotal / 5;
  console.log(`Original average time: ${origAvg.toFixed(2)} ms`);

  // Measure Optimized
  let optTotal = 0;
  for (let i = 0; i < 5; i++) optTotal += await measureOptimized();
  const optAvg = optTotal / 5;
  console.log(`Optimized average time: ${optAvg.toFixed(2)} ms`);

  console.log(`Improvement: ${((origAvg - optAvg) / origAvg * 100).toFixed(2)}%`);

  await cleanup();
}

run();
