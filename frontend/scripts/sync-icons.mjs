import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
const root = fileURLToPath(new URL('../', import.meta.url));
const catalog = JSON.parse(readFileSync(join(root, 'node_modules/@iconify-json/solar/icons.json'), 'utf8'));
function files(dir) { return readdirSync(dir, { withFileTypes: true }).flatMap(e => e.isDirectory() ? files(join(dir, e.name)) : /\.(tsx?|json)$/.test(e.name) && e.name !== 'icons.json' ? [join(dir, e.name)] : []); }
const names = [...new Set(files(join(root, 'src')).flatMap(path => [...readFileSync(path, 'utf8').matchAll(/solar:([a-z0-9-]+)/g)].map(m => m[1])))].sort();
const icons = Object.fromEntries(names.map(name => { if (!catalog.icons[name]) throw new Error(`Unknown Solar icon: ${name}`); return [`solar:${name}`, catalog.icons[name]]; }));
writeFileSync(join(root, 'src/components/icons.json'), JSON.stringify(icons));
console.log(`Bundled ${names.length} offline Solar icons.`);
