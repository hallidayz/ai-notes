import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, process.cwd(), '');
    const port = Number(process.env.PORT ?? env.PORT ?? 4783);
    const host = process.env.HOST ?? env.HOST ?? '0.0.0.0';

    return {
      server: {
        port,
        host,
        strictPort: true,
      },
      preview: {
        port,
        host,
        strictPort: true,
      },
      plugins: [react()],
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      }
    };
});
