import { defineConfig } from 'astro/config';
export default defineConfig({
  server: { port: 4321 },
  vite: { server: { proxy: {
    '/api': { target: process.env.API_PROXY_TARGET || 'http://localhost:8000', rewrite: path => path.replace(/^\/api/, '') },
  } } },
});
