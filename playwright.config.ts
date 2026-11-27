import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  timeout: 180000,
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: 'http://127.0.0.1:5233/',
    headless: true,
    launchOptions: {
      args: ['--use-angle=gl', '--enable-unsafe-webgpu'],
    },
  },
  webServer: {
    command: 'node node_modules/vite/bin/vite.js --port 5233 --strictPort --host 127.0.0.1',
    url: 'http://127.0.0.1:5233/',
    timeout: 60000,
    reuseExistingServer: true,
  },
});
