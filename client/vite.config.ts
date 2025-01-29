import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    exclude: ['lucide-react'],
  },
  server: {
    host: true,  // expose to all network interfaces
    port: 3000,
    strictPort: true  // fail if port 3000 is not available
  }
});
