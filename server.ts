/**
 * Backend Server Entry Point
 * Runs the Hono server with tRPC endpoints
 */

import { serve } from '@hono/node-server';
import app from './backend/hono';

// Render automatically sets PORT, default to 3000 for local development
const PORT = process.env.PORT || 3000;

// Log environment info
console.log('Environment:', process.env.NODE_ENV || 'development');
console.log('Port:', PORT);

console.log('🚀 Starting RETINA Backend Server...');
console.log(`📡 Server will run on: http://localhost:${PORT}`);
console.log(`🔗 tRPC endpoint: http://localhost:${PORT}/trpc`);
console.log(`🏥 Health check: http://localhost:${PORT}/`);
console.log('');

serve({
  fetch: app.fetch,
  port: Number(PORT),
}, (info) => {
  console.log(`✅ Server is running on port ${info.port}`);
  console.log(`🌐 Access your API at: http://localhost:${info.port}`);
  console.log(`📱 Frontend should connect to: http://localhost:${info.port}/trpc`);
  console.log('');
  console.log('Available endpoints:');
  console.log(`  GET  http://localhost:${info.port}/ - Health check`);
  console.log(`  GET  http://localhost:${info.port}/api/model/status - Model status`);
  console.log(`  POST http://localhost:${info.port}/trpc/detection.analyze - Standard analysis`);
  console.log(`  POST http://localhost:${info.port}/trpc/detection.analyzeEnhanced - Enhanced pipeline`);
  console.log('');
});

