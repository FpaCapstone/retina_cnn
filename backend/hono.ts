import { Hono } from "hono";
import { trpcServer } from "@hono/trpc-server";
import { cors } from "hono/cors";
import { appRouter } from "./trpc/app-router";
import { createContext } from "./trpc/create-context";
import fs from "fs";
import path from "path";

// -------------------------------------
// 🚀 Initialize Hono app
// -------------------------------------
const app = new Hono();

// Enable CORS for all routes (needed for Expo / web access)
app.use("*", cors());

// -------------------------------------
// 🔗 TRPC endpoint (matching frontend’s /trpc URL)
// -------------------------------------
app.use(
  "/trpc/*",
  trpcServer({
    router: appRouter,
    createContext,
  })
);

// -------------------------------------
// 🩺 Health check
// -------------------------------------
app.get("/", (c) => {
  return c.json({
    status: "ok",
    message: "API is running ✅",
    timestamp: new Date().toISOString(),
  });
});

// -------------------------------------
// 🧠 Model status check endpoint
// -------------------------------------
app.get("/api/model/status", (c) => {
  const modelDir = path.join(process.cwd(), "backend", "models");
  const kerasModel = path.join(modelDir, "outer_eye_mobilenetv2.h5");
  const tfliteModel = path.join(modelDir, "outer_eye_mobilenetv2.tflite");

  const status = {
    keras: fs.existsSync(kerasModel),
    tflite: fs.existsSync(tfliteModel),
  };

  return c.json({
    status: "ok",
    models: status,
    message: status.keras || status.tflite
      ? "Model files found and ready."
      : "No trained model found. Retraining required.",
  });
});

// -------------------------------------
// ⚠️ Global error handler
// -------------------------------------
app.onError((err, c) => {
  console.error("🔥 Server Error:", err);
  return c.json(
    {
      status: "error",
      message: err.message || "Internal server error",
      stack: process.env.NODE_ENV === "development" ? err.stack : undefined,
    },
    500
  );
});

export default app;
