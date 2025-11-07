import { createTRPCRouter } from "./create-context";

// 🧩 Example route
import hiRoute from "./routes/example/hi/route";

// 🔍 Detection routes
import analyzeProcedure from "./routes/detection/analyze/route";
import saveDetectionProcedure from "./routes/detection/save/route";
import historyProcedure from "./routes/detection/history/route";

// 🧠 Training routes
import validateImageProcedure from "./routes/training/validate/route";
import saveTrainingImageProcedure from "./routes/training/save/route";
import trainingStatsProcedure from "./routes/training/stats/route";
import retrainModelProcedure from "./routes/training/retrain/route"; // ✅ NEW IMPORT

// -------------------------------------
// 🌐 Main TRPC App Router
// -------------------------------------
export const appRouter = createTRPCRouter({
  example: createTRPCRouter({
    hi: hiRoute,
  }),

  detection: createTRPCRouter({
    analyze: analyzeProcedure,
    save: saveDetectionProcedure,
    history: historyProcedure,
  }),

  training: createTRPCRouter({
    validate: validateImageProcedure,
    save: saveTrainingImageProcedure,
    stats: trainingStatsProcedure,
    retrain: retrainModelProcedure, // ✅ NEW ROUTE ADDED
  }),
});

export type AppRouter = typeof appRouter;
