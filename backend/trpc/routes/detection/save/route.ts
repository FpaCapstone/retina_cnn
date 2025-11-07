import { publicProcedure } from "../../../create-context";
import { z } from "zod";
import fs from "fs";
import path from "path";

// Align schema with actual supported classes
const diseaseTypeSchema = z.enum([
  "Normal",
  "Uveitis",
  "Conjunctivitis",
  "Cataract",
  "Eyelid Drooping",
]);

const detectionSchema = z.object({
  disease: diseaseTypeSchema,
  confidence: z.number(),
  percentage: z.number().optional().default(0),
});

const saveDetectionSchema = z.object({
  id: z.string(),
  primaryDisease: diseaseTypeSchema,
  detections: z.array(detectionSchema),
  imageUri: z.string(),
  timestamp: z.string(),
  details: z.string().optional().default(""),
});

export const saveDetectionProcedure = publicProcedure
  .input(saveDetectionSchema)
  .mutation(async ({ input }) => {
    console.log("[Backend] 💾 Saving detection record:", input.id);
    console.log("[Backend] Primary disease:", input.primaryDisease);

    const storageDir = path.join(process.cwd(), "backend", "storage");
    const historyPath = path.join(storageDir, "history.json");

    // Ensure storage directory exists
    fs.mkdirSync(storageDir, { recursive: true });

    // Load existing history
    let history = [];
    if (fs.existsSync(historyPath)) {
      try {
        history = JSON.parse(fs.readFileSync(historyPath, "utf-8"));
      } catch {
        console.warn("[Backend] ⚠️ History file corrupted, recreating...");
        history = [];
      }
    }

    // Save new detection record
    const record = {
      id: input.id,
      timestamp: new Date(input.timestamp).getTime(),
      imageUri: input.imageUri,
      primaryDisease: input.primaryDisease,
      detections: input.detections,
      details: input.details,
    };

    history.push(record);

    // Write updated data
    fs.writeFileSync(historyPath, JSON.stringify(history, null, 2));

    console.log("[Backend] ✅ Detection record saved successfully.");

    return {
      success: true,
      id: input.id,
      savedAt: new Date().toISOString(),
    };
  });

export default saveDetectionProcedure;
