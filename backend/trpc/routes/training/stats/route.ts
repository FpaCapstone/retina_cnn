import { publicProcedure } from "../../../create-context";
import fs from "fs";
import path from "path";

export const trainingStatsProcedure = publicProcedure.query(async () => {
  console.log("[Backend] Fetching live training statistics...");

  const trainingRoot = path.join(process.cwd(), "backend", "storage", "training");
  const metadataPath = path.join(trainingRoot, "training_metadata.json");

  // Default structure
  const diseases = [
    "Normal",
    "Uveitis",
    "Conjunctivitis",
    "Cataract",
    "Eyelid Drooping",
  ];

  let stats = diseases.map((disease) => ({
    disease,
    totalTrainingImages: 0,
    validatedImages: 0,
    accuracy: 0,
    lastUpdated: new Date().toISOString(),
  }));

  // If metadata exists, parse it
  if (fs.existsSync(metadataPath)) {
    try {
      const data = JSON.parse(fs.readFileSync(metadataPath, "utf-8"));

      for (const entry of data) {
        const record = stats.find((s) => s.disease === entry.disease);
        if (record) {
          record.totalTrainingImages += 1;
          if (entry.validated) record.validatedImages += 1;
        }
      }

      // Simple simulated accuracy estimate
      for (const s of stats) {
        s.accuracy = s.validatedImages
          ? Number((0.75 + s.validatedImages / (s.totalTrainingImages * 10)).toFixed(2))
          : 0.7;
        s.lastUpdated = new Date().toISOString();
      }
    } catch (err) {
      console.error("[Backend] ⚠️ Failed to parse training metadata:", err);
    }
  }

  // Return structured response
  return { stats };
});

export default trainingStatsProcedure;
