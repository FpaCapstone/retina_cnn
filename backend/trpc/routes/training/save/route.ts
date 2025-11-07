import { publicProcedure } from "../../../create-context";
import { z } from "zod";
import fs from "fs";
import path from "path";

// Supported disease classes — same as your model
const diseaseTypeSchema = z.enum([
  "Normal",
  "Uveitis",
  "Conjunctivitis",
  "Cataract",
  "Eyelid Drooping",
]);

const saveTrainingImageSchema = z.object({
  id: z.string(),
  disease: diseaseTypeSchema,
  imageUri: z.string(), // can be base64 or file path
  uploadedAt: z.string(),
  validated: z.boolean(),
  validationReason: z.string().optional(),
});

export const saveTrainingImageProcedure = publicProcedure
  .input(saveTrainingImageSchema)
  .mutation(async ({ input }) => {
    console.log("[Backend] 📦 Saving training image:", input.id);
    console.log("[Backend] Disease:", input.disease);

    // -----------------------------
    // 📁 Prepare storage structure
    // -----------------------------
    const storageRoot = path.join(process.cwd(), "backend", "storage");
    const trainingRoot = path.join(storageRoot, "training");
    const diseaseDir = path.join(trainingRoot, input.disease);

    fs.mkdirSync(diseaseDir, { recursive: true });

    // -----------------------------
    // 🖼️ Save image file
    // -----------------------------
    let imageFileName = `training_${input.id}_${Date.now()}.jpg`;
    let imagePath = path.join(diseaseDir, imageFileName);

    if (input.imageUri.startsWith("data:image")) {
      // Base64 image
      const base64Data = input.imageUri.replace(/^data:image\/\w+;base64,/, "");
      const buffer = Buffer.from(base64Data, "base64");
      fs.writeFileSync(imagePath, buffer);
    } else if (fs.existsSync(input.imageUri)) {
      // Copy file if it’s a local path
      fs.copyFileSync(input.imageUri, imagePath);
    } else {
      throw new Error("Invalid or missing imageUri");
    }

    // -----------------------------
    // 🗃️ Log metadata
    // -----------------------------
    const metadataPath = path.join(trainingRoot, "training_metadata.json");
    let metadata = [];

    if (fs.existsSync(metadataPath)) {
      try {
        metadata = JSON.parse(fs.readFileSync(metadataPath, "utf-8"));
      } catch {
        console.warn("[Backend] ⚠️ Corrupted metadata file — recreating...");
        metadata = [];
      }
    }

    const record = {
      id: input.id,
      disease: input.disease,
      imageFile: imageFileName,
      uploadedAt: input.uploadedAt,
      validated: input.validated,
      validationReason: input.validationReason || null,
    };

    metadata.push(record);
    fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));

    // -----------------------------
    // ✅ Return confirmation
    // -----------------------------
    return {
      success: true,
      id: input.id,
      savedAt: new Date().toISOString(),
      imagePath,
    };
  });

export default saveTrainingImageProcedure;
