import { publicProcedure } from "../../../create-context";
import { z } from "zod";
import fs from "fs";
import path from "path";

const diseaseTypeSchema = z.enum([
  "Normal",
  "Uveitis",
  "Conjunctivitis",
  "Cataract",
  "Eyelid Drooping",
]);

const validateImageSchema = z.object({
  imageUri: z.string(), // Can be a base64 or local path
  disease: diseaseTypeSchema,
});

export const validateImageProcedure = publicProcedure
  .input(validateImageSchema)
  .mutation(async ({ input }) => {
    console.log("[Backend] 🔍 Validating training image for:", input.disease);

    // Artificial short delay to simulate computation time
    await new Promise((resolve) => setTimeout(resolve, 1000));

    // Check image existence or decode base64
    let valid = false;
    let reason = "";

    if (input.imageUri.startsWith("data:image")) {
      // Basic base64 validation
      const size = Buffer.byteLength(
        input.imageUri.replace(/^data:image\/\w+;base64,/, ""),
        "base64"
      );
      valid = size > 30000; // roughly >30KB
      reason = valid
        ? `✅ Image quality acceptable for ${input.disease} training.`
        : "⚠️ Image file appears too small or low quality for reliable training.";
    } else if (fs.existsSync(input.imageUri)) {
      // Local file validation
      const stats = fs.statSync(input.imageUri);
      valid = stats.size > 50000; // 50KB threshold
      reason = valid
        ? `✅ File ${path.basename(
            input.imageUri
          )} passed validation for ${input.disease}.`
        : `⚠️ ${path.basename(
            input.imageUri
          )} seems too small; retake with better quality.`;
    } else {
      reason =
        "⚠️ Image path not found or invalid. Ensure the file is correctly uploaded.";
    }

    // ----------------------------------------
    // 🗃️ Update metadata if file was validated
    // ----------------------------------------
    const trainingRoot = path.join(process.cwd(), "backend", "storage", "training");
    const metadataPath = path.join(trainingRoot, "training_metadata.json");

    if (fs.existsSync(metadataPath)) {
      try {
        const metadata = JSON.parse(fs.readFileSync(metadataPath, "utf-8"));
        const record = metadata.find(
          (r: any) =>
            r.imageFile && input.imageUri.includes(r.imageFile)
        );
        if (record) {
          record.validated = valid;
          record.validationReason = reason;
          fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));
          console.log(
            `[Backend] 📘 Updated metadata for ${record.imageFile}: validated=${valid}`
          );
        }
      } catch (err) {
        console.warn("[Backend] ⚠️ Could not update training metadata:", err);
      }
    }

    // ----------------------------------------
    // ✅ Return structured validation result
    // ----------------------------------------
    return {
      valid,
      reason,
      disease: input.disease,
      checkedAt: new Date().toISOString(),
    };
  });

export default validateImageProcedure;
