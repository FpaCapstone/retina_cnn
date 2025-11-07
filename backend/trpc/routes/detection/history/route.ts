import { publicProcedure } from "../../../create-context";
import { z } from "zod";
import fs from "fs";
import path from "path";

// Supported disease categories
const diseaseTypeSchema = z.enum([
  "Normal",
  "Uveitis",
  "Conjunctivitis",
  "Cataract",
  "Eyelid Drooping",
]);

const historyInputSchema = z.object({
  disease: diseaseTypeSchema.optional(),
  limit: z.number().default(20),
});

export const historyProcedure = publicProcedure
  .input(historyInputSchema)
  .query(async ({ input }) => {
    console.log("[Backend] 📜 Fetching detection history");
    console.log("[Backend] Filter:", input.disease || "All diseases");

    const historyPath = path.join(process.cwd(), "backend", "storage", "history.json");

    // Ensure storage file exists
    if (!fs.existsSync(historyPath)) {
      fs.mkdirSync(path.dirname(historyPath), { recursive: true });
      fs.writeFileSync(historyPath, JSON.stringify([]));
    }

    // Read stored results
    const data = JSON.parse(fs.readFileSync(historyPath, "utf-8"));

    // Filter and sort
    let filtered = data;
    if (input.disease) {
      filtered = data.filter(
        (r: any) => r.prediction === input.disease
      );
    }

    const sorted = filtered.sort(
      (a: any, b: any) => b.timestamp - a.timestamp
    );

    return {
      records: sorted.slice(0, input.limit),
      total: sorted.length,
    };
  });

export default historyProcedure;
