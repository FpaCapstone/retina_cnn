import { publicProcedure } from "../../../create-context";
import { z } from "zod";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";

const analyzeInputSchema = z.object({
  imageUri: z.string(), // base64 or file URI
});

export const analyzeProcedure = publicProcedure
  .input(analyzeInputSchema)
  .mutation(async ({ input }) => {
    console.log("[Backend] 🧠 Multi-class analysis request received");

    // Convert base64 URI to a temporary file if necessary
    let imagePath = input.imageUri;

    if (input.imageUri.startsWith("data:image")) {
      console.log("[Backend] Decoding base64 image...");
      const base64Data = input.imageUri.replace(/^data:image\/\w+;base64,/, "");
      const buffer = Buffer.from(base64Data, "base64");
      const tmpDir = path.join(process.cwd(), "backend", "tmp");
      if (!fs.existsSync(tmpDir)) fs.mkdirSync(tmpDir, { recursive: true });

      imagePath = path.join(tmpDir, `upload_${Date.now()}.jpg`);
      fs.writeFileSync(imagePath, buffer);
    }

    // Path to Python script and model
    const pythonPath = "python3"; // change if using virtualenv
    const scriptPath = path.join(
      process.cwd(),
      "python-scripts",
      "predict_outer_eye.py"
    );
    const modelPath = path.join(
      process.cwd(),
      "backend",
      "models",
      "outer_eye_mobilenetv2.h5"
    );

    console.log("[Backend] Running prediction via Python...");

    return new Promise((resolve, reject) => {
      const py = spawn(pythonPath, [scriptPath, imagePath, modelPath]);

      let resultData = "";
      let errorData = "";

      py.stdout.on("data", (data) => {
        resultData += data.toString();
      });

      py.stderr.on("data", (data) => {
        errorData += data.toString();
      });

      py.on("close", (code) => {
        if (code !== 0) {
          console.error("[Backend] ❌ Prediction script error:", errorData);
          reject(new Error("Prediction failed: " + errorData));
        } else {
          console.log("[Backend] ✅ Prediction complete");
          try {
            const parsed = JSON.parse(resultData);

            // -----------------------------
            // 💾 Save prediction result to history
            // -----------------------------
            const historyPath = path.join(
              process.cwd(),
              "backend",
              "storage",
              "history.json"
            );
            fs.mkdirSync(path.dirname(historyPath), { recursive: true });

            let history = [];
            if (fs.existsSync(historyPath)) {
              history = JSON.parse(fs.readFileSync(historyPath, "utf-8"));
            }

            history.push({
              timestamp: Date.now(),
              image: path.basename(imagePath),
              prediction: parsed.prediction,
              confidence: parsed.confidence,
            });

            fs.writeFileSync(historyPath, JSON.stringify(history, null, 2));
            // -----------------------------

            resolve(parsed);
          } catch (e) {
            console.error("[Backend] ❗ Could not parse JSON:", resultData);
            reject(new Error("Invalid output from Python script"));
          }
        }
      });
    });
  });

export default analyzeProcedure;
