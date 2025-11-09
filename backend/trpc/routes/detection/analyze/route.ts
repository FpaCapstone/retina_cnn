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
      // Use system temp directory (works on Render)
      const tmpDir = process.env.TMPDIR || process.env.TMP || '/tmp';
      const backendTmpDir = path.join(tmpDir, 'retina-backend');
      if (!fs.existsSync(backendTmpDir)) {
        try {
          fs.mkdirSync(backendTmpDir, { recursive: true });
        } catch (err) {
          console.warn("[Backend] Could not create tmp dir, using system tmp:", err);
        }
      }

      imagePath = path.join(backendTmpDir, `upload_${Date.now()}.jpg`);
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
    
    // Check if model exists, try fallback locations
    let finalModelPath = modelPath;
    if (!fs.existsSync(modelPath)) {
      // Try assets location as fallback
      const assetsModelPath = path.join(
        process.cwd(),
        "assets",
        "images",
        "models",
        "outer_eye_mobilenetv2.h5"
      );
      if (fs.existsSync(assetsModelPath)) {
        finalModelPath = assetsModelPath;
        console.log("[Backend] Using model from assets:", finalModelPath);
      } else {
        console.error("[Backend] Model not found in backend/models or assets/images/models");
        return Promise.reject(new Error("Model file not found. Please ensure outer_eye_mobilenetv2.h5 exists in backend/models/ or assets/images/models/"));
      }
    }

    // Check if script exists
    if (!fs.existsSync(scriptPath)) {
      console.error("[Backend] Python script not found:", scriptPath);
      return Promise.reject(new Error("Python prediction script not found. Please ensure python-scripts/predict_outer_eye.py exists."));
    }

    console.log("[Backend] Running prediction via Python...");
    console.log("[Backend] Script:", scriptPath);
    console.log("[Backend] Model:", finalModelPath);

    return new Promise((resolve, reject) => {
      const py = spawn(pythonPath, [scriptPath, imagePath, finalModelPath]);

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
            // 💾 Save prediction result to history (optional, skip if fails)
            // -----------------------------
            try {
              const historyPath = path.join(
                process.cwd(),
                "backend",
                "storage",
                "history.json"
              );
              const historyDir = path.dirname(historyPath);
              if (!fs.existsSync(historyDir)) {
                fs.mkdirSync(historyDir, { recursive: true });
              }

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
            } catch (historyError) {
              console.warn("[Backend] Could not save history (non-critical):", historyError);
              // Continue anyway - history saving is optional
            }
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
