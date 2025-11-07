import { publicProcedure } from "../../../create-context";
import { z } from "zod";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";

// Input schema — optional hyperparameters
const retrainInputSchema = z.object({
  epochs: z.number().default(10),
  learningRate: z.number().default(0.0005),
  note: z.string().optional(),
});

export const retrainModelProcedure = publicProcedure
  .input(retrainInputSchema)
  .mutation(async ({ input }) => {
    console.log("[Backend] 🧠 Retraining model...");
    console.log(`[Backend] Params: epochs=${input.epochs}, lr=${input.learningRate}`);

    const scriptPath = path.join(
      process.cwd(),
      "python-scripts",
      "train_outer_eye_mobilenetv2.py"
    );
    if (!fs.existsSync(scriptPath)) {
      throw new Error(`Training script not found at: ${scriptPath}`);
    }

    // ----------------------------------------
    // 📁 STEP 1: Prepare datasets from validated samples
    // ----------------------------------------
    const metadataPath = path.join(process.cwd(), "backend", "storage", "training", "training_metadata.json");
    const datasetRoot = path.join(process.cwd(), "python-scripts", "datasets");

    if (fs.existsSync(metadataPath)) {
      console.log("[Backend] Preparing datasets for retraining...");
      const metadata = JSON.parse(fs.readFileSync(metadataPath, "utf-8"));
      fs.mkdirSync(datasetRoot, { recursive: true });

      const diseases = ["Normal", "Uveitis", "Conjunctivitis", "Cataract", "Eyelid Drooping"];
      for (const d of diseases) {
        const folder = path.join(datasetRoot, d);
        fs.rmSync(folder, { recursive: true, force: true });
        fs.mkdirSync(folder, { recursive: true });
      }

      const validSamples = metadata.filter((m: any) => m.validated === true);
      for (const sample of validSamples) {
        const srcPath = path.join(
          process.cwd(),
          "backend",
          "storage",
          "training",
          sample.imageFile || ""
        );
        const destPath = path.join(datasetRoot, sample.disease, path.basename(srcPath));
        if (fs.existsSync(srcPath)) {
          fs.copyFileSync(srcPath, destPath);
        }
      }

      console.log(`[Backend] ✅ Copied ${validSamples.length} validated images into datasets/`);
    } else {
      console.warn("[Backend] ⚠️ No training metadata found. Skipping dataset prep.");
    }

    // ----------------------------------------
    // 📄 STEP 2: Create logs
    // ----------------------------------------
    const logsDir = path.join(process.cwd(), "backend", "storage", "logs");
    fs.mkdirSync(logsDir, { recursive: true });

    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const logPath = path.join(logsDir, `retrain_${timestamp}.log`);

    // ----------------------------------------
    // 🧩 STEP 3: Spawn Python training process
    // ----------------------------------------
    const pythonProcess = spawn("python3", [scriptPath], {
      env: {
        ...process.env,
        EPOCHS: String(input.epochs),
        LEARNING_RATE: String(input.learningRate),
      },
    });

    let stdoutLogs = "";
    let stderrLogs = "";

    pythonProcess.stdout.on("data", (data) => {
      const msg = data.toString();
      stdoutLogs += msg;
      console.log(`[Python] ${msg.trim()}`);
    });

    pythonProcess.stderr.on("data", (data) => {
      const msg = data.toString();
      stderrLogs += msg;
      console.error(`[Python ERROR] ${msg.trim()}`);
    });

    const exitCode = await new Promise<number>((resolve) => {
      pythonProcess.on("close", resolve);
    });

    fs.writeFileSync(
      logPath,
      `=== Retraining Log (${timestamp}) ===\n\nSTDOUT:\n${stdoutLogs}\n\nSTDERR:\n${stderrLogs}`
    );

    if (exitCode !== 0) {
      throw new Error(`Training failed with exit code ${exitCode}. Check logs at ${logPath}`);
    }

    // ----------------------------------------
    // 🧠 STEP 4: Return model info
    // ----------------------------------------
    const modelDir = path.join(process.cwd(), "backend", "models");
    const kerasModel = path.join(modelDir, "outer_eye_mobilenetv2.h5");
    const tfliteModel = path.join(modelDir, "outer_eye_mobilenetv2.tflite");

    const result = {
      success: true,
      message: "Retraining completed successfully",
      savedAt: new Date().toISOString(),
      logPath,
      models: {
        keras: fs.existsSync(kerasModel) ? kerasModel : null,
        tflite: fs.existsSync(tfliteModel) ? tfliteModel : null,
      },
      note: input.note || null,
    };

    console.log("[Backend] ✅ Retraining complete!");
    return result;
  });

export default retrainModelProcedure;
