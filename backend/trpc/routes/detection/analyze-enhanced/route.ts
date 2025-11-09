import { publicProcedure } from "../../../create-context";
import { z } from "zod";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";

const analyzeInputSchema = z.object({
  imageUri: z.string(), // base64 or file URI
  enableStages: z.object({
    quality_check: z.boolean().optional().default(true),
    preprocessing: z.boolean().optional().default(true),
    normal_filter: z.boolean().optional().default(true),
    disease_classification: z.boolean().optional().default(true),
    validation: z.boolean().optional().default(true),
  }).optional(),
});

export const analyzeEnhancedProcedure = publicProcedure
  .input(analyzeInputSchema)
  .mutation(async ({ input }) => {
    console.log("[Backend] 🚀 Enhanced 5-stage pipeline analysis request received");

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

    // Path to Python script
    const pythonPath = "python3";
    const scriptPath = path.join(
      process.cwd(),
      "python-scripts",
      "enhanced_pipeline.py"
    );
    
    // Check if script exists
    if (!fs.existsSync(scriptPath)) {
      console.error("[Backend] Enhanced pipeline script not found:", scriptPath);
      return Promise.reject(new Error("Enhanced pipeline script not found. Please ensure python-scripts/enhanced_pipeline.py exists."));
    }

    console.log("[Backend] Running enhanced pipeline via Python...");
    console.log("[Backend] Script path:", scriptPath);
    console.log("[Backend] Image path:", imagePath);

    return new Promise((resolve, reject) => {
      const args = [scriptPath, imagePath];
      
      // Add stage enable/disable flags if provided
      if (input.enableStages) {
        if (input.enableStages.quality_check === false) {
          args.push("--disable-stage", "quality_check");
        }
        if (input.enableStages.preprocessing === false) {
          args.push("--disable-stage", "preprocessing");
        }
        if (input.enableStages.normal_filter === false) {
          args.push("--disable-stage", "normal_filter");
        }
        if (input.enableStages.disease_classification === false) {
          args.push("--disable-stage", "disease_classification");
        }
        if (input.enableStages.validation === false) {
          args.push("--disable-stage", "validation");
        }
      }

      const py = spawn(pythonPath, args);
      let stdout = "";
      let stderr = "";

      py.stdout.on("data", (data) => {
        stdout += data.toString();
      });

      py.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      py.on("close", (code) => {
        // Clean up temp file
        try {
          if (imagePath && (imagePath.startsWith('/tmp') || imagePath.includes('tmp')) && fs.existsSync(imagePath)) {
            fs.unlinkSync(imagePath);
          }
        } catch (cleanupError) {
          // Ignore cleanup errors
        }
        
        if (code !== 0) {
          console.error("[Backend] Enhanced pipeline error:", stderr);
          // Provide more helpful error messages
          if (stderr.includes("No such file") || stderr.includes("ENOENT")) {
            reject(new Error(`File not found. Check that enhanced_pipeline.py exists and all dependencies are installed. Error: ${stderr}`));
          } else if (stderr.includes("ModuleNotFoundError") || stderr.includes("ImportError")) {
            reject(new Error(`Python dependencies missing. Install with: pip3 install -r python-scripts/requirements.txt. Error: ${stderr}`));
          } else {
            reject(new Error(`Enhanced pipeline failed: ${stderr || "Unknown error"}`));
          }
          return;
        }

        try {
          // Parse JSON output from the pipeline
          // The script outputs JSON to stdout (print statements go to stderr)
          const lines = stdout.trim().split('\n');
          
          // Find the JSON object (should be the last complete JSON in output)
          let jsonStr = '';
          let braceCount = 0;
          let jsonStart = -1;
          
          for (let i = lines.length - 1; i >= 0; i--) {
            const line = lines[i];
            if (line.includes('{')) {
              jsonStart = i;
              break;
            }
          }
          
          if (jsonStart >= 0) {
            // Extract JSON from this line onwards
            jsonStr = lines.slice(jsonStart).join('\n');
          } else {
            // Try to parse entire stdout
            jsonStr = stdout.trim();
          }
          
          // Clean up: remove any non-JSON prefix
          const jsonMatch = jsonStr.match(/\{[\s\S]*\}/);
          if (!jsonMatch) {
            throw new Error("No JSON found in output");
          }
          
          const result = JSON.parse(jsonMatch[0]);
          
          // Convert to frontend format
          const finalPrediction = result.final_prediction;
          
          // Handle case where recommendation is 'retake' (no prediction)
          if (result.recommendation === 'retake' && !finalPrediction) {
            resolve({
              prediction: 'normal',
              confidence: 0.0,
              quality_score: result.stages?.quality?.quality_score || 0.0,
              recommendation: 'retake',
              reason: result.reason || 'Image quality too low',
              stages: result.stages,
              all_probabilities: {},
            });
            return;
          }
          
          if (!finalPrediction) {
            reject(new Error("No prediction from enhanced pipeline"));
            return;
          }

          // Map disease names (handle "Eyelid Drooping" -> "eyelid_drooping")
          let diseaseName = finalPrediction.predicted_disease || finalPrediction.disease;
          if (diseaseName === "Eyelid Drooping") {
            diseaseName = "eyelid_drooping";
          } else {
            diseaseName = diseaseName.toLowerCase().replace(" ", "_");
          }

          resolve({
            prediction: diseaseName,
            confidence: result.final_confidence || finalPrediction.confidence,
            quality_score: result.stages?.quality?.quality_score,
            recommendation: result.recommendation,
            stages: result.stages,
            all_probabilities: finalPrediction.probabilities || finalPrediction.all_predictions,
          });
        } catch (error) {
          console.error("[Backend] Failed to parse enhanced pipeline output:", error);
          console.error("[Backend] stdout:", stdout);
          console.error("[Backend] stderr:", stderr);
          reject(new Error(`Failed to parse enhanced pipeline results: ${error}`));
        }
      });

      py.on("error", (error) => {
        console.error("[Backend] Enhanced pipeline spawn error:", error);
        // Check if it's a file not found error
        if (error.message.includes("ENOENT") || error.message.includes("no such file")) {
          reject(new Error(`Python or script not found. Make sure Python 3 is installed and enhanced_pipeline.py exists at: ${scriptPath}`));
        } else {
          reject(new Error(`Failed to start enhanced pipeline: ${error.message}`));
        }
      });
      
      // Clean up temp file after processing (optional, but good practice)
      if (imagePath.startsWith('/tmp') || imagePath.includes('tmp')) {
        py.on("close", () => {
          try {
            if (fs.existsSync(imagePath)) {
              fs.unlinkSync(imagePath);
            }
          } catch (cleanupError) {
            // Ignore cleanup errors
          }
        });
      }
    });
  });

