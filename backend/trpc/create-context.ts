import { FetchCreateContextFnOptions } from "@trpc/server/adapters/fetch";
import { initTRPC } from "@trpc/server";
import superjson from "superjson";
import fs from "fs";
import path from "path";

// -------------------------------------
// 🌍 Create per-request context
// -------------------------------------
export const createContext = async (opts: FetchCreateContextFnOptions) => {
  const request = opts.req;
  const ip = request.headers.get("x-forwarded-for") || "127.0.0.1";
  const userAgent = request.headers.get("user-agent") || "unknown";

  // Simple structured logger
  const logDir = path.join(process.cwd(), "backend", "storage", "logs");
  fs.mkdirSync(logDir, { recursive: true });

  const log = (msg: string) => {
    const timestamp = new Date().toISOString();
    const line = `[${timestamp}] ${msg}\n`;
    fs.appendFileSync(path.join(logDir, "server.log"), line);
    console.log(line.trim());
  };

  return {
    req: request,
    ip,
    userAgent,
    log,
  };
};

export type Context = Awaited<ReturnType<typeof createContext>>;

// -------------------------------------
// ⚙️ Initialize TRPC
// -------------------------------------
const t = initTRPC.context<Context>().create({
  transformer: superjson,
});

// Reusable exports for route definitions
export const createTRPCRouter = t.router;
export const publicProcedure = t.procedure;
