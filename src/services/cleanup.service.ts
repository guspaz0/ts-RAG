import { getPostgresDaemon } from "../store/embedding-store";

let isCleanupDone = false;

export async function cleanup() {
  if (isCleanupDone) return;
  isCleanupDone = true;

  try {
    const postgresDaemon = getPostgresDaemon();
    if (postgresDaemon) {
      console.log("\n📦 Stopping PostgreSQL daemon...");
      await postgresDaemon.stop();
      console.log("✅ PostgreSQL daemon stopped successfully");
    }
  } catch (error) {
    console.error("Error during cleanup:", error);
  }
}

export function setupCleanupHandlers() {
  process.on("SIGINT", async () => {
    console.log("\n🛑 Received SIGINT, shutting down gracefully...");
    await cleanup();
    process.exit(0);
  });

  process.on("SIGTERM", async () => {
    console.log("\n🛑 Received SIGTERM, shutting down gracefully...");
    await cleanup();
    process.exit(0);
  });

  process.on("SIGABRT", async () => {
    console.log("\n🛑 Received SIGABRT, cleaning up...");
    const daemon = getPostgresDaemon();
    if (daemon?.process?.pid) {
      try { daemon.process.kill("SIGTERM"); } catch { /* ignore */ }
    }
    await cleanup();
    process.exit(1);
  });

  process.on("uncaughtException", async (error) => {
    console.error("🚨 Uncaught Exception:", error);
    await cleanup();
    process.exit(1);
  });

  process.on("unhandledRejection", async (reason, promise) => {
    console.error("🚨 Unhandled Rejection at:", promise, "reason:", reason);
    await cleanup();
    process.exit(1);
  });

  process.on("exit", () => {
    const daemon = getPostgresDaemon();
    if (daemon?.process?.pid) {
      try {
        daemon.process.kill("SIGTERM");
      } catch { /* process may already be dead */ }
    }
  });
}
