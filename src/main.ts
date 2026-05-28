import path from "node:path";
import { getLlama } from "node-llama-cpp";
import { getInputData, showMenu } from "./cli.ts";
import type { MenuResult } from "./cli.ts";
import { PdfProcessor } from "./services/processPdf.ts";
import { QueryProcessor } from "./services/processQuery.ts";
import { MarkdownProcessor } from "./services/processMarkdown.ts";
import { cleanup, setupCleanupHandlers } from "./services/cleanup.service";

process.loadEnvFile(path.join(process.cwd(), ".env"));

async function processAction(
  llama: Awaited<ReturnType<typeof getLlama>>,
  pdfPath: string | null,
  query: string | null,
  mdPath?: string | null,
): Promise<void> {
  if (query && !pdfPath && !mdPath) {
    const processor = new QueryProcessor(llama);
    await processor.processQuery(query);
  } else if (pdfPath) {
    const processor = new PdfProcessor(llama);
    await processor.processPDF(pdfPath);
  } else if (mdPath) {
    const processor = new MarkdownProcessor(llama);
    await processor.processMarkdown(mdPath);
  }
}
async function main() {
  try {
    // Setup cleanup handlers
    setupCleanupHandlers();

    const llama = await getLlama({ gpu: "metal" });

    const hasArgs = process.argv.length >= 3;

    if (hasArgs) {
      // Single run mode with command-line arguments
      const { pdfPath, query } = await getInputData();
      await processAction(llama, pdfPath, query);
    } else {
      // Interactive menu loop
      console.log(
        "\n⚠ No command line arguments detected, using interactive menu",
      );
      console.log("  Usage: npm start -- --pdf <path> --query <query>");
      console.log("  Or:    npm start -- <pdf-path> [query]");

      let running = true;
      while (running) {
        try {
          const choice: MenuResult = await showMenu();

          if ("quit" in choice) {
            running = false;
            console.log("\nGoodbye!");
            break;
          }

          const pdfPath = "pdfPath" in choice ? choice.pdfPath : null;
          const query = "query" in choice ? choice.query : null;
          const mdPath = "mdPath" in choice ? choice.mdPath : null;

          await processAction(llama, pdfPath, query, mdPath);
          console.log("\n" + "=".repeat(50));
          console.log("Action completed. Returning to menu...");
        } catch (error) {
          console.error("\n✖ Error:", (error as Error).message);
          console.log("Returning to menu...");
        }
      }
      await cleanup();
    }
  } catch (error) {
    console.error("Fatal Error:", error);
    await cleanup();
    process.exit(1);
  }
}

main();
