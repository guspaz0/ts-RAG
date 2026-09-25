import path from "node:path";
import { getLlama } from "node-llama-cpp";
import { getInputData, showMenu, printCatalogs } from "./cli.ts";
import type { MenuResult } from "./cli.ts";
import { PdfProcessor } from "./services/processPdf.ts";
import { QueryProcessor } from "./services/processQuery.ts";
import { MarkdownProcessor } from "./services/processMarkdown.ts";
import { cleanup, setupCleanupHandlers } from "./services/cleanup.service";
import { createEmbeddingStore } from "./store/embedding-store";
import type { Catalog } from "./store/catalogs";

process.loadEnvFile(path.join(process.cwd(), ".env"));

async function processAction(
  llama: Awaited<ReturnType<typeof getLlama>>,
  pdfPath: string | null,
  query: string | null,
  mdPath?: string | null,
  catalog?: Catalog | null,
): Promise<void> {
  if (query && !pdfPath && !mdPath) {
    const processor = new QueryProcessor(llama);
    await processor.processQuery(query, undefined, catalog);
  } else if (pdfPath) {
    const processor = new PdfProcessor(llama);
    await processor.processPDF(pdfPath, undefined, catalog ?? undefined);
  } else if (mdPath) {
    const processor = new MarkdownProcessor(llama);
    await processor.processMarkdown(mdPath, undefined, catalog ?? undefined);
  }
}

async function main() {
  try {
    // Setup cleanup handlers
    setupCleanupHandlers();

    const llama = await getLlama({ gpu: "auto" });

    const hasArgs = process.argv.length >= 3;

    if (hasArgs) {
      // Single run mode with command-line arguments
      const { pdfPath, query, mdPath, catalog, listCatalogs, createCatalog } =
        await getInputData();

      // Catalog management commands (no model needed)
      if (listCatalogs || createCatalog) {
        const store = await createEmbeddingStore();
        if (createCatalog) {
          const created = await store.catalogStore.createCatalog(createCatalog);
          console.log(`✓ Created catalog "${created.name}"`);
        }
        await printCatalogs(store);
        await cleanup();
        return;
      }

      // Resolve catalog name to catalog object
      let resolvedCatalog: Catalog | null | undefined;
      if (catalog) {
        const store = await createEmbeddingStore();
        const found = await store.catalogStore.getCatalog(catalog);
        if (!found) {
          console.error(`✖ Catalog "${catalog}" not found`);
          console.error("  Use --list-catalogs to see available catalogs");
          process.exit(1);
        }
        resolvedCatalog = found;
      }

      await processAction(llama, pdfPath, query, mdPath, resolvedCatalog);
    } else {
      // Interactive menu loop
      console.log(
        "\n⚠ No command line arguments detected, using interactive menu",
      );
      console.log("  Usage: npm start -- --pdf <path> --query <query> --catalog <name>");
      console.log("  Or:    npm start -- <pdf-path> [query]");
      console.log("  Also:  npm start -- --list-catalogs | --create-catalog <name>");

      // The menu needs a store to manage catalogs
      const store = await createEmbeddingStore();

      let running = true;
      while (running) {
        try {
          const choice: MenuResult = await showMenu(store);

          if ("quit" in choice) {
            running = false;
            console.log("\nGoodbye!");
            break;
          }

          if ("listCatalogs" in choice) {
            await printCatalogs(store);
            continue;
          }

          const pdfPath = "pdfPath" in choice ? choice.pdfPath : null;
          const query = "query" in choice ? choice.query : null;
          const mdPath = "mdPath" in choice ? choice.mdPath : null;
          const catalog = "catalog" in choice ? choice.catalog : null;

          await processAction(llama, pdfPath, query, mdPath, catalog);
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
