// Import necessary modules
import { execSync } from "child_process";
import path from "node:path";
import readline from "node:readline/promises";
import type { EmbeddingStore } from "./store/embedding-store";
import {
  promptCatalogSelection,
  promptCatalogForQuery,
  type Catalog,
} from "./store/catalogs";

// Function to display CLI menu and get user choice
export type MenuResult =
  | { query: string; catalog: Catalog | null }
  | { pdfPath: string; catalog: Catalog }
  | { mdPath: string; catalog: Catalog }
  | { listCatalogs: true }
  | { quit: true };

export async function showMenu(store: EmbeddingStore): Promise<MenuResult> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  try {
    console.log("\n📋 Welcome to the RAG CLI");
    console.log("=".repeat(50));
    console.log("\t\x1b[15m1\x1b[0m. Query existing database embeddings");
    console.log("\t\x1b[15m2\x1b[0m. Process a PDF to create new embeddings");
    console.log("\t\x1b[15m3\x1b[0m. Process a Markdown file to create new embeddings");
    console.log("\t\x1b[15m4\x1b[0m. List catalogs");
    console.log("\t\x1b[15m5\x1b[0m. Quit");
    console.log("");

    const choice = await rl.question("\x1b[35mPlease select an option:\x1b[0m");

    switch (choice) {
      case "1": {
        const query = await rl.question("Please enter your query: ");
        const catalog = await promptCatalogForQuery(store.catalogStore);
        return { query, catalog };
      }
      case "2": {
        const userPath = execSync(
          process.platform == "win32" ? "echo %USERPROFILE%" : "echo $HOME",
        )
          .toString()
          .replace("\n", "");
        const pdfPath = await rl.question(
          "Please enter the path to your PDF: \n \t" + userPath,
        );
        const catalog = await promptCatalogSelection(store.catalogStore, "store");
        return { pdfPath: path.join(userPath, pdfPath), catalog };
      }
      case "3": {
        const mdUserPath = execSync(
          process.platform == "win32" ? "echo %USERPROFILE%" : "echo $HOME",
        )
          .toString()
          .replace("\n", "");
        const mdPath = await rl.question(
          "Please enter the path to your Markdown file: \n \t" + mdUserPath,
        );
        const catalog = await promptCatalogSelection(store.catalogStore, "store");
        return { mdPath: path.join(mdUserPath, mdPath), catalog };
      }
      case "4": {
        return { listCatalogs: true };
      }
      case "5":
        return { quit: true };
      default:
        console.log("Invalid choice");
        return showMenu(store);
    }
  } catch (error) {
    console.error("Error reading input:", error);
    throw error;
  } finally {
    rl.close();
  }
}

export function processSysArgs(): {
  pdfPath: string | null;
  query: string | null;
  mdPath: string | null;
  catalog: string | null;
  listCatalogs: boolean;
  createCatalog: string | null;
} {
  const args = process.argv.slice(2);
  let pdfPath: string | null = null;
  let query: string | null = null;
  let mdPath: string | null = null;
  let catalog: string | null = null;
  let listCatalogs = false;
  let createCatalog: string | null = null;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--pdf") {
      pdfPath = args[i + 1] as string;
      i++;
    } else if (args[i] === "--md") {
      mdPath = args[i + 1] as string;
      i++;
    } else if (args[i] === "--query") {
      query = args[i + 1] as string;
      i++;
    } else if (args[i] === "--catalog") {
      catalog = args[i + 1] as string;
      i++;
    } else if (args[i] === "--list-catalogs") {
      listCatalogs = true;
    } else if (args[i] === "--create-catalog") {
      createCatalog = args[i + 1] as string;
      i++;
    }
  }
  // If no --pdf/--md or --query provided, check for positional arguments
  if (!pdfPath && !mdPath && !query) {
    pdfPath = process.argv[2] as string;
    query = process.argv[3] as string;
  }
  return { pdfPath, query, mdPath, catalog, listCatalogs, createCatalog };
}

export async function getInputData(): Promise<{
  pdfPath: string | null;
  query: string | null;
  mdPath: string | null;
  catalog: string | null;
  listCatalogs: boolean;
  createCatalog: string | null;
}> {
  try {
    let pdfPath: string | null = null;
    let query: string | null = null;
    let mdPath: string | null = null;
    let catalog: string | null = null;
    let listCatalogs = false;
    let createCatalog: string | null = null;
    if (process.argv.length < 3) {
      return { pdfPath, query, mdPath, catalog, listCatalogs, createCatalog };
    }
    const sysArgs = processSysArgs();
    pdfPath = sysArgs.pdfPath;
    query = sysArgs.query;
    mdPath = sysArgs.mdPath;
    catalog = sysArgs.catalog;
    listCatalogs = sysArgs.listCatalogs;
    createCatalog = sysArgs.createCatalog;
    return { pdfPath, query, mdPath, catalog, listCatalogs, createCatalog };
  } catch (error) {
    console.error("Error reading input:", error);
    throw error;
  }
}

/**
 * Print all catalogs with their embedding counts.
 */
export async function printCatalogs(store: EmbeddingStore): Promise<void> {
  const catalogs = await store.catalogStore.listCatalogs();
  console.log("\n📚 Catalogs:");
  console.log("─".repeat(60));
  if (catalogs.length === 0) {
    console.log("  (no catalogs yet)");
    return;
  }
  for (const catalog of catalogs) {
    const count =
      catalog.embedding_count !== undefined
        ? catalog.embedding_count
        : await store.catalogStore.countEmbeddings(catalog.id);
    const desc = catalog.description ? ` — ${catalog.description}` : "";
    console.log(`  • ${catalog.name} [${count} embeddings]${desc}`);
  }
  console.log("");
}
