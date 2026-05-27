// Import necessary modules
import { execSync } from "child_process";
import path from "node:path";
import readline from "node:readline/promises";

// Function to display CLI menu and get user choice
export type MenuResult =
  | { query: string }
  | { pdfPath: string }
  | { mdPath: string }
  | { quit: true };

export async function showMenu(): Promise<MenuResult> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  try {
    console.log("\n📋 Welcome to the PDF Embeddings CLI");
    console.log("=".repeat(50));
    console.log("\t\x1b[15m1\x1b[0m. Query existing database embeddings");
    console.log("\t\x1b[15m2\x1b[0m. Process a PDF to create new embeddings");
    console.log("\t\x1b[15m3\x1b[0m. Process a Markdown file to create new embeddings");
    console.log("\t\x1b[15m4\x1b[0m. Quit");
    console.log("");

    const choice = await rl.question("\x1b[35mPlease select an option:\x1b[0m");

    switch (choice) {
      case "1":
        const query = await rl.question("Please enter your query: ");
        return { query };
      case "2":
        const userPath = execSync(
          process.platform == "win32" ? "echo %USERPROFILE%" : "echo $HOME",
        )
          .toString()
          .replace("\n", "");
        const pdfPath = await rl.question(
          "Please enter the path to your PDF: \n \t" + userPath,
        );
        return { pdfPath: path.join(userPath, pdfPath) };
      case "3":
        const mdUserPath = execSync(
          process.platform == "win32" ? "echo %USERPROFILE%" : "echo $HOME",
        )
          .toString()
          .replace("\n", "");
        const mdPath = await rl.question(
          "Please enter the path to your Markdown file: \n \t" + mdUserPath,
        );
        return { mdPath: path.join(mdUserPath, mdPath) };
      case "4":
        return { quit: true };
      default:
        console.log("Invalid choice");
        return showMenu();
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
} {
  const args = process.argv.slice(2);
  let pdfPath: string | null = null;
  let query: string | null = null;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--pdf") {
      pdfPath = args[i + 1] as string;
      i++; // Skip next argument as it's the value
    } else if (args[i] === "--query") {
      query = args[i + 1] as string;
      i++; // Skip next argument as it's the value
    }
  }
  // If no --pdf or --query provided, check for positional arguments
  if (!pdfPath && !query) {
    pdfPath = process.argv[2] as string;
    query = process.argv[3] as string;
  }
  return { pdfPath, query };
}

export async function getInputData(): Promise<{
  pdfPath: string | null;
  query: string | null;
}> {
  try {
    let pdfPath: string | null = null;
    let query: string | null = null;
    if (process.argv.length < 3) {
      return { pdfPath: null, query: null };
    }
    const sysArgs = processSysArgs();
    if (sysArgs["pdfPath"]) pdfPath = sysArgs["pdfPath"] as string;
    if (sysArgs["query"]) query = sysArgs["query"] as string;
    return { pdfPath, query };
  } catch (error) {
    console.error("Error reading input:", error);
    throw error;
  }
}
