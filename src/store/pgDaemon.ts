import * as path from "node:path";
import { spawn, execSync, ChildProcess } from "child_process";
import * as fs from "node:fs";

type Platform = "linux" | "mac" | "windows";

const PLATFORM_MAP: Record<NodeJS.Platform, Platform> = {
  darwin: "mac",
  win32: "windows",
  linux: "linux",
} as unknown as Record<NodeJS.Platform, Platform>;

export interface PostgresConfig {
  dataDir: string;
  port: number;
  host: string;
  user: string;
  password: string;
  database: string;
}

export interface PostgresServer {
  process: ChildProcess;
  stop: () => Promise<void>;
}

export class PostgresDaemon {
  #binPath: string;
  #postgres: string;
  #initdb: string;
  #proc: ChildProcess;
  config: PostgresConfig;

  constructor({
    user,
    password,
    database,
    port = 5432,
    host = "127.0.0.1",
    dataDir,
  }: PostgresConfig) {
    this.#binPath = this.#getPostgresBinPath();
    this.#proc = null as unknown as ChildProcess;
    this.config = {
      user,
      database,
      port,
      host,
      dataDir,
      password,
    };
    if (!this.config.dataDir) {
      let userPath: string = "";
      switch (PLATFORM_MAP[process.platform]) {
        case "linux":
          userPath = execSync("echo $HOME").toString().replace("\n", "");
          break;
        case "mac":
          userPath = execSync("echo $HOME").toString().replace("\n", "");
          break;
        case "windows":
          userPath = execSync("echo %USERPROFILE%")
            .toString()
            .replace("\r\n", "");
          break;
        default:
          throw new Error("Unsupported platform");
      }
      this.config.dataDir = path.join(userPath, "postgres");
    }
    this.#postgres = path.join(this.#binPath, "postgres");
    this.#initdb = path.join(this.#binPath, "initdb");
  }
  #onData(data: Buffer, process: string, color: number) {
    const prefix = `\x1b[${color}m[${process}]\x1b[0m`;
    console.log(prefix + data.toString().replace(/\n/g, "\n" + prefix));
  }

  async #initDataDir(): Promise<void> {
    await new Promise<void>((resolve, reject) => {
      const proc = spawn(
        this.#initdb,
        ["--pgdata", this.config.dataDir, "--auth", "trust"],
        {
          stdio: "pipe",
        },
      );
      proc.stdout?.on("data", (data: Buffer) => {
        this.#onData(data, "initdb", 32);
      });
      proc.on("close", (code) =>
        code === 0
          ? resolve()
          : reject(new Error(`initdb exited with ${code}`)),
      );
    });
  }

  async #setupAuthentication(): Promise<void> {
    // After database initialization, modify pg_hba.conf to require password authentication
    const pgHbaPath = path.join(this.config.dataDir, "pg_hba.conf");

    try {
      // Read the current pg_hba.conf file
      let pgHbaContent = fs.readFileSync(pgHbaPath, "utf8");

      // Replace trust authentication with password authentication for local connections
      // This replaces lines that have 'trust' with 'md5' (password authentication)
      pgHbaContent = pgHbaContent.replace(
        /(local\s+all\s+all\s+)trust/g,
        "$1md5",
      );

      // Also handle the case where it might be set to 'peer' or other auth methods
      pgHbaContent = pgHbaContent.replace(
        /(local\s+all\s+all\s+)peer/g,
        "$1md5",
      );

      // Also handle IPv4 and IPv6 trust connections
      pgHbaContent = pgHbaContent.replace(
        /(host\s+all\s+all\s+127\.0\.0\.1\/32\s+)trust/g,
        "$1md5",
      );
      pgHbaContent = pgHbaContent.replace(
        /(host\s+all\s+all\s+::1\/128\s+)trust/g,
        "$1md5",
      );

      await this.#setPostgresPassword();
      // Write the updated content back
      fs.writeFileSync(pgHbaPath, pgHbaContent, "utf8");
    } catch (error) {
      console.error("Failed to setup authentication:", error);
      throw new Error("Failed to setup PostgreSQL authentication");
    }
  }

  async #userExists(username: string): Promise<boolean> {
    // Check if a user already exists in the database
    const psql = path.join(this.#binPath, "psql");

    return new Promise((resolve) => {
      const proc = spawn(
        psql,
        [
          "-d",
          "postgres",
          "-t",
          "-c",
          `SELECT 1 FROM pg_roles WHERE rolname='${username}';`,
        ],
        {
          stdio: "pipe",
          env: {
            ...process.env,
            PGDATA: this.config.dataDir,
            PGHOST: this.config.host,
            PGPORT: String(this.config.port),
            //PGUSER: 'postgres',
            PGPASSWORD: this.config.password,
          },
        },
      );

      let output = "";
      proc.stdout?.on("data", (data: Buffer) => {
        this.#onData(data, "psql", 35);
        output += data.toString();
      });

      proc.on("close", (code) => {
        resolve(code === 0 && output.trim() !== "");
      });
    });
  }

  async #createPredefinedUser(
    username: string,
    password: string,
  ): Promise<void> {
    // Check if user already exists first
    if (await this.#userExists(username)) {
      return; // User already exists
    }

    // Create a new user with the specified credentials
    const psql = path.join(this.#binPath, "psql");

    // We'll connect to the default postgres database and create a new user
    await new Promise<void>((resolve, reject) => {
      const proc = spawn(
        psql,
        [
          "-d",
          "postgres",
          "-c",
          `CREATE USER IF NOT EXISTS ${username} WITH PASSWORD '${password}';`,
        ],
        {
          stdio: "inherit",
          env: {
            ...process.env,
            PGDATA: this.config.dataDir,
            PGHOST: this.config.host,
            PGPORT: String(this.config.port),
            //PGUSER: 'postgres',
            PGPASSWORD: this.config.password,
          },
        },
      );
      proc.on("close", (code) =>
        code === 0
          ? resolve()
          : reject(new Error(`psql create user failed with ${code}`)),
      );
    });
  }

  async #setPostgresPassword(): Promise<void> {
    // Create a new user with the specified credentials
    const psql = path.join(this.#binPath, "psql");

    // We'll connect to the default postgres database and create a new user
    await new Promise<void>((resolve, reject) => {
      const proc = spawn(
        psql,
        [
          "-d",
          "postgres",
          "-c",
          `ALTER USER ${this.config.user} WITH PASSWORD '${this.config.password}';`,
        ],
        {
          stdio: "pipe",
          env: {
            ...process.env,
            PGDATA: this.config.dataDir,
            PGHOST: this.config.host,
            PGPORT: String(this.config.port),
            //PGUSER: 'postgres',
            PGPASSWORD: this.config.password,
          },
        },
      );
      proc.stdout?.on("data", (data: Buffer) => {
        this.#onData(data, "psql", 35);
      });
      proc.on("close", (code) =>
        code === 0
          ? resolve()
          : reject(
              new Error(
                `psql alter ${this.config.user} user failed with ${code}`,
              ),
            ),
      );
    });
  }

  #isDatabaseInitialized(): boolean {
    // Check if the database is already initialized
    const pgHbaPath = path.join(this.config.dataDir, "pg_hba.conf");
    return fs.existsSync(pgHbaPath);
  }

  #isAuthenticationConfigured(): boolean {
    // Check if authentication is already properly configured
    const pgHbaPath = path.join(this.config.dataDir, "pg_hba.conf");
    if (!fs.existsSync(pgHbaPath)) {
      return false;
    }

    try {
      const pgHbaContent = fs.readFileSync(pgHbaPath, "utf8");
      // Check if there's a trust authentication method (which we want to replace)
      const regexMatch = /(local\s+all\s+all\s+md5)/g;
      const testResult = regexMatch.test(pgHbaContent);

      return testResult; // Return true if using md5 authentication
    } catch {
      return false;
    }
  }

  async #ensureDatabaseInitialized(): Promise<void> {
    if (!this.#isDatabaseInitialized()) {
      fs.mkdirSync(this.config.dataDir, { recursive: true });
      await this.#initDataDir();
    }
  }

  #killProcessOnPort(port: number): void {
    try {
      execSync(`lsof -ti:${port} | xargs kill 2>/dev/null; true`);
    } catch { /* no process to kill */ }
  }

  async startServer(): Promise<PostgresServer> {
    try {
      // Kill any orphaned postgres on the same port from a previous crash
      this.#killProcessOnPort(this.config.port);

      // Initialize database if needed
      await this.#ensureDatabaseInitialized();

      // Check if authentication is already configured
      const isAuthConfigured = this.#isAuthenticationConfigured();

      this.#proc = spawn(
        this.#postgres,
        [
          "-D",
          this.config.dataDir,
          "-p",
          String(this.config.port),
          "-h",
          this.config.host,
        ],
        {
          stdio: ["ignore", "pipe", "pipe"],
          env: {
            ...process.env,
            PGDATA: this.config.dataDir,
            PGHOST: this.config.host,
            PGPORT: String(this.config.port),
            //PGUSER: 'postgres',
            PGPASSWORD: this.config.password,
          },
        },
      );
      await this.#waitForReady();

      // Only set up authentication and create user if not already done
      if (!isAuthConfigured) {
        // Setup authentication to require username and password
        await this.#setupAuthentication();

        // Restart PostgreSQL to apply authentication changes
        return await this.#restartPostgres();
      }

      // Check if a database exists and create it if it doesn't
      if (!(await this.#databaseExists())) {
        await this.#createDatabase();
      }

      // Create a predefined user with username and password if not already exists
      await this.#createPredefinedUser(this.config.user, this.config.password);

      return {
        process: this.#proc,
        stop: () => this.#stopPostgres(),
      };
    } catch (error) {
      await this.#stopPostgres(); // Ensure we clean up the process if something goes wrong
      this.#proc?.kill("SIGTERM"); // Ensure we clean up the process if something goes wrong
      throw error;
    }
  }

  async #stopPostgres(): Promise<void> {
    // Stop the PostgreSQL daemon
    return new Promise((resolve) => {
      this.#proc.on("close", () => {
        console.log(
          "\x1b[31m[postgres]\x1b[0m" +
            "stopped with code " +
            this.#proc.exitCode,
        );
        resolve();
      });
      this.#proc.kill("SIGTERM");
    });
  }

  #waitForReady(): Promise<void> {
    return new Promise((resolve, reject) => {
      const onData = (data: Buffer) => {
        this.#onData(data, "postgres", 34);
        if (data.toString().includes("ready to accept connections")) {
          this.#proc.stderr?.off("data", onData);
          resolve();
        }
      };

      this.#proc.stderr?.on("data", onData);
      this.#proc.on("error", reject);
      this.#proc.on("close", (code) => {
        if (code !== 0)
          reject(new Error(`postgres exited early with code ${code}`));
      });

      // Timeout fallback
      setTimeout(() => {
        this.#proc.stderr?.off("data", onData);
        resolve(); // assume ready after 10s
      }, 10_000);
    });
  }

  #getPostgresBinPath(): string {
    const platform = PLATFORM_MAP[process.platform];

    if (!platform) {
      throw new Error(`Unsupported platform: ${process.platform}`);
    }
    // Points to the extraResources destination inside the .app / installed dir
    let programsPath: string = "";
    switch (platform) {
      case "windows":
        programsPath = path.join(execSync("echo %PROGRAMFILES%")
          .toString()
          .replace("\r\n", ""), 'postgres')
        break;
      case "mac":
        // just put the prebuilt binaries in the Applications folder, since we don't have an installer that can place them in Program Files
        programsPath = path.join("/Applications/postgres");
        break;
      case "linux":
        programsPath = path.join("/usr/lib/postgresql/16/")
        break
      default:
        console.log(platform);
        throw new Error("no se puede determinar la carpeta PROGRAMFILES");
    }
    return path.join(programsPath, "bin");
  }

  async #restartPostgres(): Promise<PostgresServer> {
    // Restart the PostgreSQL daemon after authentication changes
    await this.#stopPostgres();

    return await this.startServer();
  }

  async #databaseExists(): Promise<boolean> {
    // Check if a database already exists in the PostgreSQL instance
    const psql = path.join(this.#binPath, "psql");

    return new Promise((resolve) => {
      const proc = spawn(
        psql,
        [
          "-d",
          "postgres",
          "-t",
          "-c",
          `SELECT 1 FROM pg_database WHERE datname='${this.config.database}';`,
        ],
        {
          stdio: "pipe",
          env: {
            ...process.env,
            PGDATA: this.config.dataDir,
            PGHOST: this.config.host,
            PGPORT: String(this.config.port),
            //PGUSER: 'postgres',
            PGPASSWORD: this.config.password,
          },
        },
      );

      let output = "";
      proc.stdout.on("data", (data: Buffer) => {
        output += data.toString();
      });

      proc.on("close", (code) => {
        resolve(code === 0 && output.trim() !== "");
      });
    });
  }

  async #createDatabase(): Promise<void> {
    // Create a new database with the specified name
    const psql = path.join(this.#binPath, "psql");

    await new Promise<void>((resolve, reject) => {
      const proc = spawn(
        psql,
        ["-d", "postgres", "-c", `CREATE DATABASE ${this.config.database};`],
        {
          stdio: "pipe",
          env: {
            ...process.env,
            PGDATA: this.config.dataDir,
            PGHOST: this.config.host,
            PGPORT: String(this.config.port),
            //PGUSER: 'postgres',
            PGPASSWORD: this.config.password,
          },
        },
      );
      proc.stdout?.on("data", (data: Buffer) => {
        this.#onData(data, "psql", 35);
      });
      proc.on("close", (code: number) =>
        code === 0
          ? resolve()
          : reject(new Error(`psql create database failed with ${code}`)),
      );
    });
  }
}
