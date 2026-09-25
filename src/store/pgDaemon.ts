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
  socketDir?: string;
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
  #socketDir: string;
  config: PostgresConfig;

  constructor({
    user,
    password,
    database,
    port = 5432,
    host = "127.0.0.1",
    dataDir,
    socketDir,
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
    // Socket directory must be writable by the current user. Postgres
    // defaults to /var/run/postgresql, which is often owned by the
    // system postgres user (or nobody) and causes:
    //   FATAL: could not create lock file "/var/run/postgresql/.s.PGSQL.<port>.lock": Permission denied
    // Use a per-user directory (under the data dir by default) instead.
    this.#socketDir = socketDir ?? path.join(this.config.dataDir, "run");
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
    // Modify pg_hba.conf to require password authentication for TCP
    // connections. Local (unix socket) connections keep 'trust' because the
    // socket lives in a 0700 directory owned by the current user, so peer
    // trust there is safe — and it lets the bootstrap psql calls (run as the
    // current OS user, not as the 'postgres' superuser) keep working.
    const pgHbaPath = path.join(this.config.dataDir, "pg_hba.conf");

    try {
      // Read the current pg_hba.conf file
      let pgHbaContent = fs.readFileSync(pgHbaPath, "utf8");

      // Require md5 (password) authentication for TCP connections
      pgHbaContent = pgHbaContent.replace(
        /(host\s+all\s+all\s+127\.0\.0\.1\/32\s+)trust/g,
        "$1md5",
      );
      pgHbaContent = pgHbaContent.replace(
        /(host\s+all\s+all\s+::1\/128\s+)trust/g,
        "$1md5",
      );

      // Write the updated content back before setting the password, so the
      // ALTER USER below is not left hanging if it fails.
      fs.writeFileSync(pgHbaPath, pgHbaContent, "utf8");

      // initdb creates a superuser role named after the current OS user.
      // If the app connects as a different role (e.g. 'postgres'), create it.
      await this.#ensureRoleExists(this.config.user, this.config.password);

      await this.#setPostgresPassword();
    } catch (error) {
      console.error("Failed to setup authentication:", error);
      throw new Error("Failed to setup PostgreSQL authentication");
    }
  }

  /**
   * Ensure the configured role has superuser privileges (needed for
   * CREATE EXTENSION and pg_catalog helpers). Connects over the local unix
   * socket as the current OS user (peer/trust).
   */
  async #ensureSuperuser(username: string): Promise<void> {
    const psql = path.join(this.#binPath, "psql");

    await new Promise<void>((resolve, reject) => {
      const proc = spawn(
        psql,
        [
          "-d",
          "postgres",
          "-v",
          "ON_ERROR_STOP=1",
          "-c",
          `DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${username}' AND rolsuper) THEN ALTER ROLE ${username} WITH SUPERUSER; END IF; END $$;`,
        ],
        {
          stdio: "pipe",
          env: {
            ...process.env,
            PGDATA: this.config.dataDir,
            PGHOST: this.#socketDir,
            PGPORT: String(this.config.port),
          },
        },
      );
      proc.stderr?.on("data", (data: Buffer) => {
        this.#onData(data, "psql", 35);
      });
      proc.on("close", (code) =>
        code === 0
          ? resolve()
          : reject(new Error(`psql ensure superuser failed with ${code}`)),
      );
    });
  }

  /**
   * Ensure a role exists (creating it as a superuser if needed). Connects
   * over the local unix socket as the current OS user (peer/trust).
   */
  async #ensureRoleExists(username: string, password: string): Promise<void> {
    const psql = path.join(this.#binPath, "psql");

    await new Promise<void>((resolve, reject) => {
      const proc = spawn(
        psql,
        [
          "-d",
          "postgres",
          "-v",
          `ON_ERROR_STOP=1`,
          "-c",
          `DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${username}') THEN CREATE ROLE ${username} WITH LOGIN SUPERUSER PASSWORD '${password}'; ELSE ALTER ROLE ${username} WITH LOGIN SUPERUSER PASSWORD '${password}'; END IF; END $$;`,
        ],
        {
          stdio: "pipe",
          env: {
            ...process.env,
            PGDATA: this.config.dataDir,
            PGHOST: this.#socketDir,
            PGPORT: String(this.config.port),
          },
        },
      );
      proc.stdout?.on("data", (data: Buffer) => {
        this.#onData(data, "psql", 35);
      });
      proc.stderr?.on("data", (data: Buffer) => {
        this.#onData(data, "psql", 35);
      });
      proc.on("close", (code) =>
        code === 0
          ? resolve()
          : reject(
              new Error(`psql ensure role ${username} failed with ${code}`),
            ),
      );
    });
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
            PGHOST: this.#socketDir,
            PGPORT: String(this.config.port),
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

    // Create a new user with the specified credentials. Connects over the
    // local unix socket as the current OS user (peer/trust). Note: PG16 has
    // no CREATE USER IF NOT EXISTS, so #userExists() guards this.
    const psql = path.join(this.#binPath, "psql");

    await new Promise<void>((resolve, reject) => {
      const proc = spawn(
        psql,
        [
          "-d",
          "postgres",
          "-c",
          `CREATE USER ${username} WITH PASSWORD '${password}';`,
        ],
        {
          stdio: "inherit",
          env: {
            ...process.env,
            PGDATA: this.config.dataDir,
            PGHOST: this.#socketDir,
            PGPORT: String(this.config.port),
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
    // Set the password for the configured user. Connects over the local unix
    // socket as the current OS user (peer/trust), so this works even when
    // config.user is 'postgres' and the OS user is not.
    const psql = path.join(this.#binPath, "psql");

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
            PGHOST: this.#socketDir,
            PGPORT: String(this.config.port),
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
    // Check if TCP authentication is already set to md5 (local socket lines
    // intentionally stay 'trust' — see #setupAuthentication).
    const pgHbaPath = path.join(this.config.dataDir, "pg_hba.conf");
    if (!fs.existsSync(pgHbaPath)) {
      return false;
    }

    try {
      const pgHbaContent = fs.readFileSync(pgHbaPath, "utf8");
      return /host\s+all\s+all\s+127\.0\.0\.1\/32\s+md5/.test(pgHbaContent);
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

  /**
   * If a previous run crashed and left postmaster.pid behind while no
   * postgres is actually running from this data dir, remove the stale file
   * so the next start is not refused with "lock file exists".
   */
  #removeStalePidFile(): void {
    const pidFile = path.join(this.config.dataDir, "postmaster.pid");
    if (!fs.existsSync(pidFile)) return;
    try {
      const pid = parseInt(fs.readFileSync(pidFile, "utf8").split("\n")[0] ?? "", 10);
      if (Number.isNaN(pid)) return;
      // Signal 0 checks existence without sending anything
      process.kill(pid, 0);
    } catch {
      // Process is gone → stale pid file
      try {
        fs.unlinkSync(pidFile);
      } catch { /* ignore */ }
    }
  }

  /**
   * Ensure the pgvector extension is available to the embedded daemon.
   *
   * PostgreSQL requires the extension's control file to live in the system
   * share dir (e.g. /usr/share/postgresql/16/extension) and the .so in the
   * system lib dir — both are root-only. So this method:
   *   1. If the system install exists (apt postgresql-16-pgvector), use it.
   *   2. Otherwise build pgvector from source into a persistent, user-writable
   *      build dir (no root needed), then install the artifacts to the system
   *      dirs. The install step needs root; if it fails we leave the built
   *      files in place and print the exact sudo command to finish the job.
   *
   * Returns true only if the extension is actually loadable by the daemon.
   */
  async #ensurePgvectorInstalled(): Promise<boolean> {
    const version = "0.8.0";
    // On Debian/Ubuntu the postgres binaries live in /usr/lib/postgresql/16/bin,
    // but the extension control/SQL files live in /usr/share/postgresql/16/extension
    // (NOT under /usr/lib). Probe both the conventional /usr/share path and the
    // /usr/lib/.../share path to be robust across layouts.
    const systemLibDir = path.join(this.#binPath, "..", "lib");
    const systemSo = path.join(systemLibDir, "vector.so");
    const candidateExtDirs = [
      "/usr/share/postgresql/16/extension",
      path.join(this.#binPath, "..", "share", "postgresql", "16", "extension"),
    ];
    const systemExtDir =
      candidateExtDirs.find((d) => fs.existsSync(d)) ??
      "/usr/share/postgresql/16/extension";
    const systemControl = path.join(systemExtDir, "vector.control");

    // 1. System-wide install already present?
    if (fs.existsSync(systemSo) && fs.existsSync(systemControl)) {
      console.log("✓ pgvector found in system PostgreSQL directories, using it");
      return true;
    }

    // Persistent build dir (survives across runs so we build only once).
    const buildDir = path.join(this.config.dataDir, "build-pgvector");
    const srcDirName = `pgvector-${version}`;
    const builtSo = path.join(buildDir, srcDirName, "vector.so");
    const builtControl = path.join(buildDir, srcDirName, "vector.control");
    const builtSql = path.join(buildDir, srcDirName, "sql", `vector--${version}.sql`);

    const { execFileSync } = await import("node:child_process");
    const osModule = await import("node:os");

    // 2. Build from source if we don't already have the artifacts.
    if (!fs.existsSync(builtSo) || !fs.existsSync(builtControl) || !fs.existsSync(builtSql)) {
      console.log("🔧 pgvector not found, building from source (one-time setup)...");
      try {
        fs.mkdirSync(buildDir, { recursive: true });

        // a. Download pgvector source (cached)
        const tarball = path.join(buildDir, "pgvector.tar.gz");
        if (!fs.existsSync(tarball)) {
          const url = `https://github.com/pgvector/pgvector/archive/refs/tags/v${version}.tar.gz`;
          console.log(`⬇ Downloading ${url}`);
          execFileSync("curl", ["-sL", "--fail", "-o", tarball, url], { stdio: "pipe" });
        }

        // b. Extract fresh
        const srcDir = path.join(buildDir, srcDirName);
        fs.rmSync(srcDir, { recursive: true, force: true });
        execFileSync("tar", ["xzf", tarball, "-C", buildDir], { stdio: "pipe" });

        // c. Ensure PostgreSQL server headers are available for the build.
        //    The system package (postgresql-16) ships pg_config but not the
        //    headers, so fetch postgresql-server-dev-<ver> from the apt mirror
        //    and extract it into the persistent build dir (no root needed).
        let includeDir: string | null = null;
        const headerProbe = path.join(
          "/usr/include/postgresql", "16", "server", "postgres.h",
        );
        if (fs.existsSync(headerProbe)) {
          includeDir = "/usr/include/postgresql/16";
        } else {
          console.log("⬇ Fetching PostgreSQL server headers (postgresql-server-dev)...");
          const aptDir = path.join(buildDir, "pgdev");
          fs.mkdirSync(aptDir, { recursive: true });
          execFileSync("apt-get", ["download", "postgresql-server-dev-16"], {
            cwd: aptDir,
            stdio: "pipe",
          });
          const deb = fs
            .readdirSync(aptDir)
            .find((f) => f.endsWith(".deb"));
          if (!deb) throw new Error("deb not downloaded");
          const extractDir = path.join(aptDir, "extracted");
          execFileSync("dpkg-deb", ["-x", path.join(aptDir, deb), extractDir], {
            stdio: "pipe",
          });
          const inc = path.join(
            extractDir, "usr", "include", "postgresql", "16",
          );
          if (fs.existsSync(path.join(inc, "server", "postgres.h"))) {
            includeDir = inc;
          }
        }
        const includeFlags = includeDir ? `-I${includeDir}/server` : "";

        // d. Build. with_llvm=no avoids needing clang/LLVM bitcode tools;
        //    CC=gcc avoids a missing default compiler. CPPFLAGS is used for the
        //    include paths because pgxs appends its own -I flags after CFLAGS.
        console.log("🔨 Compiling pgvector (this can take a minute)...");
        const makeArgs = [
          "-C", srcDir,
          "-j", String(osModule.cpus().length),
          "CC=gcc",
          "with_llvm=no",
        ];
        if (includeFlags) {
          makeArgs.push(`CPPFLAGS=${includeFlags}`);
        }
        execFileSync("make", makeArgs, { stdio: "pipe" });

        if (!fs.existsSync(builtSo)) {
          throw new Error("build finished but vector.so is missing");
        }
        console.log(`✓ pgvector ${version} built to ${srcDir}`);
      } catch (error) {
        console.warn(`⚠ Failed to build pgvector: ${(error as Error).message}`);
        console.warn("  You can install it system-wide instead: sudo apt install postgresql-16-pgvector");
        return false;
      }
    }

    // 3. Install the built artifacts to the system dirs (needs root).
    const systemSql = path.join(systemExtDir, `vector--${version}.sql`);
    try {
      fs.mkdirSync(systemExtDir, { recursive: true });
      fs.copyFileSync(builtSo, systemSo);
      fs.copyFileSync(builtControl, systemControl);
      fs.copyFileSync(builtSql, systemSql);
      console.log("✓ pgvector installed to system PostgreSQL directories");
      return true;
    } catch (error) {
      console.warn(
        `⚠ Could not install pgvector to system directories: ${(error as Error).message}`,
      );
      console.warn("  The built files are ready. Finish the install with (needs sudo):\n");
      console.warn(`    sudo cp ${builtSo} ${systemSo}`);
      console.warn(`    sudo cp ${builtControl} ${systemControl}`);
      console.warn(`    sudo cp ${builtSql} ${systemSql}`);
      console.warn("\n  …or install the packaged version instead:\n");
      console.warn("    sudo apt install postgresql-16-pgvector\n");
      return false;
    }
  }

  /**
   * After the daemon is up (and the configured role is superuser), enable the
   * pgvector extension in the target database if it is missing. The extension
   * files must already be in the system dirs (see #ensurePgvectorInstalled).
   */
  async #ensurePgvectorExtension(): Promise<void> {
    const psql = path.join(this.#binPath, "psql");

    const runPsql = async (sql: string): Promise<void> => {
      await new Promise<void>((resolve, reject) => {
        const proc = spawn(
          psql,
          ["-d", this.config.database, "-v", "ON_ERROR_STOP=1", "-c", sql],
          {
            stdio: "pipe",
            env: {
              ...process.env,
              PGDATA: this.config.dataDir,
              PGHOST: this.#socketDir,
              PGPORT: String(this.config.port),
              PGUSER: this.config.user,
              PGPASSWORD: this.config.password,
            },
          },
        );
        let err = "";
        proc.stderr?.on("data", (d: Buffer) => { err += d.toString(); });
        proc.on("close", (code) =>
          code === 0
            ? resolve()
            : reject(new Error(`psql failed: ${err.trim()}`)),
        );
      });
    };

    try {
      await runPsql("CREATE EXTENSION IF NOT EXISTS vector;");
      console.log("✓ pgvector extension ready");
    } catch (error) {
      console.warn(
        `⚠ Could not enable pgvector extension: ${(error as Error).message}`,
      );
    }
  }

  async startServer(): Promise<PostgresServer> {
    try {
      // Kill any orphaned postgres on the same port from a previous crash
      this.#killProcessOnPort(this.config.port);

      // Initialize database if needed
      await this.#ensureDatabaseInitialized();

      // Clear a stale postmaster.pid left behind by a crashed run
      this.#removeStalePidFile();

      // Check if authentication is already configured
      const isAuthConfigured = this.#isAuthenticationConfigured();

      fs.mkdirSync(this.#socketDir, { recursive: true });
      fs.chmodSync(this.#socketDir, 0o700);
      this.#proc = spawn(
        this.#postgres,
        [
          "-D",
          this.config.dataDir,
          "-p",
          String(this.config.port),
          "-h",
          this.config.host,
          "-k",
          this.#socketDir,
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

      // Make sure the configured role is a superuser (needed to create the
      // pgvector extension).
      await this.#ensureSuperuser(this.config.user);

      // Install the pgvector extension (builds from source on first run if the
      // system package is not present) and enable it in the database.
      const pgvectorReady = await this.#ensurePgvectorInstalled();
      if (pgvectorReady) {
        await this.#ensurePgvectorExtension();
      }

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
    const proc = this.#proc;
    if (!proc || proc.exitCode !== null) {
      // Already stopped (or never started)
      return;
    }
    await new Promise<void>((resolve) => {
      const onClose = () => {
        console.log(
          "\x1b[31m[postgres]\x1b[0m" +
            "stopped with code " +
            proc.exitCode,
        );
        resolve();
      };
      proc.once("close", onClose);
      // Safety net: if the process ignores SIGTERM, force-kill it
      setTimeout(() => {
        if (proc.exitCode === null) {
          try {
            proc.kill("SIGKILL");
          } catch { /* already gone */ }
        }
      }, 10_000).unref();
      proc.kill("SIGTERM");
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
            PGHOST: this.#socketDir,
            PGPORT: String(this.config.port),
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
            PGHOST: this.#socketDir,
            PGPORT: String(this.config.port),
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
