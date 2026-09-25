import { Pool } from "pg";
import readline from "node:readline/promises";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Catalog {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  embedding_count?: number;
}

export interface CatalogStore {
  listCatalogs(): Promise<Catalog[]>;
  createCatalog(name: string, description?: string): Promise<Catalog>;
  getCatalog(idOrName: string): Promise<Catalog | null>;
  deleteCatalog(id: string): Promise<void>;
  countEmbeddings(catalogId: string): Promise<number>;
}

// ---------------------------------------------------------------------------
// In-memory catalog store (used when PostgreSQL is unavailable)
// ---------------------------------------------------------------------------

export class InMemoryCatalogStore implements CatalogStore {
  private catalogs = new Map<string, Catalog>();
  private counter = 0;

  async listCatalogs(): Promise<Catalog[]> {
    return Array.from(this.catalogs.values()).sort(
      (a, b) => a.created_at.localeCompare(b.created_at),
    );
  }

  async createCatalog(name: string, description?: string): Promise<Catalog> {
    const normalized = name.trim().toLowerCase();
    const existing = await this.getCatalog(normalized);
    if (existing) return existing;

    const catalog: Catalog = {
      id: `cat_${++this.counter}_${Date.now()}`,
      name: normalized,
      description: description?.trim() || null,
      created_at: new Date().toISOString(),
    };
    this.catalogs.set(catalog.id, catalog);
    return catalog;
  }

  async getCatalog(idOrName: string): Promise<Catalog | null> {
    const direct = this.catalogs.get(idOrName);
    if (direct) return direct;
    const normalized = idOrName.trim().toLowerCase();
    for (const catalog of this.catalogs.values()) {
      if (catalog.name === normalized) return catalog;
    }
    return null;
  }

  async deleteCatalog(id: string): Promise<void> {
    this.catalogs.delete(id);
  }

  async countEmbeddings(_catalogId: string): Promise<number> {
    return 0;
  }
}

// ---------------------------------------------------------------------------
// PostgreSQL catalog store
// ---------------------------------------------------------------------------

export class PgCatalogStore implements CatalogStore {
  constructor(private pool: Pool) {}

  async ensureSchema(): Promise<void> {
    const client = await this.pool.connect();
    try {
      await client.query(`
        CREATE TABLE IF NOT EXISTS catalogs (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          name TEXT NOT NULL UNIQUE,
          description TEXT,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
      `);
      await client.query(`
        ALTER TABLE embeddings
        ADD COLUMN IF NOT EXISTS catalog_id UUID REFERENCES catalogs(id) ON DELETE CASCADE;
      `);
      await client.query(`
        CREATE INDEX IF NOT EXISTS embeddings_catalog_idx ON embeddings (catalog_id);
      `);
    } finally {
      client.release();
    }
  }

  async listCatalogs(): Promise<Catalog[]> {
    const result = await this.pool.query(
      `SELECT c.id, c.name, c.description, c.created_at,
              COUNT(e.id) AS embedding_count
         FROM catalogs c
         LEFT JOIN embeddings e ON e.catalog_id = c.id
        GROUP BY c.id
        ORDER BY c.created_at ASC`,
    );
    return result.rows.map((row) => ({
      id: row.id,
      name: row.name,
      description: row.description,
      created_at: row.created_at,
      embedding_count: parseInt(row.embedding_count, 10),
    }));
  }

  async createCatalog(name: string, description?: string): Promise<Catalog> {
    const normalized = name.trim().toLowerCase();
    if (!normalized) throw new Error("Catalog name cannot be empty");
    const existing = await this.getCatalog(normalized);
    if (existing) return existing;

    const result = await this.pool.query(
      `INSERT INTO catalogs (name, description)
       VALUES ($1, $2)
       RETURNING id, name, description, created_at`,
      [normalized, description?.trim() || null],
    );
    const row = result.rows[0];
    return {
      id: row.id,
      name: row.name,
      description: row.description,
      created_at: row.created_at,
    };
  }

  async getCatalog(idOrName: string): Promise<Catalog | null> {
    const trimmed = idOrName.trim();
    if (!trimmed) return null;
    const isUuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(trimmed);
    const result = await this.pool.query(
      `SELECT id, name, description, created_at
         FROM catalogs
        WHERE (id = $1::uuid) OR (name = $2)
        LIMIT 1`,
      [isUuid ? trimmed : null, trimmed.toLowerCase()],
    );
    if (result.rows.length === 0) return null;
    const row = result.rows[0];
    return {
      id: row.id,
      name: row.name,
      description: row.description,
      created_at: row.created_at,
    };
  }

  async deleteCatalog(id: string): Promise<void> {
    await this.pool.query(`DELETE FROM catalogs WHERE id = $1`, [id]);
  }

  async countEmbeddings(catalogId: string): Promise<number> {
    const result = await this.pool.query(
      `SELECT COUNT(*) AS count FROM embeddings WHERE catalog_id = $1`,
      [catalogId],
    );
    return parseInt(result.rows[0].count, 10);
  }
}

// ---------------------------------------------------------------------------
// Interactive prompts
// ---------------------------------------------------------------------------

/**
 * Prompt the user to pick an existing catalog or create a new one.
 * Returns the selected/created catalog.
 */
export async function promptCatalogSelection(
  store: CatalogStore,
  purpose: "store" | "query",
): Promise<Catalog> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  try {
    const catalogs = await store.listCatalogs();

    while (true) {
      console.log(`\n📚 Catalogs (${purpose === "store" ? "to store embeddings in" : "to query"}):`);
      if (catalogs.length === 0) {
        console.log("  (none yet)");
      }
      for (const catalog of catalogs) {
        const count = catalog.embedding_count !== undefined ? ` [${catalog.embedding_count} embeddings]` : "";
        const desc = catalog.description ? ` — ${catalog.description}` : "";
        console.log(`  ${catalogs.indexOf(catalog) + 1}. ${catalog.name}${desc}${count}`);
      }
      console.log(`  ${catalogs.length + 1}. ➕ Create new catalog`);
      console.log("");

      const choice = await rl.question(
        `\x1b[35mSelect a catalog (number or name):\x1b[0m `,
      );

      const trimmed = choice.trim();
      const asIndex = parseInt(trimmed, 10);
      if (!isNaN(asIndex) && asIndex >= 1 && asIndex <= catalogs.length) {
        return catalogs[asIndex - 1]!;
      }

      if (trimmed.toLowerCase() === String(catalogs.length + 1) || trimmed === "new" || trimmed === "n") {
        const name = await rl.question("Catalog name: ");
        if (!name.trim()) {
          console.log("⚠ Name cannot be empty, try again");
          continue;
        }
        const descAnswer = await rl.question("Description (optional, press Enter to skip): ");
        const catalog = await store.createCatalog(name, descAnswer || undefined);
        console.log(`✓ Created catalog "${catalog.name}"`);
        return catalog;
      }

      const byName = await store.getCatalog(trimmed);
      if (byName) return byName;

      console.log("⚠ Invalid selection, try again");
    }
  } finally {
    rl.close();
  }
}

/**
 * Prompt the user to pick a catalog to query, or query all catalogs.
 * Returns null when "all" is selected.
 */
export async function promptCatalogForQuery(
  store: CatalogStore,
): Promise<Catalog | null> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  try {
    const catalogs = await store.listCatalogs();

    while (true) {
      console.log(`\n📚 Catalogs:`);
      if (catalogs.length === 0) {
        console.log("  (none yet — all embeddings are unassigned)");
      }
      for (const catalog of catalogs) {
        const count = catalog.embedding_count !== undefined ? ` [${catalog.embedding_count} embeddings]` : "";
        const desc = catalog.description ? ` — ${catalog.description}` : "";
        console.log(`  ${catalogs.indexOf(catalog) + 1}. ${catalog.name}${desc}${count}`);
      }
      console.log(`  ${catalogs.length + 1}. 🌐 All catalogs`);
      console.log("");

      const choice = await rl.question(
        `\x1b[35mSelect a catalog to query (number or name):\x1b[0m `,
      );

      const trimmed = choice.trim();
      const asIndex = parseInt(trimmed, 10);
      if (!isNaN(asIndex) && asIndex >= 1 && asIndex <= catalogs.length) {
        return catalogs[asIndex - 1]!;
      }
      if (
        trimmed.toLowerCase() === String(catalogs.length + 1) ||
        trimmed.toLowerCase() === "all" ||
        trimmed === "a"
      ) {
        return null;
      }

      const byName = await store.getCatalog(trimmed);
      if (byName) return byName;

      console.log("⚠ Invalid selection, try again");
    }
  } finally {
    rl.close();
  }
}
