/**
 * Electron main entry for Acaiguardian (AI Notes).
 *
 * Starts the bundled Express server, then opens a BrowserWindow at localhost.
 */

const { app, BrowserWindow, shell, dialog } = require("electron");
const path = require("path");
const http = require("http");
const fs = require("fs");

const PORT = Number(process.env.PORT || 4783);
const HOST = process.env.HOST || "127.0.0.1";
const SERVER_URL = `http://${HOST}:${PORT}`;

function installFileLogger() {
  try {
    const logDir = app.getPath("logs");
    fs.mkdirSync(logDir, { recursive: true });
    const current = path.join(logDir, "acaiguardian.log");
    const previous = path.join(logDir, "acaiguardian.prev.log");
    if (fs.existsSync(current)) {
      try {
        fs.renameSync(current, previous);
      } catch {
        /* best-effort log rotation */
      }
    }
    const stream = fs.createWriteStream(current, { flags: "a" });
    const write = (prefix) => (...args) => {
      const line =
        `[${new Date().toISOString()}] ${prefix} ` +
        args
          .map((a) => (typeof a === "string" ? a : safeStringify(a)))
          .join(" ") +
        "\n";
      try {
        stream.write(line);
      } catch {
        /* ignore write-after-end */
      }
    };
    const origLog = console.log.bind(console);
    const origWarn = console.warn.bind(console);
    const origError = console.error.bind(console);
    console.log = (...a) => {
      origLog(...a);
      write("INFO")(...a);
    };
    console.warn = (...a) => {
      origWarn(...a);
      write("WARN")(...a);
    };
    console.error = (...a) => {
      origError(...a);
      write("ERROR")(...a);
    };
    process.on("uncaughtException", (err) => write("FATAL")(err?.stack || err));
    process.on("unhandledRejection", (err) =>
      write("FATAL")(err?.stack || String(err))
    );
    console.log(`[electron] logging to ${current}`);
  } catch (err) {
    console.error("[electron] failed to install file logger:", err);
  }
}

function safeStringify(obj) {
  if (obj instanceof Error) {
    const base = {
      name: obj.name,
      message: obj.message,
      stack: obj.stack,
    };
    for (const key of Object.keys(obj)) {
      base[key] = obj[key];
    }
    try {
      return JSON.stringify(base);
    } catch {
      return obj.stack || obj.message || String(obj);
    }
  }
  try {
    return JSON.stringify(obj);
  } catch {
    return String(obj);
  }
}

function resolveBundledServerPath() {
  const candidates = [
    path.join(__dirname, "..", "dist-electron", "server.cjs"),
    path.join(process.resourcesPath || "", "app", "dist-electron", "server.cjs"),
    path.join(
      process.resourcesPath || "",
      "app.asar.unpacked",
      "dist-electron",
      "server.cjs"
    ),
  ];
  for (const candidate of candidates) {
    if (candidate && fs.existsSync(candidate)) return candidate;
  }
  return candidates[0];
}

function ensureWritableUserData() {
  const userRoot = app.getPath("userData");
  fs.mkdirSync(path.join(userRoot, "local_storage"), { recursive: true });
  return userRoot;
}

function loadPackagedEnv() {
  try {
    const userEnv = path.join(app.getPath("userData"), ".env");
    if (fs.existsSync(userEnv)) {
      require("dotenv").config({ path: userEnv });
      console.log(`[electron] loaded env from ${userEnv}`);
    }
  } catch (err) {
    console.warn("[electron] dotenv not available:", err?.message || err);
  }
}

function loadBundledDefaultEnv() {
  try {
    const bundledEnv = path.join(__dirname, "bundled.env");
    if (!fs.existsSync(bundledEnv)) return;
    const parsed = require("dotenv").parse(fs.readFileSync(bundledEnv, "utf8"));
    let applied = 0;
    for (const [key, value] of Object.entries(parsed)) {
      if (process.env[key] === undefined || process.env[key] === "") {
        process.env[key] = value;
        applied += 1;
      }
    }
    console.log(`[electron] loaded bundled defaults (${applied} keys)`);
  } catch (err) {
    console.warn("[electron] failed to load bundled.env:", err?.message || err);
  }
}

function waitForServer(url, timeoutMs = 60000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const tryOnce = () => {
      const req = http.get(url, (res) => {
        res.resume();
        resolve();
      });
      req.on("error", () => {
        if (Date.now() - start > timeoutMs) {
          reject(new Error(`Server did not respond at ${url} within ${timeoutMs}ms`));
          return;
        }
        setTimeout(tryOnce, 250);
      });
    };
    tryOnce();
  });
}

async function startEmbeddedServer() {
  process.env.NODE_ENV = "production";
  process.env.PORT = String(PORT);
  process.env.HOST = HOST;

  const workingRoot = ensureWritableUserData();
  try {
    process.chdir(workingRoot);
    console.log(`[electron] working directory: ${workingRoot}`);
  } catch (err) {
    console.warn(`[electron] could not chdir to ${workingRoot}:`, err?.message || err);
  }

  if (app.isPackaged) {
    process.env.APP_DIST_DIR = path.join(__dirname, "..", "dist");
  }

  try {
    process.env.APP_VERSION = app.getVersion();
  } catch {
    /* non-fatal */
  }

  const serverPath = resolveBundledServerPath();
  console.log(`[electron] loading server bundle: ${serverPath}`);
  require(serverPath);

  await waitForServer(SERVER_URL);
}

function createWindow() {
  const iconCandidates = [
    path.join(__dirname, "assets", "icon.png"),
    path.join(process.resourcesPath || "", "app", "electron", "assets", "icon.png"),
  ];
  const iconPath = iconCandidates.find((p) => p && fs.existsSync(p));

  const win = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 900,
    minHeight: 600,
    backgroundColor: "#1a2033",
    title: "Acaiguardian",
    icon: iconPath || undefined,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//.test(url)) {
      shell.openExternal(url);
      return { action: "deny" };
    }
    return { action: "allow" };
  });

  win.loadURL(SERVER_URL);
  return win;
}

app.whenReady().then(async () => {
  installFileLogger();
  loadPackagedEnv();
  loadBundledDefaultEnv();
  try {
    await startEmbeddedServer();
  } catch (err) {
    console.error("[electron] failed to start embedded server:", err);
    dialog.showErrorBox(
      "Acaiguardian failed to start",
      `The embedded server did not start.\n\n${err?.stack || err}`
    );
    app.quit();
    return;
  }
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
