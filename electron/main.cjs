/**
 * Electron main entry for AI Notes desktop builds.
 */

const { app, BrowserWindow, shell, dialog } = require("electron");
const path = require("path");
const http = require("http");
const fs = require("fs");

const PORT = Number(process.env.PORT || 3000);
const HOST = process.env.HOST || "127.0.0.1";
const SERVER_URL = `http://${HOST}:${PORT}`;

function installFileLogger() {
  try {
    const logDir = app.getPath("logs");
    fs.mkdirSync(logDir, { recursive: true });
    const current = path.join(logDir, "ai-notes.log");
    const previous = path.join(logDir, "ai-notes.prev.log");
    if (fs.existsSync(current)) {
      try {
        fs.renameSync(current, previous);
      } catch {
        /* best-effort */
      }
    }
    const stream = fs.createWriteStream(current, { flags: "a" });
    const write = (prefix) => (...args) => {
      const line =
        `[${new Date().toISOString()}] ${prefix} ` +
        args.map((a) => (typeof a === "string" ? a : String(a))).join(" ") +
        "\n";
      try {
        stream.write(line);
      } catch {
        /* ignore */
      }
    };
    const origLog = console.log.bind(console);
    const origError = console.error.bind(console);
    console.log = (...a) => {
      origLog(...a);
      write("INFO")(...a);
    };
    console.error = (...a) => {
      origError(...a);
      write("ERROR")(...a);
    };
    console.log(`[electron] logging to ${current}`);
  } catch (err) {
    console.error("[electron] failed to install file logger:", err);
  }
}

function resolveBundledServerPath() {
  const candidates = [
    path.join(__dirname, "..", "dist", "server.cjs"),
    path.join(process.resourcesPath || "", "app", "dist", "server.cjs"),
    path.join(process.resourcesPath || "", "app.asar", "dist", "server.cjs"),
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
  process.env.AI_NOTES_DESKTOP = "1";
  process.env.PORT = String(PORT);
  process.env.HOST = HOST;

  const workingRoot = ensureWritableUserData();
  try {
    process.chdir(workingRoot);
    console.log(`[electron] working directory: ${workingRoot}`);
  } catch (err) {
    console.warn(`[electron] could not chdir:`, err?.message || err);
  }

  if (app.isPackaged) {
    process.env.APP_DIST_DIR = path.join(__dirname, "..", "dist");
  }

  const serverPath = resolveBundledServerPath();
  console.log(`[electron] loading server: ${serverPath}`);
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
    title: "AI Notes",
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
  try {
    await startEmbeddedServer();
  } catch (err) {
    console.error("[electron] failed to start embedded server:", err);
    dialog.showErrorBox(
      "AI Notes failed to start",
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
