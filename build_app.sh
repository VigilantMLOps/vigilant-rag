#!/bin/bash
# Creates Atlas RAG.app on the Desktop
set -euo pipefail

REPO="/Users/macbook/Documents/GitHub/vigilant-rag"
PYTHON="$REPO/path/to/venv/bin/python3"
APP="$HOME/Desktop/Atlas RAG.app"

echo "Building Atlas RAG.app..."

mkdir -p "$APP/Contents/MacOS"
mkdir -p "$APP/Contents/Resources"

# ── Info.plist ────────────────────────────────────────────────────────────────
cat > "$APP/Contents/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key>      <string>atlas-rag</string>
  <key>CFBundleIdentifier</key>      <string>com.atlasai.rag</string>
  <key>CFBundleName</key>            <string>Atlas RAG</string>
  <key>CFBundleDisplayName</key>     <string>Atlas RAG</string>
  <key>CFBundleVersion</key>         <string>1.0.0</string>
  <key>CFBundleShortVersionString</key> <string>1.0</string>
  <key>CFBundleIconFile</key>        <string>AppIcon</string>
  <key>LSMinimumSystemVersion</key>  <string>13.0</string>
  <key>NSHighResolutionCapable</key> <true/>
</dict>
</plist>
EOF

# ── Launcher script ───────────────────────────────────────────────────────────
cat > "$APP/Contents/MacOS/atlas-rag" << LAUNCHER
#!/bin/bash
cd "$REPO"
exec "$PYTHON" "$REPO/desktop.py"
LAUNCHER

chmod +x "$APP/Contents/MacOS/atlas-rag"

# ── Icon ──────────────────────────────────────────────────────────────────────
ICON="$REPO/AppIcon.icns"
if [ ! -f "$ICON" ]; then
  echo "Generating icon..."
  cd "$REPO"
  "$PYTHON" make_icon.py
fi
cp "$ICON" "$APP/Contents/Resources/AppIcon.icns"

echo ""
echo "✓ Atlas RAG.app created on Desktop"
echo "  Double-click it to launch."
