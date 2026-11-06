#!/bin/bash
# 在 Mac 上執行：bash build_mac.sh   → 產生「台股訊號.app」，拖進「應用程式」資料夾即可。
# 在自己的 Mac 上建置的 App 不會被 Gatekeeper 擋；首次開啟會自動建立 Python 環境（需 Python 3.10+）。
set -e
cd "$(dirname "$0")"
APP="台股訊號.app"; RES="$APP/Contents/Resources"
rm -rf "$APP"; mkdir -p "$APP/Contents/MacOS" "$RES/taiex_signal"
cp app.py desktop.py export_html.py pipeline.py news.py stock.py dashboard.html manifest.webmanifest requirements.txt \
   icon-192.png icon-512.png apple-touch-icon.png signals.json weekly.json news.json "$RES/taiex_signal/"
if command -v sips >/dev/null && command -v iconutil >/dev/null; then
  I="$(mktemp -d)/icon.iconset"; mkdir -p "$I"
  for s in 16 32 128 256 512; do
    sips -z $s $s icon-1024.png --out "$I/icon_${s}x${s}.png" >/dev/null
    sips -z $((s*2)) $((s*2)) icon-1024.png --out "$I/icon_${s}x${s}@2x.png" >/dev/null
  done
  iconutil -c icns "$I" -o "$RES/icon.icns"
fi
cat > "$APP/Contents/MacOS/launch" <<'EOS'
#!/bin/bash
R="$(cd "$(dirname "$0")/../Resources/taiex_signal" && pwd)"
D="$HOME/Library/Application Support/TaiexSignal"; mkdir -p "$D"
PY="$(command -v python3 || true)"
if [ -z "$PY" ]; then
  osascript -e 'display dialog "需要先安裝 Python 3.10 以上（python.org），再重新開啟。" buttons {"好"}'; exit 1
fi
if [ ! -x "$D/venv/bin/python" ]; then
  osascript -e 'display notification "首次啟動，正在安裝元件，約需數分鐘" with title "台股訊號"'
  "$PY" -m venv "$D/venv" && "$D/venv/bin/pip" install -q -r "$R/requirements.txt" pywebview
fi
export TAIEX_DATA="$D/data"
exec "$D/venv/bin/python" "$R/desktop.py"
EOS
chmod +x "$APP/Contents/MacOS/launch"
cat > "$APP/Contents/Info.plist" <<'EOP'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleName</key><string>台股訊號</string>
<key>CFBundleDisplayName</key><string>台股訊號</string>
<key>CFBundleIdentifier</key><string>local.taiex.signal</string>
<key>CFBundleExecutable</key><string>launch</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleVersion</key><string>1.0</string>
<key>CFBundleIconFile</key><string>icon</string>
<key>LSMinimumSystemVersion</key><string>11.0</string>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>
EOP
echo "完成：$APP"
