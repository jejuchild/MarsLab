#!/bin/bash
# Start Ollama with model kept in memory forever
# Usage: ./start_ollama.sh

# Kill existing if running
pkill -f "ollama serve" 2>/dev/null
sleep 2

# Start with keep-alive forever (model stays in RAM)
export OLLAMA_KEEP_ALIVE=-1
nohup ollama serve > /tmp/ollama_serve.log 2>&1 &
sleep 3

echo "Ollama started. Pre-loading llama3.3..."

# Pre-load the model into memory
python3 -c "
import urllib.request, json, sys
data = json.dumps({'model':'llama3.3','prompt':'warmup','stream':False,'keep_alive':-1}).encode()
req = urllib.request.Request('http://localhost:11434/api/generate', data=data, headers={'Content-Type':'application/json'})
try:
    resp = urllib.request.urlopen(req, timeout=300)
    r = json.loads(resp.read())
    print(f'Model loaded in {r.get(\"load_duration\",0)/1e9:.1f}s')
except Exception as e:
    print(f'Warning: {e}')
    sys.exit(1)
"

echo ""
ollama ps
echo ""
echo "Llama3.3 is loaded and will stay in memory (OLLAMA_KEEP_ALIVE=-1)"
