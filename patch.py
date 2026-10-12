import re

with open('04_web/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

replacement = """          <div id="liveFeedContainer" style="margin-top:12px; margin-bottom:12px; background: rgba(7, 17, 29, 0.55); border: 1px solid var(--line); border-radius: 9px; padding: 15px;">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
              <span style="font-size:11px;font-weight:600;color:var(--cyan);text-transform:uppercase;letter-spacing:0.5px;">ESP32-CAM Live Feed</span>
              <span id="feedStatus" style="font-size:10px;color:var(--amber);font-weight:600;">● OFFLINE</span>
            </div>
            
            <div style="display:flex; gap: 8px; margin-bottom: 12px;">
                <input type="text" id="espIpInput" placeholder="ESP32 IP (e.g. http://192.168.1.100)" style="flex:1; border:1px solid var(--line); border-radius:6px; padding:6px 8px; font-size:11px; background:var(--surface2); color:var(--text)">
                <button type="button" onclick="setEspIp()" style="background:var(--cyan); color:#06121e; border:none; border-radius:6px; padding:6px 12px; font-weight:700; font-size:11px;">Connect</button>
            </div>

            <img id="lookcamFeed" src="/video_feed" style="width:100%; border-radius:8px; border:1px solid var(--line); display:block; min-height:200px; max-height:360px; object-fit:contain; background:#000;" alt="ESP32-CAM Live Feed">
            
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px;">
                <div>
                    <label style="font-size:10px; color:var(--muted); display:block; margin-bottom:4px;">Resolution</label>
                    <select onchange="updateEspControl('framesize', this.value)" style="width:100%; background:var(--surface2); color:var(--text); border:1px solid var(--line); border-radius:4px; padding:4px;">
                        <option value="10">UXGA</option>
                        <option value="9">SXGA</option>
                        <option value="8">XGA</option>
                        <option value="6" selected>VGA</option>
                        <option value="4">QVGA</option>
                    </select>
                </div>
                <div>
                    <label style="font-size:10px; color:var(--muted); display:block; margin-bottom:4px;">LED Flash</label>
                    <input type="range" min="0" max="255" value="0" onchange="updateEspControl('led_intensity', this.value)" style="width:100%;">
                </div>
                <div>
                    <label style="font-size:10px; color:var(--muted); display:block; margin-bottom:4px;">V-Flip</label>
                    <select onchange="updateEspControl('vflip', this.value)" style="width:100%; background:var(--surface2); color:var(--text); border:1px solid var(--line); border-radius:4px; padding:4px;">
                        <option value="0">Off</option><option value="1">On</option>
                    </select>
                </div>
                <div>
                    <label style="font-size:10px; color:var(--muted); display:block; margin-bottom:4px;">Quality (low=better)</label>
                    <input type="range" min="10" max="63" value="10" onchange="updateEspControl('quality', this.value)" style="width:100%;">
                </div>
            </div>
            
            <div style="margin-top: 12px; text-align: center;">
                <button type="button" onclick="captureEspHighRes()" style="background:#4ab6ff; color:#06121e; border:none; border-radius:6px; padding:6px 12px; font-weight:700; font-size:11px; width: 100%;">📸 Capture High-Res Image to Disk</button>
            </div>
          </div>
          <script>
          (function() {
            var status = document.getElementById('feedStatus');
            var img = document.getElementById('lookcamFeed');
            function checkFeed() {
              fetch('/api/camera/status').then(r => r.json()).then(function(data) {
                if (data.connected && data.has_frame) {
                  status.textContent = '● LIVE (' + data.configured_ip + ')';
                  status.style.color = 'var(--green)';
                  if (!img.src || img.src.indexOf('/video_feed') === -1) {
                    img.src = '/video_feed?t=' + Date.now();
                  }
                } else {
                  status.textContent = '● OFFLINE (' + data.last_error + ')';
                  status.style.color = 'var(--red)';
                }
                setTimeout(checkFeed, 3000);
              }).catch(function() {
                status.textContent = '● OFFLINE';
                status.style.color = 'var(--red)';
                setTimeout(checkFeed, 3000);
              });
            }
            img.onerror = function() {
              setTimeout(function() {
                img.src = '/video_feed?t=' + Date.now();
              }, 2000);
            };
            setTimeout(checkFeed, 2000);
          })();

          function setEspIp() {
              const ip = document.getElementById('espIpInput').value;
              fetch('/set_ip', {
                  method: 'POST',
                  headers: {'Content-Type': 'application/json'},
                  body: JSON.stringify({ip: ip})
              }).then(r => r.json()).then(data => {
                  if(data.status === 'success') {
                      alert("Connecting to ESP32: " + data.ip);
                      document.getElementById('lookcamFeed').src = '/video_feed?t=' + Date.now();
                  } else {
                      alert("Error setting IP");
                  }
              });
          }

          function updateEspControl(varName, value) {
              fetch(`/control?var=${varName}&val=${value}`)
                .then(r => r.json())
                .then(data => { if(data.error) alert("Error: " + data.error); });
          }
          
          function captureEspHighRes() {
              fetch('/capture')
                .then(r => r.json())
                .then(data => {
                    if (data.status === 'success') {
                        alert(data.message);
                    } else {
                        alert("Capture error: " + data.error);
                    }
                });
          }
          </script>"""

pattern = re.compile(r'          <div id="liveFeedContainer".*?          </script>', re.DOTALL)
new_content = pattern.sub(replacement, content)

with open('04_web/index.html', 'w', encoding='utf-8') as f:
    f.write(new_content)
    
print("Replaced!")
