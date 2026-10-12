import re

with open('04_web/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace("fetch('/set_ip'", "fetch(`${API_BASE}/set_ip`")
content = content.replace("fetch(`/control", "fetch(`${API_BASE}/control")
content = content.replace("fetch('/capture'", "fetch(`${API_BASE}/capture`")
content = content.replace("fetch('/api/camera/status')", "fetch(`${API_BASE}/api/camera/status`)")

# Let's fix the src of the video feed as well
content = content.replace("img.src = '/video_feed?t='", "img.src = `${API_BASE}/video_feed?t=`")
content = content.replace('src="/video_feed"', 'src="/video_feed"') # Wait, for the HTML image tag it's fine without API_BASE if served by the same host.

with open('04_web/index.html', 'w', encoding='utf-8') as f:
    f.write(content)
    
print("Replaced API_BASE!")
