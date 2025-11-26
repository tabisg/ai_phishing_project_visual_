import re
from urllib.parse import urlparse
import tldextract

IP_RE = re.compile(r'^(?:\d{1,3}\.){3}\d{1,3}$')

def extract_url_features(url: str) -> dict:
    url = url.strip()
    if not url:
        return {}
    parsed = urlparse(url if '://' in url else 'http://' + url)
    domain = parsed.netloc or parsed.path
    # remove credentials if present
    if '@' in domain:
        domain = domain.split('@')[-1]
    te = tldextract.extract(domain)
    host = domain.lower()
    path = parsed.path or ""
    query = parsed.query or ""
    features = {}
    features['url_len'] = len(url)
    features['host_len'] = len(host)
    features['path_len'] = len(path)
    features['count_dots'] = host.count('.')
    features['count_hyphens'] = host.count('-') + path.count('-')
    features['has_at'] = 1 if '@' in url else 0
    features['has_ip'] = 1 if IP_RE.match(te.domain) or IP_RE.match(host) else 0
    features['has_query'] = 1 if query else 0
    features['num_subdirs'] = path.count('/')
    features['is_https'] = 1 if parsed.scheme == 'https' else 0
    features['sld_len'] = len(te.domain)
    features['tld'] = te.suffix or ''
    return features
