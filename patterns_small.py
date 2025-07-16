patterns = {
    'Email': {'pattern': r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', 'token': '[EMAIL]'},
    'URL_HTTP': {'pattern': r'https?://[^\s<>"{}|\\^`\[\]]+', 'token': '[URL]'},
    'USER_AGENT_STRING': {'pattern': r'(?:Mozilla|Opera|Chrome|Safari|Edge|Firefox|MSIE|Trident|Googlebot|Bingbot|Slurp|DuckDuckBot|YandexBot|curl|Wget|PostmanRuntime|Dalvik|okhttp|AppleWebKit|python-requests|Java)/[a-zA-Z0-9\s\(\)\;\.\/\-\_:,rv=x_]+[a-zA-Z0-9\/]', 'token': '[AGENT]'},
}
