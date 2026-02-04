import os

STOCK_SYMBOLS = ["AAPL", "GOOGL", "MSFT"]  # You can add more symbols
PRICE_CHANGE_THRESHOLD = 5.0  # percent
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
