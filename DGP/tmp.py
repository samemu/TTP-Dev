from datetime import datetime, timedelta


def get_date(row, interval=2):
    start = datetime(2015, 01, 01, 00, 00)
    return (start + timedelta(minutes=row * interval)).strftime("%Y-%m-%d %H:%M")

print get_date(50)