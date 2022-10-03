import datetime

def cumulative_distinct_values(series):
    return (~series.duplicated()).cumsum()

def last_day_of_month(d: datetime.date) -> datetime.date:
    
        date = datetime.date(d.year + d.month//12, d.month % 12 + 1, 1) - datetime.timedelta(days=1)
        return datetime.datetime(date.year, date.month, date.day)
