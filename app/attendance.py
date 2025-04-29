import csv
from datetime import datetime

def mark_attendance(name, log_path='attendance_logs/logs.csv'):
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, time])
