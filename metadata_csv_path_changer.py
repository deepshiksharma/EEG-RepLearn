import csv
from pathlib import PureWindowsPath, PurePosixPath


OLD_PREFIX = r"C:\\"
NEW_PREFIX = "/media/govtech/Disk 1/EEG_representation_learning"
INPUT_CSV = "/media/govtech/Disk 1/EEG_representation_learning/TUH-EEG-8S/PT_val/metadata.csv"
OUTPUT_CSV = "metadata_new.csv"


def convert_path(path_str):
    p = PureWindowsPath(path_str)

    if not str(p).startswith(OLD_PREFIX):
        raise ValueError(f'Unexpected path prefix: {path_str}')

    relative = str(p)[len(OLD_PREFIX):]
    return str(PurePosixPath(NEW_PREFIX) / PurePosixPath(relative))


rows_read = 0
rows_written = 0

with open(INPUT_CSV, newline='', encoding='utf-8') as infile, \
     open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as outfile:

    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)

    writer.writeheader()

    for row in reader:
        rows_read += 1

        row['filepath'] = convert_path(row['filepath'])
        writer.writerow(row)

        rows_written += 1

print(f'Rows read: {rows_read}')
print(f'Rows written: {rows_written}')

if rows_read != rows_written:
    print('WARNING: mismatch between rows read and written')
