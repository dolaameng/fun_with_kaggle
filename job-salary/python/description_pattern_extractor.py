## The main purpose of this script is to extract certain sub-texts
## from the full description file of train and test (../data/descriptions.csv)
## because the full description file is too large to parse (at least on my machine)

## the extracted texts are Capitilized words for now (URL, emails)

import csv, re, sys, os.path

DESCRIPTION_FILE = '../data/descriptions.csv'
SIMPLE_FILE = '../data/simple_descriptions.csv'

def main():
    reader = csv.reader(open(DESCRIPTION_FILE))
    writer = csv.writer(open(SIMPLE_FILE, 'w'))
    pattern = re.compile(r'\b[A-Z]\w+\b')
    writer.writerow(reader.next()) ## header
    for text in reader:
        pattern_text = set(pattern.findall(text[0])) ## ignore term frequence - should be OK
        writer.writerow([' '.join(pattern_text)])

if __name__ == '__main__':
    main()