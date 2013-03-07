## feature geneartion strategy 1: basic santanizing data
## STRATEGY: 
## id (keep), Title (keep), FullDescription (keep), LocationRaw ()

import json, sys
import os.path

def main():
    setting = json.load(open('settings.json'))['feature_generators']['fg_1']
    in_path, out_path = setting['input'], setting['output']
    if os.path.exists(out_path):
        print 'extracted feature file', out_path, 'already exists. delete or backup it first'
        sys.exit(-1)

if __name__ == '__main__':
    main()